#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{path::PathBuf, time::Instant};

use clap::Parser;
use dfdx::{data::*, nn::Module, optim::Adam, prelude::*};
use eyre::Result;
use indicatif::{ProgressIterator, ProgressStyle};
use rand::{rngs::StdRng, SeedableRng};

mod mnist;

const BATCH_SIZE: usize = 64;

type Device = AutoDevice;
type Dtype = f32;

type Cnn = (
    (Conv2D<1, 16, 5, 1, 2>, ReLU, MaxPool2D<2, 2>),
    (Conv2D<16, 32, 5, 1, 2>, ReLU, MaxPool2D<2, 2>),
    Flatten2D,
    Linear<1568, 10>,
);
type Model = <Cnn as BuildOnDevice<Device, Dtype>>::Built;
type Dataset = mnist::Mnist;

#[derive(Parser)]
struct Args {
    #[clap(long)]
    /// Path to a checkpoint to load, if any.
    checkpoint: Option<PathBuf>,

    #[clap(long)]
    /// Debug mode.
    debug: bool,

    #[clap(subcommand)]
    subcommand: Subcommand,
}

#[derive(clap::Subcommand)]
enum Subcommand {
    /// Train a model.
    Train {
        #[clap(flatten)]
        args: TrainArgs,
    },

    /// Test a model.
    Test,
}

#[derive(Parser, Debug)]
struct TrainArgs {
    #[clap(long, default_value = "3")]
    /// Number of epoches to train for.
    epoches: usize,

    #[clap(long, default_value = "0")]
    epoch_start: usize,

    #[clap(long, default_value = "0.01")]
    /// Learning rate.
    lr: f64,
}

fn train(
    model: &mut Model,
    dataset: &Dataset,
    dataset_test: &Dataset,
    device: &Device,
    rng: &mut StdRng,
    args: TrainArgs,
) -> Result<()> {
    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(
        model,
        AdamConfig {
            lr: args.lr,
            ..Default::default()
        },
    );

    for i_epoch in args.epoch_start..args.epoch_start + args.epoches {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (img, lbl) in dataset
            .shuffled(rng)
            .map(|(img, lbl)| {
                (
                    mnist::Mnist::preprocess_image(img, device),
                    mnist::Mnist::preprocess_label(lbl, device),
                )
            })
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress_with_style(ProgressStyle::default_bar().template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} eta {eta_precise}",
            )?)
        {
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);

            total_epoch_loss += loss.array();
            num_batches += 1;

            grads = loss.backward();
            opt.update(model, &grads)?;
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

        let test_accuracy = test(model, dataset_test, device)?;

        log::info!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}, test accuracy {:.3}%",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            100.0 * test_accuracy,
        );

        model.save_safetensors(format!("./models/model_{:03}.safetensors", i_epoch))?;
    }
    Ok(())
}

fn test(model: &Model, dataset: &Dataset, device: &Device) -> Result<f32> {
    let mut num_correct = 0;
    for (img, lbl) in dataset
        .iter()
        .map(|(img, lbl)| (mnist::Mnist::preprocess_image(img, device), lbl))
        .batch_exact(Const::<BATCH_SIZE>)
        .collate()
        .progress_with_style(ProgressStyle::default_bar().template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} eta {eta_precise}",
        )?)
    {
        let logits = model.forward(img.stack()).array();
        for (idx, logit) in logits.iter().enumerate() {
            if argmax(logit) == lbl[idx] {
                num_correct += 1;
            }
        }
    }
    Ok(num_correct as f32 / dataset.len() as f32)
}

fn main() -> Result<()> {
    let args = Args::parse();

    env_logger::Builder::new()
        .filter_level(if !args.debug {
            log::LevelFilter::Info
        } else {
            log::LevelFilter::Debug
        })
        .init();

    log::info!("Initializing device");
    let device = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(0);
    log::info!("Device: {:?}", device);

    log::info!("Creating model");
    let mut model = device.build_module::<Cnn, f32>();

    if let Some(checkpoint) = args.checkpoint {
        log::info!("Loading model from checkpoint {:?}", checkpoint);
        model
            .load_safetensors(checkpoint)
            .map_err(|e| eyre::eyre!("Failed to load checkpoint: {:?}", e))?;
    }

    match args.subcommand {
        Subcommand::Train { args } => {
            log::info!("Loading data");
            let dataset = mnist::Mnist::load(
                "./datasets/mnist_data/train-images-idx3-ubyte.gz",
                "./datasets/mnist_data/train-labels-idx1-ubyte.gz",
                60000,
            )?;
            let dataset_test = mnist::Mnist::load(
                "./datasets/mnist_data/t10k-images-idx3-ubyte.gz",
                "./datasets/mnist_data/t10k-labels-idx1-ubyte.gz",
                10000,
            )?;

            log::info!("Training with args {:?}", args);
            train(&mut model, &dataset, &dataset_test, &device, &mut rng, args)?;
        }
        Subcommand::Test => {
            log::info!("Loading data");
            let dataset_test = mnist::Mnist::load(
                "./datasets/mnist_data/t10k-images-idx3-ubyte.gz",
                "./datasets/mnist_data/t10k-labels-idx1-ubyte.gz",
                10000,
            )?;

            log::info!("Testing");
            let test_accuracy = test(&model, &dataset_test, &device)?;
            log::info!("Test accuracy: {:.3}%", 100.0 * test_accuracy);
        }
    }

    Ok(())
}

/// Returns the index of the maximum element in `data`.
///
/// # Panics
/// Panics if `data` is empty.
fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .reduce(|(i_max, x_max), (i, x)| if x > x_max { (i, x) } else { (i_max, x_max) })
        .unwrap()
        .0
}
