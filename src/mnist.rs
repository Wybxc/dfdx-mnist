use std::path::Path;

use dfdx::{data::ExactSizeDataset, prelude::*};
use eyre::Result;

#[derive(Debug)]
pub struct Mnist {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
}

impl Mnist {
    pub fn load<P: AsRef<Path>>(images: P, labels: P, data_count: u32) -> Result<Self> {
        log::debug!(
            "Loading MNIST data from {:?} and {:?}",
            images.as_ref(),
            labels.as_ref()
        );

        let images = read_mnist_data(images.as_ref(), false, data_count)?;
        log::debug!("Loaded MNIST images, {} bytes", images.len());

        let labels = read_mnist_data(labels.as_ref(), true, data_count)?;
        log::debug!("Loaded MNIST labels, {} bytes", labels.len());

        Ok(Self { images, labels })
    }

    pub fn preprocess_image<E, D>(img: Vec<E>, device: &D) -> Tensor<Rank3<1, 28, 28>, E, D>
    where
        E: Dtype,
        D: Device<E>,
    {
        device.tensor_from_vec(img, Rank3::<1, 28, 28>::default())
    }

    pub fn preprocess_label<E, D>(lbl: usize, device: &D) -> Tensor<Rank1<10>, E, D>
    where
        E: Dtype,
        D: Device<E>,
    {
        let mut one_hotted = [E::from_f32(0.0).unwrap(); 10];
        one_hotted[lbl] = E::from_f32(1.0).unwrap();
        device.tensor(one_hotted)
    }
}

impl ExactSizeDataset for Mnist {
    type Item<'a> = (Vec<f32>, usize)
    where
        Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        let start = 784 * index;
        img_data.extend(
            self.images[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );
        (img_data, self.labels[index] as usize)
    }

    fn len(&self) -> usize {
        self.labels.len()
    }
}

fn read_mnist_data(filename: &Path, is_label: bool, data_count: u32) -> Result<Vec<u8>> {
    use byteorder::ReadBytesExt;
    use std::io::Read;

    let file = std::fs::File::open(filename)?;
    let gz_decoder = flate2::read::GzDecoder::new(file);
    let mut reader = std::io::BufReader::new(gz_decoder);

    // Read the magic number
    let magic = reader.read_u32::<byteorder::BigEndian>()?;
    check_eq(
        magic,
        if is_label { 0x00000801 } else { 0x00000803 },
        "Invalid magic number",
    )?;

    // Read the count
    let count = reader.read_u32::<byteorder::BigEndian>()?;
    check_eq(count, data_count, "Invalid count")?;

    // Read the rows and cols
    if !is_label {
        let rows = reader.read_u32::<byteorder::BigEndian>()?;
        check_eq(rows, 28, "Invalid rows")?;

        let cols = reader.read_u32::<byteorder::BigEndian>()?;
        check_eq(cols, 28, "Invalid cols")?;
    }

    // Read the data
    let mut data = Vec::with_capacity((count * (if is_label { 1 } else { 28 * 28 })) as usize);
    reader.read_to_end(&mut data)?;

    Ok(data)
}

fn check_eq<T: Eq + Copy>(a: T, b: T, msg: &str) -> Result<()> {
    if a == b {
        Ok(())
    } else {
        Err(eyre::eyre!("{}", msg))
    }
}
