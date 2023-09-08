{
  description = "try dfdx";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";

    crane = {
      url = "github:ipetkov/crane";
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

outputs = { self, nixpkgs, fenix, flake-utils, crane, ... }:
    let
      inherit (nixpkgs.lib) optionals;
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        fenixPkgs = fenix.packages.${system};
        pkgs = nixpkgs.legacyPackages.${system};

        toolchain = fenixPkgs.combine [
          fenixPkgs.complete.cargo
          fenixPkgs.complete.clippy
          fenixPkgs.complete.rustc
          fenixPkgs.complete.rustfmt
          fenixPkgs.complete.rust-std
          fenixPkgs.targets.x86_64-unknown-linux-musl.latest.rust-std
        ];
        craneLib = crane.lib.${system}.overrideToolchain toolchain;
      in
      let
        commonArgs = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          buildInputs = [ ] ++ optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
        };
        cargoArtifacts = craneLib.buildDepsOnly (commonArgs // {
          pname = "try-dfdx-deps";
        });
        cargoClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--all-targets -- --deny warnings";
        });
        cargoPackage = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      rec {
        packages.default = cargoPackage;
        checks = { inherit cargoPackage cargoClippy; };

        devShells.default = pkgs.mkShell {
          packages = [
            toolchain
            fenixPkgs.rust-analyzer
            pkgs.cargo-expand
            pkgs.cargo-bloat

            pkgs.python310
            pkgs.python310Packages.requests
            pkgs.python310Packages.tqdm
            pkgs.python310Packages.numpy
          ];

          buildInputs = optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.CoreServices
            pkgs.libiconv
          ];

          RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
        };

        devShells.msrv = pkgs.mkShell {
          packages = [
            pkgs.cargo
            pkgs.rustup
            pkgs.cargo-msrv
          ];
        };
      });
}
