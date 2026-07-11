# mlx-rs

The code is organized into multiple crates in a Rust workspace:
- crates/mlx-internal-macros
- crates/mlx-macros
- crates/mlx-rs
- crates/mlx-sys

## Overview

MLX is an array framework for machine learning on Apple silicon. mlx-rs provides Rust
bindings for MLX.

## IMPORTANT

- Dependencies MUST be declared in the workspace Cargo.toml, and used in subcrates with
  `<crate>.workspace = true`
- READ THE DOCS. Rust libs can be found at https://libs.rs. Docs can be found at
  https://docs.rs. YOU DO NOT NEED TO BUILD DOCS.
- Prefer nested submodules with `mod.rs` files at depth of 1. Submodules deeper than 1 may
  be siblings to `mod.rs`.

## Commands

- `cargo fmt`
- `cargo nextest run` - always run all tests for all crates
- `cargo clippy -- -Dwarnings`
