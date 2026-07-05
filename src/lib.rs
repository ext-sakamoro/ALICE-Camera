#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

//! ALICE-Camera: Camera capture and ISP (Image Signal Processing)
//!
//! Pure Rust implementation of core ISP pipeline stages:
//! - White balance
//! - Demosaicing (Bayer pattern)
//! - Exposure control
//! - Auto-focus metrics
//! - Lens distortion correction
//! - Histogram equalization
//! - Noise reduction
//! - HDR merge
//! - Gamma correction

pub mod bayer;
pub mod demosaic;
pub mod denoise;
pub mod distortion;
pub mod exposure;
pub mod focus;
pub mod gamma;
pub mod hdr;
pub mod histogram;
pub mod image;
pub mod pipeline;
pub mod prelude;
pub mod white_balance;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::bayer::*;
pub use crate::demosaic::*;
pub use crate::denoise::*;
pub use crate::distortion::*;
pub use crate::exposure::*;
pub use crate::focus::*;
pub use crate::gamma::*;
pub use crate::hdr::*;
pub use crate::histogram::*;
pub use crate::image::*;
pub use crate::pipeline::*;
pub use crate::white_balance::*;
