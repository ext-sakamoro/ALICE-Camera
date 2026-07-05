//! Full ISP pipeline.

use crate::bayer::RawImage;
use crate::demosaic::demosaic_bilinear;
use crate::denoise::{denoise_box_blur, denoise_median};
use crate::distortion::{correct_distortion, DistortionCoeffs};
use crate::exposure::{apply_exposure, auto_exposure_ev, ExposureParams};
use crate::gamma::{apply_gamma, apply_srgb_gamma};
use crate::histogram::histogram_equalize;
use crate::image::Image;
use crate::white_balance::{apply_white_balance, grey_world_white_balance, WhiteBalanceGains};

/// Configuration for the full ISP pipeline.
#[derive(Debug, Clone)]
pub struct IspConfig {
    pub white_balance: Option<WhiteBalanceGains>,
    pub auto_white_balance: bool,
    pub ev_offset: Option<f32>,
    pub auto_exposure_target: Option<f32>,
    pub distortion: Option<DistortionCoeffs>,
    pub histogram_equalize: bool,
    pub denoise: DenoiseMethod,
    pub gamma: GammaMode,
}

/// Denoising method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethod {
    None,
    BoxBlur,
    Median,
}

/// Gamma mode selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GammaMode {
    None,
    Power(f32),
    Srgb,
}

impl Default for IspConfig {
    fn default() -> Self {
        Self {
            white_balance: None,
            auto_white_balance: true,
            ev_offset: None,
            auto_exposure_target: Some(0.18),
            distortion: None,
            histogram_equalize: false,
            denoise: DenoiseMethod::None,
            gamma: GammaMode::Srgb,
        }
    }
}

/// Run the full ISP pipeline: demosaic -> white balance -> exposure -> distortion -> denoise -> histogram eq -> gamma.
#[must_use]
pub fn run_isp_pipeline(raw: &RawImage, config: &IspConfig) -> Image {
    let mut img = demosaic_bilinear(raw);

    if let Some(gains) = config.white_balance {
        apply_white_balance(&mut img, gains);
    } else if config.auto_white_balance {
        let gains = grey_world_white_balance(&img);
        apply_white_balance(&mut img, gains);
    }

    if let Some(ev) = config.ev_offset {
        apply_exposure(&mut img, ExposureParams { ev_offset: ev });
    } else if let Some(target) = config.auto_exposure_target {
        let ev = auto_exposure_ev(&img, target);
        apply_exposure(&mut img, ExposureParams { ev_offset: ev });
    }

    if let Some(coeffs) = config.distortion {
        img = correct_distortion(&img, coeffs);
    }

    match config.denoise {
        DenoiseMethod::None => {}
        DenoiseMethod::BoxBlur => denoise_box_blur(&mut img),
        DenoiseMethod::Median => denoise_median(&mut img),
    }

    if config.histogram_equalize {
        histogram_equalize(&mut img);
    }

    match config.gamma {
        GammaMode::None => {}
        GammaMode::Power(g) => apply_gamma(&mut img, g),
        GammaMode::Srgb => apply_srgb_gamma(&mut img),
    }

    img.clamp_all();
    img
}
