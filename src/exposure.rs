//! Exposure control.

use crate::image::Image;

/// Parameters for exposure adjustment.
#[derive(Debug, Clone, Copy)]
pub struct ExposureParams {
    /// Exposure value offset (stops). Positive = brighter.
    pub ev_offset: f32,
}

/// Apply exposure compensation (multiply by 2^ev).
pub fn apply_exposure(img: &mut Image, params: ExposureParams) {
    let factor = params.ev_offset.exp2();
    for p in &mut img.pixels {
        p.r *= factor;
        p.g *= factor;
        p.b *= factor;
    }
}

/// Compute average luminance of the image.
#[must_use]
pub fn average_luminance(img: &Image) -> f32 {
    if img.pixels.is_empty() {
        return 0.0;
    }
    let sum: f32 = img.pixels.iter().map(|p| p.luminance()).sum();
    sum / img.pixels.len() as f32
}

/// Suggest an EV offset to bring average luminance to a target (default 0.18 mid-grey).
#[must_use]
pub fn auto_exposure_ev(img: &Image, target: f32) -> f32 {
    let avg = average_luminance(img);
    if avg < 1e-9 {
        return 0.0;
    }
    (target / avg).log2()
}
