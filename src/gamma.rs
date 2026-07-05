//! Gamma correction.

use crate::image::Image;

/// Apply gamma correction (power curve).
pub fn apply_gamma(img: &mut Image, gamma: f32) {
    let inv = 1.0 / gamma;
    for p in &mut img.pixels {
        p.r = p.r.max(0.0).powf(inv);
        p.g = p.g.max(0.0).powf(inv);
        p.b = p.b.max(0.0).powf(inv);
    }
}

/// Apply sRGB gamma curve (linear-to-sRGB).
pub fn apply_srgb_gamma(img: &mut Image) {
    for p in &mut img.pixels {
        p.r = linear_to_srgb(p.r);
        p.g = linear_to_srgb(p.g);
        p.b = linear_to_srgb(p.b);
    }
}

/// Apply inverse sRGB gamma curve (sRGB-to-linear).
pub fn apply_srgb_degamma(img: &mut Image) {
    for p in &mut img.pixels {
        p.r = srgb_to_linear(p.r);
        p.g = srgb_to_linear(p.g);
        p.b = srgb_to_linear(p.b);
    }
}

pub(crate) fn linear_to_srgb(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

pub(crate) fn srgb_to_linear(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
