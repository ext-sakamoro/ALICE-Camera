//! HDR merge + Reinhard tone-mapping.

use crate::image::{Image, Rgb};

/// Merge multiple exposures into an HDR image using exposure-weighted averaging.
///
/// # Panics
/// Panics if `images` is empty or images have mismatched dimensions.
#[must_use]
pub fn hdr_merge(images: &[(Image, f32)]) -> Image {
    assert!(!images.is_empty(), "need at least one exposure");
    let w = images[0].0.width;
    let h = images[0].0.height;
    for (img, _) in images {
        assert_eq!(img.width, w);
        assert_eq!(img.height, h);
    }

    let mut out = Image::new(w, h);

    for i in 0..w * h {
        let (mut sr, mut sg, mut sb, mut sw) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for (img, exposure) in images {
            let p = img.pixels[i];
            let lum = p.luminance().clamp(0.0, 1.0);
            let weight = 1.0 - 2.0f32.mul_add(lum, -1.0).abs();
            let weight = weight.max(0.01);

            let inv_exp = 1.0 / exposure;
            sr += p.r * inv_exp * weight;
            sg += p.g * inv_exp * weight;
            sb += p.b * inv_exp * weight;
            sw += weight;
        }
        if sw > 0.0 {
            out.pixels[i] = Rgb::new(sr / sw, sg / sw, sb / sw);
        }
    }
    out
}

/// Simple Reinhard tone-mapping for HDR images.
pub fn tonemap_reinhard(img: &mut Image) {
    for p in &mut img.pixels {
        p.r = p.r / (1.0 + p.r);
        p.g = p.g / (1.0 + p.g);
        p.b = p.b / (1.0 + p.b);
    }
}

/// Extended Reinhard tone-mapping with white point.
pub fn tonemap_reinhard_extended(img: &mut Image, white_point: f32) {
    let w2 = white_point * white_point;
    for p in &mut img.pixels {
        p.r = p.r * (1.0 + p.r / w2) / (1.0 + p.r);
        p.g = p.g * (1.0 + p.g / w2) / (1.0 + p.g);
        p.b = p.b * (1.0 + p.b / w2) / (1.0 + p.b);
    }
}
