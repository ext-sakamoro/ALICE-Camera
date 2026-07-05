//! Lens distortion correction + bilinear sampling.

use crate::image::{Image, Rgb};

/// Radial distortion coefficients (Brown-Conrady model).
#[derive(Debug, Clone, Copy)]
pub struct DistortionCoeffs {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
}

impl DistortionCoeffs {
    #[must_use]
    pub const fn new(k1: f32, k2: f32, k3: f32) -> Self {
        Self { k1, k2, k3 }
    }

    /// No distortion.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
        }
    }
}

/// Correct radial lens distortion.
#[must_use]
pub fn correct_distortion(img: &Image, coeffs: DistortionCoeffs) -> Image {
    let w = img.width;
    let h = img.height;
    let mut out = Image::new(w, h);
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_r = cx.hypot(cy);

    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / max_r;
            let dy = (y as f32 - cy) / max_r;
            let r2 = dx.mul_add(dx, dy * dy);
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let factor = coeffs
                .k1
                .mul_add(r2, coeffs.k2.mul_add(r4, coeffs.k3.mul_add(r6, 1.0)));

            let src_x = (dx * factor).mul_add(max_r, cx);
            let src_y = (dy * factor).mul_add(max_r, cy);

            out.set(x, y, bilinear_sample(img, src_x, src_y));
        }
    }
    out
}

/// Bilinear sampling from an image at fractional coordinates.
pub(crate) fn bilinear_sample(img: &Image, fx: f32, fy: f32) -> Rgb {
    let w = img.width;
    let h = img.height;
    let x0 = (fx.floor() as i32).clamp(0, w as i32 - 1) as usize;
    let y0 = (fy.floor() as i32).clamp(0, h as i32 - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = (fx - fx.floor()).clamp(0.0, 1.0);
    let ty = (fy - fy.floor()).clamp(0.0, 1.0);

    let c00 = img.get(x0, y0);
    let c10 = img.get(x1, y0);
    let c01 = img.get(x0, y1);
    let c11 = img.get(x1, y1);

    let lerp = |a: f32, b: f32, t: f32| (b - a).mul_add(t, a);
    Rgb::new(
        lerp(lerp(c00.r, c10.r, tx), lerp(c01.r, c11.r, tx), ty),
        lerp(lerp(c00.g, c10.g, tx), lerp(c01.g, c11.g, tx), ty),
        lerp(lerp(c00.b, c10.b, tx), lerp(c01.b, c11.b, tx), ty),
    )
}
