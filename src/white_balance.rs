//! White balance: `WhiteBalanceGains` + `apply_white_balance` + `grey_world_white_balance`.

use crate::image::Image;

/// White balance gains.
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalanceGains {
    pub r_gain: f32,
    pub g_gain: f32,
    pub b_gain: f32,
}

impl WhiteBalanceGains {
    #[must_use]
    pub const fn new(r_gain: f32, g_gain: f32, b_gain: f32) -> Self {
        Self {
            r_gain,
            g_gain,
            b_gain,
        }
    }
}

/// Apply white balance gains to an image (in-place).
pub fn apply_white_balance(img: &mut Image, gains: WhiteBalanceGains) {
    for p in &mut img.pixels {
        p.r *= gains.r_gain;
        p.g *= gains.g_gain;
        p.b *= gains.b_gain;
    }
}

/// Estimate white balance gains using the grey-world algorithm.
#[must_use]
pub fn grey_world_white_balance(img: &Image) -> WhiteBalanceGains {
    let n = img.pixels.len() as f32;
    if n == 0.0 {
        return WhiteBalanceGains::new(1.0, 1.0, 1.0);
    }
    let (mut sr, mut sg, mut sb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for p in &img.pixels {
        sr += p.r;
        sg += p.g;
        sb += p.b;
    }
    let avg_r = sr / n;
    let avg_g = sg / n;
    let avg_b = sb / n;
    let avg_all = (avg_r + avg_g + avg_b) / 3.0;
    let safe = |v: f32| if v.abs() < 1e-9 { 1.0 } else { avg_all / v };
    WhiteBalanceGains::new(safe(avg_r), safe(avg_g), safe(avg_b))
}
