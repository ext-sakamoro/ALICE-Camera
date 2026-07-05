//! Noise reduction: box blur / median / bilateral.

use crate::image::{Image, Rgb};

/// 3x3 box-blur noise reduction.
pub fn denoise_box_blur(img: &mut Image) {
    let w = img.width;
    let h = img.height;
    let src = img.pixels.clone();

    for y in 0..h {
        for x in 0..w {
            let mut sr = 0.0_f32;
            let mut sg = 0.0_f32;
            let mut sb = 0.0_f32;
            let mut count = 0_u32;
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let idx = ny as usize * w + nx as usize;
                        sr += src[idx].r;
                        sg += src[idx].g;
                        sb += src[idx].b;
                        count += 1;
                    }
                }
            }
            let inv = 1.0 / count as f32;
            img.set(x, y, Rgb::new(sr * inv, sg * inv, sb * inv));
        }
    }
}

/// 3x3 median filter noise reduction (per channel).
pub fn denoise_median(img: &mut Image) {
    let w = img.width;
    let h = img.height;
    let src = img.pixels.clone();

    for y in 0..h {
        for x in 0..w {
            let mut rs = Vec::with_capacity(9);
            let mut gs = Vec::with_capacity(9);
            let mut bs = Vec::with_capacity(9);
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let idx = ny as usize * w + nx as usize;
                        rs.push(src[idx].r);
                        gs.push(src[idx].g);
                        bs.push(src[idx].b);
                    }
                }
            }
            rs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            gs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            bs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let mid = rs.len() / 2;
            img.set(x, y, Rgb::new(rs[mid], gs[mid], bs[mid]));
        }
    }
}

/// Bilateral filter noise reduction.
pub fn denoise_bilateral(img: &mut Image, radius: i32, sigma_s: f32, sigma_r: f32) {
    let w = img.width;
    let h = img.height;
    let src = img.pixels.clone();
    let inv_2ss = -0.5 / (sigma_s * sigma_s);
    let inv_2sr = -0.5 / (sigma_r * sigma_r);

    for y in 0..h {
        for x in 0..w {
            let center = src[y * w + x];
            let cl = center.luminance();
            let (mut wr, mut wg, mut wb, mut wsum) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let neighbor = src[ny as usize * w + nx as usize];
                    let dist2 = (dx * dx + dy * dy) as f32;
                    let diff = neighbor.luminance() - cl;
                    let weight = dist2.mul_add(inv_2ss, diff * diff * inv_2sr).exp();
                    wr += neighbor.r * weight;
                    wg += neighbor.g * weight;
                    wb += neighbor.b * weight;
                    wsum += weight;
                }
            }
            if wsum > 0.0 {
                img.set(x, y, Rgb::new(wr / wsum, wg / wsum, wb / wsum));
            }
        }
    }
}
