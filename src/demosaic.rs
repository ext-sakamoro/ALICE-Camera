//! Bilinear demosaicing of raw Bayer images.

use crate::bayer::{bayer_channel, RawImage};
use crate::image::{Image, Rgb};

/// Bilinear demosaicing of a raw Bayer image.
///
/// # Panics
/// Panics if `raw.width < 2` or `raw.height < 2`.
#[must_use]
pub fn demosaic_bilinear(raw: &RawImage) -> Image {
    assert!(raw.width >= 2 && raw.height >= 2, "image too small");
    let w = raw.width;
    let h = raw.height;
    let mut img = Image::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let ch = bayer_channel(raw.pattern, x, y);
            let val = raw.get(x, y);

            let mut rgb = [0.0_f32; 3];
            rgb[ch] = val;

            for (c, slot) in rgb.iter_mut().enumerate() {
                if c == ch {
                    continue;
                }
                let mut sum = 0.0_f32;
                let mut count = 0_u32;
                for dy in -1_i32..=1 {
                    for dx in -1_i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let (ux, uy) = (nx as usize, ny as usize);
                        if bayer_channel(raw.pattern, ux, uy) == c {
                            sum += raw.get(ux, uy);
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    *slot = sum / count as f32;
                }
            }

            img.set(x, y, Rgb::new(rgb[0], rgb[1], rgb[2]));
        }
    }
    img
}
