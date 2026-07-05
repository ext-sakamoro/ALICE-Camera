//! Histogram equalization.

use crate::image::Image;

/// Compute a luminance histogram with the given number of bins.
#[must_use]
pub fn luminance_histogram(img: &Image, bins: usize) -> Vec<u32> {
    let mut hist = vec![0_u32; bins];
    for p in &img.pixels {
        let lum = p.luminance().clamp(0.0, 1.0);
        let idx = lum.mul_add(bins as f32 - 1.0, 0.5) as usize;
        let idx = idx.min(bins - 1);
        hist[idx] += 1;
    }
    hist
}

/// Apply histogram equalization on the luminance channel.
pub fn histogram_equalize(img: &mut Image) {
    let bins = 256;
    let hist = luminance_histogram(img, bins);
    let n = img.pixels.len() as f32;
    if n == 0.0 {
        return;
    }

    let mut cdf = vec![0.0_f32; bins];
    cdf[0] = hist[0] as f32 / n;
    for i in 1..bins {
        cdf[i] = cdf[i - 1] + hist[i] as f32 / n;
    }

    for p in &mut img.pixels {
        let lum = p.luminance().clamp(0.0, 1.0);
        let idx = lum.mul_add(bins as f32 - 1.0, 0.5) as usize;
        let idx = idx.min(bins - 1);
        let new_lum = cdf[idx];

        if lum > 1e-9 {
            let scale = new_lum / lum;
            p.r *= scale;
            p.g *= scale;
            p.b *= scale;
        } else {
            p.r = new_lum;
            p.g = new_lum;
            p.b = new_lum;
        }
    }
}
