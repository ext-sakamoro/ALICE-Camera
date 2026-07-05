//! Auto-focus metrics.

use crate::image::Image;

/// Compute the Laplacian variance as a focus quality metric.
#[must_use]
pub fn laplacian_variance(img: &Image) -> f32 {
    if img.width < 3 || img.height < 3 {
        return 0.0;
    }
    let w = img.width;
    let h = img.height;
    let mut sum = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0_u32;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = img.get(x, y).luminance();
            let n = img.get(x, y - 1).luminance();
            let s = img.get(x, y + 1).luminance();
            let e = img.get(x + 1, y).luminance();
            let west = img.get(x - 1, y).luminance();
            let lap = 4.0f32.mul_add(-c, n + s + e + west);
            sum += lap;
            sum_sq += lap * lap;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f32;
    mean.mul_add(-mean, sum_sq / count as f32)
}

/// Compute the Tenengrad focus metric (Sobel gradient magnitude variance).
#[must_use]
pub fn tenengrad_metric(img: &Image) -> f32 {
    if img.width < 3 || img.height < 3 {
        return 0.0;
    }
    let w = img.width;
    let h = img.height;
    let mut sum = 0.0_f32;
    let mut count = 0_u32;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let tl = img.get(x - 1, y - 1).luminance();
            let tc = img.get(x, y - 1).luminance();
            let tr = img.get(x + 1, y - 1).luminance();
            let ml = img.get(x - 1, y).luminance();
            let mr = img.get(x + 1, y).luminance();
            let bl = img.get(x - 1, y + 1).luminance();
            let bc = img.get(x, y + 1).luminance();
            let br = img.get(x + 1, y + 1).luminance();

            let gx = 2.0f32.mul_add(mr, -tl + tr) + 2.0f32.mul_add(-ml, -bl + br);
            let gy = 2.0f32.mul_add(bc, bl + br) + 2.0f32.mul_add(-tc, -tl - tr);
            sum += gx.mul_add(gx, gy * gy);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}
