#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

//! ALICE-Camera: Camera capture and ISP (Image Signal Processing)
//!
//! Pure Rust implementation of core ISP pipeline stages:
//! - White balance
//! - Demosaicing (Bayer pattern)
//! - Exposure control
//! - Auto-focus metrics
//! - Lens distortion correction
//! - Histogram equalization
//! - Noise reduction
//! - HDR merge
//! - Gamma correction

// ---------------------------------------------------------------------------
// Core pixel / image types
// ---------------------------------------------------------------------------

/// RGB pixel with f32 channels in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    #[must_use]
    pub const fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    #[must_use]
    pub fn luminance(self) -> f32 {
        0.2126f32.mul_add(self.r, 0.7152f32.mul_add(self.g, 0.0722 * self.b))
    }

    /// Convert to 8-bit sRGB tuple.
    #[must_use]
    pub fn to_u8(self) -> (u8, u8, u8) {
        let c = self.clamp01();
        (
            c.r.mul_add(255.0, 0.5) as u8,
            c.g.mul_add(255.0, 0.5) as u8,
            c.b.mul_add(255.0, 0.5) as u8,
        )
    }
}

/// A simple image buffer (row-major, width * height pixels).
#[derive(Debug, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Rgb>,
}

impl Image {
    /// Create a new image filled with black.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Rgb::new(0.0, 0.0, 0.0); width * height],
        }
    }

    /// Create an image from existing pixel data.
    ///
    /// # Panics
    /// Panics if `pixels.len() != width * height`.
    #[must_use]
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Rgb>) -> Self {
        assert_eq!(pixels.len(), width * height, "pixel count mismatch");
        Self {
            width,
            height,
            pixels,
        }
    }

    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Rgb {
        self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: Rgb) {
        self.pixels[y * self.width + x] = c;
    }

    /// Clamp all pixels to [0, 1].
    pub fn clamp_all(&mut self) {
        for p in &mut self.pixels {
            *p = p.clamp01();
        }
    }
}

// ---------------------------------------------------------------------------
// Bayer pattern (raw sensor)
// ---------------------------------------------------------------------------

/// Bayer CFA pattern layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    /// R G / G B
    Rggb,
    /// B G / G R
    Bggr,
    /// G R / B G
    Grbg,
    /// G B / R G
    Gbrg,
}

/// Single-channel raw sensor image.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub pattern: BayerPattern,
}

impl RawImage {
    #[must_use]
    pub fn new(width: usize, height: usize, pattern: BayerPattern) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
            pattern,
        }
    }

    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }
}

// ---------------------------------------------------------------------------
// 1. White Balance
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 2. Demosaicing (Bayer)
// ---------------------------------------------------------------------------

/// Determine the color channel at a given Bayer position.
/// Returns 0=R, 1=G, 2=B.
const fn bayer_channel(pattern: BayerPattern, x: usize, y: usize) -> usize {
    let (ex, ey) = (x & 1, y & 1);
    match (pattern, ey, ex) {
        (BayerPattern::Rggb, 0, 0)
        | (BayerPattern::Bggr, 1, 1)
        | (BayerPattern::Grbg, 0, 1)
        | (BayerPattern::Gbrg, 1, 0) => 0,

        (BayerPattern::Rggb, 1, 1)
        | (BayerPattern::Bggr, 0, 0)
        | (BayerPattern::Grbg, 1, 0)
        | (BayerPattern::Gbrg, 0, 1) => 2,

        // Green positions (all remaining patterns)
        _ => 1,
    }
}

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

            // Interpolate missing channels via neighbor average
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

// ---------------------------------------------------------------------------
// 3. Exposure Control
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 4. Auto-Focus Metrics
// ---------------------------------------------------------------------------

/// Compute the Laplacian variance as a focus quality metric.
/// Higher value = sharper image.
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

// ---------------------------------------------------------------------------
// 5. Lens Distortion Correction
// ---------------------------------------------------------------------------

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
/// Uses inverse mapping with bilinear interpolation.
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
fn bilinear_sample(img: &Image, fx: f32, fy: f32) -> Rgb {
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

// ---------------------------------------------------------------------------
// 6. Histogram Equalization
// ---------------------------------------------------------------------------

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

    // CDF
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

// ---------------------------------------------------------------------------
// 7. Noise Reduction
// ---------------------------------------------------------------------------

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
/// `sigma_s` controls spatial extent, `sigma_r` controls range (color) extent.
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

// ---------------------------------------------------------------------------
// 8. HDR Merge
// ---------------------------------------------------------------------------

/// Merge multiple exposures into an HDR image using exposure-weighted averaging.
/// `images` is a slice of (image, `exposure_time`) pairs.
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
            // Triangle weighting: weight peaks at mid-tones
            let lum = p.luminance().clamp(0.0, 1.0);
            let weight = 1.0 - 2.0f32.mul_add(lum, -1.0).abs();
            let weight = weight.max(0.01); // avoid zero weight

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

// ---------------------------------------------------------------------------
// 9. Gamma Correction
// ---------------------------------------------------------------------------

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

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ---------------------------------------------------------------------------
// Full ISP Pipeline
// ---------------------------------------------------------------------------

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
    // 1. Demosaic
    let mut img = demosaic_bilinear(raw);

    // 2. White balance
    if let Some(gains) = config.white_balance {
        apply_white_balance(&mut img, gains);
    } else if config.auto_white_balance {
        let gains = grey_world_white_balance(&img);
        apply_white_balance(&mut img, gains);
    }

    // 3. Exposure
    if let Some(ev) = config.ev_offset {
        apply_exposure(&mut img, ExposureParams { ev_offset: ev });
    } else if let Some(target) = config.auto_exposure_target {
        let ev = auto_exposure_ev(&img, target);
        apply_exposure(&mut img, ExposureParams { ev_offset: ev });
    }

    // 4. Lens distortion correction
    if let Some(coeffs) = config.distortion {
        img = correct_distortion(&img, coeffs);
    }

    // 5. Noise reduction
    match config.denoise {
        DenoiseMethod::None => {}
        DenoiseMethod::BoxBlur => denoise_box_blur(&mut img),
        DenoiseMethod::Median => denoise_median(&mut img),
    }

    // 6. Histogram equalization
    if config.histogram_equalize {
        histogram_equalize(&mut img);
    }

    // 7. Gamma
    match config.gamma {
        GammaMode::None => {}
        GammaMode::Power(g) => apply_gamma(&mut img, g),
        GammaMode::Srgb => apply_srgb_gamma(&mut img),
    }

    img.clamp_all();
    img
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn solid_image(w: usize, h: usize, c: Rgb) -> Image {
        Image::from_pixels(w, h, vec![c; w * h])
    }

    fn gradient_image(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let t = (x as f32 + y as f32 * w as f32) / (w * h) as f32;
                img.set(x, y, Rgb::new(t, t * 0.5, 1.0 - t));
            }
        }
        img
    }

    fn make_raw_checkerboard(w: usize, h: usize, pattern: BayerPattern) -> RawImage {
        let mut raw = RawImage::new(w, h, pattern);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0.8 } else { 0.3 };
                raw.set(x, y, v);
            }
        }
        raw
    }

    // -----------------------------------------------------------------------
    // Rgb tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rgb_new() {
        let c = Rgb::new(0.5, 0.6, 0.7);
        assert!(approx_eq(c.r, 0.5, 1e-6));
        assert!(approx_eq(c.g, 0.6, 1e-6));
        assert!(approx_eq(c.b, 0.7, 1e-6));
    }

    #[test]
    fn test_rgb_clamp01() {
        let c = Rgb::new(-0.1, 1.5, 0.5).clamp01();
        assert!(approx_eq(c.r, 0.0, 1e-6));
        assert!(approx_eq(c.g, 1.0, 1e-6));
        assert!(approx_eq(c.b, 0.5, 1e-6));
    }

    #[test]
    fn test_rgb_luminance() {
        let white = Rgb::new(1.0, 1.0, 1.0);
        assert!(approx_eq(white.luminance(), 1.0, 1e-4));
        let black = Rgb::new(0.0, 0.0, 0.0);
        assert!(approx_eq(black.luminance(), 0.0, 1e-6));
    }

    #[test]
    fn test_rgb_to_u8() {
        let c = Rgb::new(0.0, 0.5, 1.0);
        let (r, g, b) = c.to_u8();
        assert_eq!(r, 0);
        assert_eq!(b, 255);
        assert!(g >= 127 && g <= 128);
    }

    #[test]
    fn test_rgb_to_u8_clamped() {
        let c = Rgb::new(-1.0, 2.0, 0.5);
        let (r, g, _) = c.to_u8();
        assert_eq!(r, 0);
        assert_eq!(g, 255);
    }

    // -----------------------------------------------------------------------
    // Image tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_new() {
        let img = Image::new(4, 4);
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.pixels.len(), 16);
    }

    #[test]
    fn test_image_get_set() {
        let mut img = Image::new(4, 4);
        img.set(2, 3, Rgb::new(1.0, 0.0, 0.0));
        let c = img.get(2, 3);
        assert!(approx_eq(c.r, 1.0, 1e-6));
    }

    #[test]
    fn test_image_clamp_all() {
        let mut img = Image::from_pixels(
            2,
            1,
            vec![Rgb::new(-1.0, 2.0, 0.5), Rgb::new(0.3, 0.3, 0.3)],
        );
        img.clamp_all();
        assert!(approx_eq(img.get(0, 0).r, 0.0, 1e-6));
        assert!(approx_eq(img.get(0, 0).g, 1.0, 1e-6));
    }

    #[test]
    #[should_panic(expected = "pixel count mismatch")]
    fn test_image_from_pixels_mismatch() {
        let _ = Image::from_pixels(2, 2, vec![Rgb::new(0.0, 0.0, 0.0)]);
    }

    // -----------------------------------------------------------------------
    // White Balance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_white_balance_apply() {
        let mut img = solid_image(2, 2, Rgb::new(0.5, 0.5, 0.5));
        apply_white_balance(&mut img, WhiteBalanceGains::new(2.0, 1.0, 0.5));
        assert!(approx_eq(img.get(0, 0).r, 1.0, 1e-6));
        assert!(approx_eq(img.get(0, 0).g, 0.5, 1e-6));
        assert!(approx_eq(img.get(0, 0).b, 0.25, 1e-6));
    }

    #[test]
    fn test_grey_world_neutral() {
        let img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        let gains = grey_world_white_balance(&img);
        assert!(approx_eq(gains.r_gain, 1.0, 1e-4));
        assert!(approx_eq(gains.g_gain, 1.0, 1e-4));
        assert!(approx_eq(gains.b_gain, 1.0, 1e-4));
    }

    #[test]
    fn test_grey_world_biased() {
        let img = solid_image(4, 4, Rgb::new(0.8, 0.4, 0.4));
        let gains = grey_world_white_balance(&img);
        assert!(gains.r_gain < 1.0);
        assert!(gains.g_gain > 1.0);
    }

    #[test]
    fn test_grey_world_empty() {
        let img = Image::new(0, 0);
        let gains = grey_world_white_balance(&img);
        assert!(approx_eq(gains.r_gain, 1.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Demosaicing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_demosaic_rggb_uniform() {
        let mut raw = RawImage::new(4, 4, BayerPattern::Rggb);
        for v in &mut raw.data {
            *v = 0.5;
        }
        let img = demosaic_bilinear(&raw);
        let c = img.get(1, 1);
        assert!(approx_eq(c.r, 0.5, 0.1));
        assert!(approx_eq(c.g, 0.5, 0.1));
        assert!(approx_eq(c.b, 0.5, 0.1));
    }

    #[test]
    fn test_demosaic_bggr() {
        let raw = make_raw_checkerboard(4, 4, BayerPattern::Bggr);
        let img = demosaic_bilinear(&raw);
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
    }

    #[test]
    fn test_demosaic_grbg() {
        let raw = make_raw_checkerboard(4, 4, BayerPattern::Grbg);
        let img = demosaic_bilinear(&raw);
        assert_eq!(img.pixels.len(), 16);
    }

    #[test]
    fn test_demosaic_gbrg() {
        let raw = make_raw_checkerboard(6, 6, BayerPattern::Gbrg);
        let img = demosaic_bilinear(&raw);
        let c = img.get(3, 3);
        assert!(c.r >= 0.0 && c.r <= 1.0);
    }

    #[test]
    fn test_demosaic_preserves_dimensions() {
        let raw = RawImage::new(8, 6, BayerPattern::Rggb);
        let img = demosaic_bilinear(&raw);
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 6);
    }

    #[test]
    #[should_panic]
    fn test_demosaic_too_small() {
        let raw = RawImage::new(1, 1, BayerPattern::Rggb);
        let _ = demosaic_bilinear(&raw);
    }

    #[test]
    fn test_bayer_channel_rggb() {
        assert_eq!(bayer_channel(BayerPattern::Rggb, 0, 0), 0);
        assert_eq!(bayer_channel(BayerPattern::Rggb, 1, 0), 1);
        assert_eq!(bayer_channel(BayerPattern::Rggb, 0, 1), 1);
        assert_eq!(bayer_channel(BayerPattern::Rggb, 1, 1), 2);
    }

    #[test]
    fn test_bayer_channel_bggr() {
        assert_eq!(bayer_channel(BayerPattern::Bggr, 0, 0), 2);
        assert_eq!(bayer_channel(BayerPattern::Bggr, 1, 1), 0);
    }

    // -----------------------------------------------------------------------
    // Exposure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exposure_positive() {
        let mut img = solid_image(2, 2, Rgb::new(0.25, 0.25, 0.25));
        apply_exposure(&mut img, ExposureParams { ev_offset: 1.0 });
        assert!(approx_eq(img.get(0, 0).r, 0.5, 1e-4));
    }

    #[test]
    fn test_exposure_negative() {
        let mut img = solid_image(2, 2, Rgb::new(0.5, 0.5, 0.5));
        apply_exposure(&mut img, ExposureParams { ev_offset: -1.0 });
        assert!(approx_eq(img.get(0, 0).r, 0.25, 1e-4));
    }

    #[test]
    fn test_exposure_zero() {
        let mut img = solid_image(2, 2, Rgb::new(0.5, 0.5, 0.5));
        apply_exposure(&mut img, ExposureParams { ev_offset: 0.0 });
        assert!(approx_eq(img.get(0, 0).r, 0.5, 1e-4));
    }

    #[test]
    fn test_average_luminance() {
        let img = solid_image(4, 4, Rgb::new(1.0, 1.0, 1.0));
        assert!(approx_eq(average_luminance(&img), 1.0, 1e-4));
    }

    #[test]
    fn test_average_luminance_black() {
        let img = solid_image(4, 4, Rgb::new(0.0, 0.0, 0.0));
        assert!(approx_eq(average_luminance(&img), 0.0, 1e-6));
    }

    #[test]
    fn test_auto_exposure_bright() {
        let img = solid_image(4, 4, Rgb::new(0.9, 0.9, 0.9));
        let ev = auto_exposure_ev(&img, 0.18);
        assert!(ev < 0.0);
    }

    #[test]
    fn test_auto_exposure_dark() {
        let img = solid_image(4, 4, Rgb::new(0.05, 0.05, 0.05));
        let ev = auto_exposure_ev(&img, 0.18);
        assert!(ev > 0.0);
    }

    #[test]
    fn test_auto_exposure_black() {
        let img = solid_image(4, 4, Rgb::new(0.0, 0.0, 0.0));
        let ev = auto_exposure_ev(&img, 0.18);
        assert!(approx_eq(ev, 0.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Focus metric tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_laplacian_uniform() {
        let img = solid_image(8, 8, Rgb::new(0.5, 0.5, 0.5));
        let v = laplacian_variance(&img);
        assert!(approx_eq(v, 0.0, 1e-6));
    }

    #[test]
    fn test_laplacian_sharp_higher() {
        // Create an image with sharp edges
        let mut sharp = Image::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                sharp.set(x, y, Rgb::new(v, v, v));
            }
        }
        let mut blurry = sharp.clone();
        denoise_box_blur(&mut blurry);
        denoise_box_blur(&mut blurry);
        denoise_box_blur(&mut blurry);
        assert!(laplacian_variance(&sharp) > laplacian_variance(&blurry));
    }

    #[test]
    fn test_laplacian_small() {
        let img = Image::new(2, 2);
        assert!(approx_eq(laplacian_variance(&img), 0.0, 1e-6));
    }

    #[test]
    fn test_tenengrad_uniform() {
        let img = solid_image(8, 8, Rgb::new(0.5, 0.5, 0.5));
        let v = tenengrad_metric(&img);
        assert!(approx_eq(v, 0.0, 1e-6));
    }

    #[test]
    fn test_tenengrad_sharp_higher() {
        let sharp = gradient_image(16, 16);
        let mut blurry = sharp.clone();
        denoise_box_blur(&mut blurry);
        denoise_box_blur(&mut blurry);
        assert!(tenengrad_metric(&sharp) > tenengrad_metric(&blurry));
    }

    #[test]
    fn test_tenengrad_small() {
        let img = Image::new(1, 1);
        assert!(approx_eq(tenengrad_metric(&img), 0.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Distortion correction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_distortion_identity() {
        let img = gradient_image(8, 8);
        let out = correct_distortion(&img, DistortionCoeffs::identity());
        for i in 0..img.pixels.len() {
            assert!(approx_eq(img.pixels[i].r, out.pixels[i].r, 0.05));
        }
    }

    #[test]
    fn test_distortion_barrel() {
        let img = gradient_image(16, 16);
        let out = correct_distortion(&img, DistortionCoeffs::new(-0.3, 0.0, 0.0));
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
    }

    #[test]
    fn test_distortion_pincushion() {
        let img = gradient_image(16, 16);
        let out = correct_distortion(&img, DistortionCoeffs::new(0.3, 0.0, 0.0));
        assert_eq!(out.width, 16);
    }

    #[test]
    fn test_distortion_center_preserved() {
        let img = solid_image(8, 8, Rgb::new(0.5, 0.5, 0.5));
        let out = correct_distortion(&img, DistortionCoeffs::new(-0.5, 0.1, 0.0));
        let c = out.get(4, 4);
        assert!(approx_eq(c.r, 0.5, 0.01));
    }

    #[test]
    fn test_distortion_coeffs_new() {
        let c = DistortionCoeffs::new(0.1, 0.2, 0.3);
        assert!(approx_eq(c.k1, 0.1, 1e-6));
        assert!(approx_eq(c.k2, 0.2, 1e-6));
        assert!(approx_eq(c.k3, 0.3, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Histogram equalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_histogram_bins() {
        let img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        let hist = luminance_histogram(&img, 256);
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 16);
    }

    #[test]
    fn test_histogram_black_image() {
        let img = solid_image(4, 4, Rgb::new(0.0, 0.0, 0.0));
        let hist = luminance_histogram(&img, 256);
        assert_eq!(hist[0], 16);
    }

    #[test]
    fn test_histogram_white_image() {
        let img = solid_image(4, 4, Rgb::new(1.0, 1.0, 1.0));
        let hist = luminance_histogram(&img, 256);
        assert_eq!(hist[255], 16);
    }

    #[test]
    fn test_histogram_equalize_uniform() {
        let mut img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        histogram_equalize(&mut img);
        let c = img.get(0, 0);
        assert!(c.r >= 0.0);
    }

    #[test]
    fn test_histogram_equalize_gradient() {
        let mut img = gradient_image(8, 8);
        histogram_equalize(&mut img);
        for p in &img.pixels {
            assert!(p.r >= -0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Noise reduction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_box_blur_uniform() {
        let mut img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        denoise_box_blur(&mut img);
        let c = img.get(1, 1);
        assert!(approx_eq(c.r, 0.5, 1e-4));
    }

    #[test]
    fn test_box_blur_reduces_noise() {
        let mut img = Image::new(8, 8);
        for (i, p) in img.pixels.iter_mut().enumerate() {
            let v = if i % 2 == 0 { 0.8 } else { 0.2 };
            *p = Rgb::new(v, v, v);
        }
        let variance_before: f32 =
            img.pixels.iter().map(|p| (p.r - 0.5).powi(2)).sum::<f32>() / img.pixels.len() as f32;
        denoise_box_blur(&mut img);
        let variance_after: f32 =
            img.pixels.iter().map(|p| (p.r - 0.5).powi(2)).sum::<f32>() / img.pixels.len() as f32;
        assert!(variance_after < variance_before);
    }

    #[test]
    fn test_median_uniform() {
        let mut img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        denoise_median(&mut img);
        let c = img.get(1, 1);
        assert!(approx_eq(c.r, 0.5, 1e-4));
    }

    #[test]
    fn test_median_salt_pepper() {
        let mut img = solid_image(5, 5, Rgb::new(0.5, 0.5, 0.5));
        img.set(2, 2, Rgb::new(1.0, 1.0, 1.0));
        denoise_median(&mut img);
        let c = img.get(2, 2);
        assert!(approx_eq(c.r, 0.5, 1e-4));
    }

    #[test]
    fn test_bilateral_uniform() {
        let mut img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        denoise_bilateral(&mut img, 1, 1.0, 0.1);
        let c = img.get(1, 1);
        assert!(approx_eq(c.r, 0.5, 1e-3));
    }

    #[test]
    fn test_bilateral_edge_preserve() {
        let mut img = Image::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let v = if x < 4 { 0.2 } else { 0.8 };
                img.set(x, y, Rgb::new(v, v, v));
            }
        }
        denoise_bilateral(&mut img, 1, 1.0, 0.05);
        assert!(img.get(1, 4).r < 0.4);
        assert!(img.get(6, 4).r > 0.6);
    }

    // -----------------------------------------------------------------------
    // HDR merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hdr_merge_single() {
        let img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        let merged = hdr_merge(&[(img, 1.0)]);
        let c = merged.get(0, 0);
        assert!(approx_eq(c.r, 0.5, 0.1));
    }

    #[test]
    fn test_hdr_merge_two_exposures() {
        let dark = solid_image(4, 4, Rgb::new(0.1, 0.1, 0.1));
        let bright = solid_image(4, 4, Rgb::new(0.8, 0.8, 0.8));
        let merged = hdr_merge(&[(dark, 0.25), (bright, 2.0)]);
        assert_eq!(merged.width, 4);
        assert!(merged.get(0, 0).r > 0.0);
    }

    #[test]
    fn test_hdr_merge_three_exposures() {
        let e1 = solid_image(4, 4, Rgb::new(0.05, 0.05, 0.05));
        let e2 = solid_image(4, 4, Rgb::new(0.4, 0.4, 0.4));
        let e3 = solid_image(4, 4, Rgb::new(0.95, 0.95, 0.95));
        let merged = hdr_merge(&[(e1, 0.125), (e2, 1.0), (e3, 4.0)]);
        assert!(merged.get(0, 0).r > 0.0);
    }

    #[test]
    #[should_panic]
    fn test_hdr_merge_empty() {
        let _ = hdr_merge(&[]);
    }

    #[test]
    fn test_tonemap_reinhard() {
        let mut img = solid_image(2, 2, Rgb::new(2.0, 4.0, 0.5));
        tonemap_reinhard(&mut img);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 2.0 / 3.0, 1e-4));
        assert!(approx_eq(c.g, 0.8, 1e-4));
    }

    #[test]
    fn test_tonemap_reinhard_extended() {
        let mut img = solid_image(2, 2, Rgb::new(1.0, 1.0, 1.0));
        tonemap_reinhard_extended(&mut img, 2.0);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 0.625, 1e-4));
    }

    #[test]
    fn test_tonemap_reinhard_zero() {
        let mut img = solid_image(2, 2, Rgb::new(0.0, 0.0, 0.0));
        tonemap_reinhard(&mut img);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 0.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Gamma tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_22() {
        let mut img = solid_image(2, 2, Rgb::new(0.5, 0.5, 0.5));
        apply_gamma(&mut img, 2.2);
        let c = img.get(0, 0);
        let expected = 0.5_f32.powf(1.0 / 2.2);
        assert!(approx_eq(c.r, expected, 1e-4));
    }

    #[test]
    fn test_gamma_1_identity() {
        let mut img = solid_image(2, 2, Rgb::new(0.3, 0.6, 0.9));
        apply_gamma(&mut img, 1.0);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 0.3, 1e-4));
    }

    #[test]
    fn test_gamma_black() {
        let mut img = solid_image(2, 2, Rgb::new(0.0, 0.0, 0.0));
        apply_gamma(&mut img, 2.2);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 0.0, 1e-6));
    }

    #[test]
    fn test_gamma_white() {
        let mut img = solid_image(2, 2, Rgb::new(1.0, 1.0, 1.0));
        apply_gamma(&mut img, 2.2);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 1.0, 1e-4));
    }

    #[test]
    fn test_srgb_gamma_roundtrip() {
        let original = Rgb::new(0.3, 0.5, 0.8);
        let mut img = solid_image(1, 1, original);
        apply_srgb_gamma(&mut img);
        apply_srgb_degamma(&mut img);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, original.r, 1e-4));
        assert!(approx_eq(c.g, original.g, 1e-4));
        assert!(approx_eq(c.b, original.b, 1e-4));
    }

    #[test]
    fn test_srgb_gamma_low_value() {
        let v = linear_to_srgb(0.001);
        assert!(v > 0.0 && v < 0.02);
    }

    #[test]
    fn test_srgb_gamma_high_value() {
        let v = linear_to_srgb(0.5);
        assert!(v > 0.5);
    }

    #[test]
    fn test_srgb_degamma_low() {
        let v = srgb_to_linear(0.01);
        assert!(v < 0.01);
    }

    #[test]
    fn test_linear_to_srgb_zero() {
        assert!(approx_eq(linear_to_srgb(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_linear_to_srgb_one() {
        assert!(approx_eq(linear_to_srgb(1.0), 1.0, 1e-4));
    }

    #[test]
    fn test_srgb_to_linear_zero() {
        assert!(approx_eq(srgb_to_linear(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_srgb_to_linear_one() {
        assert!(approx_eq(srgb_to_linear(1.0), 1.0, 1e-4));
    }

    // -----------------------------------------------------------------------
    // ISP Pipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_default() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig::default();
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        for p in &img.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
            assert!(p.g >= 0.0 && p.g <= 1.0);
            assert!(p.b >= 0.0 && p.b <= 1.0);
        }
    }

    #[test]
    fn test_pipeline_with_box_denoise() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            denoise: DenoiseMethod::BoxBlur,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_with_median_denoise() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            denoise: DenoiseMethod::Median,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_with_distortion() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            distortion: Some(DistortionCoeffs::new(-0.1, 0.0, 0.0)),
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_with_histogram_eq() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            histogram_equalize: true,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.pixels.len(), 64);
    }

    #[test]
    fn test_pipeline_manual_wb() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            white_balance: Some(WhiteBalanceGains::new(1.2, 1.0, 0.8)),
            auto_white_balance: false,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_manual_ev() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            ev_offset: Some(1.0),
            auto_exposure_target: None,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_power_gamma() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            gamma: GammaMode::Power(2.2),
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_no_gamma() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Rggb);
        let config = IspConfig {
            gamma: GammaMode::None,
            ..IspConfig::default()
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_bggr_pattern() {
        let raw = make_raw_checkerboard(8, 8, BayerPattern::Bggr);
        let img = run_isp_pipeline(&raw, &IspConfig::default());
        assert_eq!(img.width, 8);
    }

    #[test]
    fn test_pipeline_all_features() {
        let raw = make_raw_checkerboard(16, 16, BayerPattern::Grbg);
        let config = IspConfig {
            white_balance: Some(WhiteBalanceGains::new(1.1, 1.0, 0.9)),
            auto_white_balance: false,
            ev_offset: Some(0.5),
            auto_exposure_target: None,
            distortion: Some(DistortionCoeffs::new(-0.05, 0.01, 0.0)),
            histogram_equalize: true,
            denoise: DenoiseMethod::Median,
            gamma: GammaMode::Srgb,
        };
        let img = run_isp_pipeline(&raw, &config);
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        for p in &img.pixels {
            assert!(p.r >= 0.0 && p.r <= 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // RawImage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_raw_image_new() {
        let raw = RawImage::new(4, 4, BayerPattern::Rggb);
        assert_eq!(raw.data.len(), 16);
        assert_eq!(raw.pattern, BayerPattern::Rggb);
    }

    #[test]
    fn test_raw_image_get_set() {
        let mut raw = RawImage::new(4, 4, BayerPattern::Rggb);
        raw.set(2, 1, 0.75);
        assert!(approx_eq(raw.get(2, 1), 0.75, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Bilinear sample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bilinear_sample_exact() {
        let img = solid_image(4, 4, Rgb::new(0.5, 0.5, 0.5));
        let c = bilinear_sample(&img, 1.0, 1.0);
        assert!(approx_eq(c.r, 0.5, 1e-4));
    }

    #[test]
    fn test_bilinear_sample_fractional() {
        let mut img = Image::new(2, 2);
        img.set(0, 0, Rgb::new(0.0, 0.0, 0.0));
        img.set(1, 0, Rgb::new(1.0, 0.0, 0.0));
        img.set(0, 1, Rgb::new(0.0, 1.0, 0.0));
        img.set(1, 1, Rgb::new(1.0, 1.0, 0.0));
        let c = bilinear_sample(&img, 0.5, 0.5);
        assert!(approx_eq(c.r, 0.5, 0.1));
        assert!(approx_eq(c.g, 0.5, 0.1));
    }

    #[test]
    fn test_bilinear_sample_edge() {
        let img = solid_image(4, 4, Rgb::new(0.3, 0.3, 0.3));
        let c = bilinear_sample(&img, -1.0, -1.0);
        assert!(approx_eq(c.r, 0.3, 1e-4));
    }

    // -----------------------------------------------------------------------
    // Edge case and integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wb_gains_constructor() {
        let g = WhiteBalanceGains::new(1.5, 1.0, 0.8);
        assert!(approx_eq(g.r_gain, 1.5, 1e-6));
    }

    #[test]
    fn test_exposure_params() {
        let p = ExposureParams { ev_offset: 2.0 };
        assert!(approx_eq(p.ev_offset, 2.0, 1e-6));
    }

    #[test]
    fn test_denoise_method_eq() {
        assert_eq!(DenoiseMethod::None, DenoiseMethod::None);
        assert_ne!(DenoiseMethod::BoxBlur, DenoiseMethod::Median);
    }

    #[test]
    fn test_isp_config_default() {
        let c = IspConfig::default();
        assert!(c.auto_white_balance);
        assert_eq!(c.denoise, DenoiseMethod::None);
    }

    #[test]
    fn test_multiple_box_blurs() {
        let mut img = gradient_image(8, 8);
        denoise_box_blur(&mut img);
        denoise_box_blur(&mut img);
        denoise_box_blur(&mut img);
        for p in &img.pixels {
            assert!(p.r >= 0.0);
        }
    }

    #[test]
    fn test_histogram_16_bins() {
        let img = gradient_image(8, 8);
        let hist = luminance_histogram(&img, 16);
        assert_eq!(hist.len(), 16);
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_large_image_pipeline() {
        let raw = make_raw_checkerboard(32, 32, BayerPattern::Gbrg);
        let img = run_isp_pipeline(&raw, &IspConfig::default());
        assert_eq!(img.pixels.len(), 1024);
    }

    #[test]
    fn test_negative_gamma_input() {
        let mut img = solid_image(2, 2, Rgb::new(-0.5, 0.5, 0.5));
        apply_gamma(&mut img, 2.2);
        assert!(approx_eq(img.get(0, 0).r, 0.0, 1e-6));
    }

    #[test]
    fn test_srgb_negative_input() {
        assert!(approx_eq(linear_to_srgb(-1.0), 0.0, 1e-6));
    }

    #[test]
    fn test_luminance_red_only() {
        let c = Rgb::new(1.0, 0.0, 0.0);
        assert!(approx_eq(c.luminance(), 0.2126, 1e-4));
    }

    #[test]
    fn test_luminance_green_only() {
        let c = Rgb::new(0.0, 1.0, 0.0);
        assert!(approx_eq(c.luminance(), 0.7152, 1e-4));
    }

    #[test]
    fn test_luminance_blue_only() {
        let c = Rgb::new(0.0, 0.0, 1.0);
        assert!(approx_eq(c.luminance(), 0.0722, 1e-4));
    }

    #[test]
    fn test_bayer_pattern_equality() {
        assert_eq!(BayerPattern::Rggb, BayerPattern::Rggb);
        assert_ne!(BayerPattern::Rggb, BayerPattern::Bggr);
    }

    #[test]
    fn test_bayer_channel_grbg() {
        assert_eq!(bayer_channel(BayerPattern::Grbg, 0, 0), 1); // G
        assert_eq!(bayer_channel(BayerPattern::Grbg, 1, 0), 0); // R
        assert_eq!(bayer_channel(BayerPattern::Grbg, 0, 1), 2); // B
        assert_eq!(bayer_channel(BayerPattern::Grbg, 1, 1), 1); // G
    }

    #[test]
    fn test_bayer_channel_gbrg() {
        assert_eq!(bayer_channel(BayerPattern::Gbrg, 0, 0), 1); // G
        assert_eq!(bayer_channel(BayerPattern::Gbrg, 1, 0), 2); // B
        assert_eq!(bayer_channel(BayerPattern::Gbrg, 0, 1), 0); // R
        assert_eq!(bayer_channel(BayerPattern::Gbrg, 1, 1), 1); // G
    }

    #[test]
    fn test_tonemap_reinhard_extended_zero() {
        let mut img = solid_image(2, 2, Rgb::new(0.0, 0.0, 0.0));
        tonemap_reinhard_extended(&mut img, 2.0);
        let c = img.get(0, 0);
        assert!(approx_eq(c.r, 0.0, 1e-6));
    }

    #[test]
    fn test_average_luminance_empty() {
        let img = Image::new(0, 0);
        assert!(approx_eq(average_luminance(&img), 0.0, 1e-6));
    }
}
