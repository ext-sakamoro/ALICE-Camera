//! Cross-module integration tests.

#![allow(
    clippy::doc_markdown,
    clippy::assertions_on_constants,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::redundant_clone,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::manual_range_contains,
    clippy::should_panic_without_expect
)]

use crate::bayer::{bayer_channel, BayerPattern, RawImage};
use crate::demosaic::demosaic_bilinear;
use crate::denoise::{denoise_bilateral, denoise_box_blur, denoise_median};
use crate::distortion::{bilinear_sample, correct_distortion, DistortionCoeffs};
use crate::exposure::{apply_exposure, auto_exposure_ev, average_luminance, ExposureParams};
use crate::focus::{laplacian_variance, tenengrad_metric};
use crate::gamma::{
    apply_gamma, apply_srgb_degamma, apply_srgb_gamma, linear_to_srgb, srgb_to_linear,
};
use crate::hdr::{hdr_merge, tonemap_reinhard, tonemap_reinhard_extended};
use crate::histogram::{histogram_equalize, luminance_histogram};
use crate::image::{Image, Rgb};
use crate::pipeline::{run_isp_pipeline, DenoiseMethod, GammaMode, IspConfig};
use crate::white_balance::{apply_white_balance, grey_world_white_balance, WhiteBalanceGains};

// -- Helpers --

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

// -- Rgb --

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

// -- Image --

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

// -- White Balance --

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

// -- Demosaicing --

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

// -- Exposure --

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

// -- Focus --

#[test]
fn test_laplacian_uniform() {
    let img = solid_image(8, 8, Rgb::new(0.5, 0.5, 0.5));
    let v = laplacian_variance(&img);
    assert!(approx_eq(v, 0.0, 1e-6));
}

#[test]
fn test_laplacian_sharp_higher() {
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

// -- Distortion --

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

// -- Histogram --

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

// -- Denoise --

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

// -- HDR --

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

// -- Gamma --

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

// -- ISP Pipeline --

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

// -- RawImage --

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

// -- Bilinear sample --

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

// -- Edge / integration --

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
    assert_eq!(bayer_channel(BayerPattern::Grbg, 0, 0), 1);
    assert_eq!(bayer_channel(BayerPattern::Grbg, 1, 0), 0);
    assert_eq!(bayer_channel(BayerPattern::Grbg, 0, 1), 2);
    assert_eq!(bayer_channel(BayerPattern::Grbg, 1, 1), 1);
}

#[test]
fn test_bayer_channel_gbrg() {
    assert_eq!(bayer_channel(BayerPattern::Gbrg, 0, 0), 1);
    assert_eq!(bayer_channel(BayerPattern::Gbrg, 1, 0), 2);
    assert_eq!(bayer_channel(BayerPattern::Gbrg, 0, 1), 0);
    assert_eq!(bayer_channel(BayerPattern::Gbrg, 1, 1), 1);
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
