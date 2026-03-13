**English** | [日本語](README_JP.md)

# ALICE-Camera

Camera capture and ISP library for [Project A.L.I.C.E.](https://github.com/anthropics/alice)

## Overview

`alice-camera` provides a pure Rust Image Signal Processing (ISP) pipeline covering the full chain from raw Bayer sensor data to final corrected output.

## Features

- **White Balance** — per-channel gain adjustment for color temperature correction
- **Demosaicing** — Bayer pattern (RGGB) to full RGB reconstruction
- **Exposure Control** — automatic exposure adjustment
- **Auto-Focus Metrics** — Laplacian-based focus scoring
- **Lens Distortion Correction** — radial and tangential distortion removal
- **Histogram Equalization** — contrast enhancement via cumulative distribution
- **Noise Reduction** — spatial denoising filters
- **HDR Merge** — multi-exposure fusion for high dynamic range
- **Gamma Correction** — perceptual luminance mapping

## Quick Start

```rust
use alice_camera::{Rgb, Image};

let mut img = Image::new(640, 480);
img.pixels[0] = Rgb::new(0.8, 0.5, 0.3);

let lum = img.pixels[0].luminance();
let (r, g, b) = img.pixels[0].to_u8();
```

## Architecture

```
alice-camera
├── Rgb              — f32 RGB pixel type with luminance & conversion
├── Image            — row-major image buffer
├── white_balance    — per-channel gain correction
├── demosaic         — Bayer RGGB interpolation
├── exposure         — auto-exposure control
├── autofocus        — Laplacian focus metric
├── distortion       — radial/tangential lens correction
├── histogram        — histogram equalization
├── denoise          — spatial noise reduction
├── hdr              — multi-exposure HDR merge
└── gamma            — gamma curve correction
```

## License

MIT OR Apache-2.0
