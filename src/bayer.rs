//! Bayer CFA pattern + `RawImage`.

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

/// Determine the color channel at a given Bayer position.
/// Returns 0=R, 1=G, 2=B.
pub(crate) const fn bayer_channel(pattern: BayerPattern, x: usize, y: usize) -> usize {
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

        _ => 1,
    }
}
