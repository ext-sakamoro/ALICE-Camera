//! `Rgb` pixel + `Image` buffer.

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
