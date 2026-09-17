#[derive(Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
    pub is_ground: bool,
    pub wrong: u8,
}

impl kdtree::spatial::Spatial for Point {
    fn axis(&self, axis: u8) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => 0.0,
        }
    }
}
