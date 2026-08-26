#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elem {
    pub value: f64,
    pub depth: u8,
}

impl Elem {
    pub const fn new(value: f64) -> Self {
        Self { value, depth: 0 }
    }
}

pub const MAX_UNARY_DEPTH: u8 = 3;
pub const EPSILON: f64 = 1e-6;
pub const MAGNITUDE_LIMIT: f64 = 1e8;

pub fn near_target(value: f64, target: f64) -> bool {
    (value - target).abs() < EPSILON
}
