use crate::value::{Elem, MAGNITUDE_LIMIT, MAX_UNARY_DEPTH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Ln,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Binary { i: usize, j: usize, op: BinaryOp },
    Unary { i: usize, op: UnaryOp },
}

fn bounded(v: f64) -> Option<f64> {
    if v.is_finite() && v.abs() <= MAGNITUDE_LIMIT {
        Some(v)
    } else {
        None
    }
}

pub fn apply_binary(a: &Elem, b: &Elem, op: BinaryOp) -> Option<Elem> {
    let raw = match op {
        BinaryOp::Add => a.value + b.value,
        BinaryOp::Sub => a.value - b.value,
        BinaryOp::Mul => a.value * b.value,
        BinaryOp::Div => {
            if b.value == 0.0 {
                return None;
            }
            a.value / b.value
        }
        BinaryOp::Pow => a.value.powf(b.value),
    };
    let value = bounded(raw)?;
    Some(Elem { value, depth: 0 })
}

pub fn apply_unary(elem: &Elem, op: UnaryOp) -> Option<Elem> {
    if elem.depth >= MAX_UNARY_DEPTH {
        return None;
    }
    let raw = match op {
        UnaryOp::Neg => -elem.value,
        UnaryOp::Sqrt => {
            if elem.value < 0.0 {
                return None;
            }
            elem.value.sqrt()
        }
        UnaryOp::Ln => {
            if elem.value <= 0.0 {
                return None;
            }
            elem.value.ln()
        }
    };
    let value = bounded(raw)?;
    Some(Elem {
        value,
        depth: elem.depth + 1,
    })
}
