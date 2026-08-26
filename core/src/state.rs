use crate::ops::{Action, BinaryOp, UnaryOp, apply_binary, apply_unary};
use crate::value::{Elem, SUCCESS_BONUS, near_target};

#[derive(Debug, Clone)]
pub struct State {
    pub elems: Vec<Elem>,
    pub target: f64,
    pub steps: u32,
}

impl State {
    pub fn new(values: &[f64], target: f64) -> Self {
        Self {
            elems: values.iter().map(|v| Elem::new(*v)).collect(),
            target,
            steps: 0,
        }
    }

    pub fn solved(&self) -> bool {
        self.elems.iter().any(|e| near_target(e.value, self.target))
    }

    pub fn min_distance(&self) -> f64 {
        self.elems
            .iter()
            .map(|e| (e.value - self.target).abs())
            .fold(f64::INFINITY, f64::min)
    }

    pub fn shaped_reward(&self, lambda: f64) -> f64 {
        -(1.0 + self.min_distance()).ln() - lambda * f64::from(self.steps)
    }

    pub fn terminal_reward(&self) -> Option<f64> {
        if self.solved() {
            Some(SUCCESS_BONUS)
        } else {
            None
        }
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        let n = self.elems.len();
        let mut actions = Vec::with_capacity(3 * n * (n - 1) / 2 + 3 * n);
        for i in 0..n {
            for op in [UnaryOp::Neg, UnaryOp::Sqrt, UnaryOp::Ln] {
                actions.push(Action::Unary { i, op });
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for op in [BinaryOp::Add, BinaryOp::Mul] {
                    actions.push(Action::Binary { i, j, op });
                }
                for op in [BinaryOp::Sub, BinaryOp::Div, BinaryOp::Pow] {
                    actions.push(Action::Binary { i, j, op });
                    actions.push(Action::Binary { i: j, j: i, op });
                }
            }
        }
        actions
    }

    pub fn apply(&self, action: &Action) -> Option<State> {
        let mut elems = self.elems.clone();
        match *action {
            Action::Binary { i, j, op } => {
                let merged = apply_binary(&elems[i], &elems[j], op)?;
                if i < j {
                    elems[i] = merged;
                    elems.remove(j);
                } else {
                    elems[j] = merged;
                    elems.remove(i);
                }
            }
            Action::Unary { i, op } => {
                elems[i] = apply_unary(&elems[i], op)?;
            }
        }
        Some(Self {
            elems,
            target: self.target,
            steps: self.steps + 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::MAX_UNARY_DEPTH;

    fn solvable(state: &State, budget: u32) -> bool {
        if state.solved() {
            return true;
        }
        if budget == 0 {
            return false;
        }
        state
            .legal_actions()
            .iter()
            .filter_map(|a| state.apply(a))
            .any(|next| solvable(&next, budget - 1))
    }

    #[test]
    fn finds_twenty_four_from_two_three_four() {
        let state = State::new(&[2.0, 3.0, 4.0], 24.0);
        assert!(solvable(&state, 2));
    }

    #[test]
    fn solves_nested_fraction_case() {
        let state = State::new(&[1.0, 3.0, 4.0, 6.0], 24.0);
        assert!(solvable(&state, 3));
    }

    #[test]
    fn unary_depth_cap_blocks_fourth_application() {
        let state = State::new(&[16.0], 2.0);
        let s1 = state
            .apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt,
            })
            .unwrap();
        let s2 = s1
            .apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt,
            })
            .unwrap();
        let s3 = s2
            .apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt,
            })
            .unwrap();
        assert!((s2.elems[0].value - 2.0).abs() < 1e-12);
        assert!((s3.elems[0].value - 2.0f64.sqrt()).abs() < 1e-12);
        assert_eq!(s3.elems[0].depth, MAX_UNARY_DEPTH);
        assert!(
            s3.apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt
            })
            .is_none()
        );
    }

    #[test]
    fn magnitude_guard_rejects_explosion() {
        let state = State::new(&[1e7, 1e7], 1.0);
        let boom = state.apply(&Action::Binary {
            i: 0,
            j: 1,
            op: BinaryOp::Mul,
        });
        assert!(boom.is_none());
    }

    #[test]
    fn domain_guards_reject_invalid_inputs() {
        let negative = State::new(&[-4.0], 2.0);
        assert!(
            negative
                .apply(&Action::Unary {
                    i: 0,
                    op: UnaryOp::Sqrt
                })
                .is_none()
        );

        let zero = State::new(&[0.0], 2.0);
        assert!(
            zero.apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Ln
            })
            .is_none()
        );

        let divisor = State::new(&[1.0, 0.0], 2.0);
        assert!(
            divisor
                .apply(&Action::Binary {
                    i: 0,
                    j: 1,
                    op: BinaryOp::Div
                })
                .is_none()
        );
    }

    #[test]
    fn reward_shaping_orders_states_by_distance() {
        let near = State::new(&[23.999], 24.0);
        let far = State::new(&[-100.0], 24.0);
        assert!(near.shaped_reward(1e-2) > far.shaped_reward(1e-2));
        let distance: f64 = 24.0 - 23.999;
        let expected_zero_lambda = -(1.0 + distance).ln();
        assert!((near.shaped_reward(0.0) - expected_zero_lambda).abs() < 1e-12);
    }

    #[test]
    fn terminal_reward_pays_only_on_success() {
        assert_eq!(
            State::new(&[24.0], 24.0).terminal_reward(),
            Some(SUCCESS_BONUS)
        );
        assert_eq!(State::new(&[23.0], 24.0).terminal_reward(), None);
    }

    #[test]
    fn action_counts_match_formula() {
        let values = [1.0f64; 5];
        let state = State::new(&values, 24.0);
        let pairs = 5 * (5 - 1) / 2;
        let expected = 3 * 5 + pairs * (2 + 6);
        assert_eq!(state.legal_actions().len(), expected);
    }

    #[test]
    fn binary_reset_and_unary_advance_depth() {
        let state = State::new(&[16.0, 2.0], 18.0);
        let deepened = state
            .apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt,
            })
            .unwrap()
            .apply(&Action::Unary {
                i: 0,
                op: UnaryOp::Sqrt,
            })
            .unwrap();
        assert_eq!(deepened.elems[0].depth, 2);
        let merged = deepened
            .apply(&Action::Binary {
                i: 0,
                j: 1,
                op: BinaryOp::Add,
            })
            .unwrap();
        assert_eq!(merged.elems.len(), 1);
        assert_eq!(merged.elems[0].depth, 0);
        assert!((merged.elems[0].value - 4.0).abs() < 1e-12);
    }
}
