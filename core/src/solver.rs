use std::collections::HashSet;

use crate::ops::Action;
use crate::state::State;

type Fingerprint = Vec<(u64, u8)>;

fn fingerprint(state: &State) -> Fingerprint {
    let mut items: Fingerprint = state
        .elems
        .iter()
        .map(|e| (e.value.to_bits(), e.depth))
        .collect();
    items.sort_unstable();
    items
}

pub fn solve_plan(start: &State, node_budget: u64) -> Option<Vec<Action>> {
    let mut failed: HashSet<Fingerprint> = HashSet::new();
    dfs(start, node_budget, &mut failed).map(|mut plan| {
        plan.reverse();
        plan
    })
}

pub fn plan_solves(start: &State, plan: &[Action]) -> bool {
    let mut current = start.clone();
    for action in plan {
        match current.apply(action) {
            Some(next) => current = next,
            None => return false,
        }
    }
    current.solved()
}

fn dfs(state: &State, budget: u64, failed: &mut HashSet<Fingerprint>) -> Option<Vec<Action>> {
    if state.solved() {
        return Some(Vec::new());
    }
    if budget == 0 {
        return None;
    }
    let key = fingerprint(state);
    if failed.contains(&key) {
        return None;
    }
    for action in state.legal_actions() {
        if let Some(next) = state.apply(&action)
            && let Some(mut plan) = dfs(&next, budget - 1, failed)
        {
            plan.push(action);
            return Some(plan);
        }
    }
    failed.insert(key);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::UnaryOp;

    #[test]
    fn solves_classic_fraction_case_with_replayable_plan() {
        let start = State::new(&[1.0, 3.0, 4.0, 6.0], 24.0);
        let plan = solve_plan(&start, 200_000).expect("instance is solvable");
        assert!(!plan.is_empty());
        assert!(plan_solves(&start, &plan));
    }

    #[test]
    fn proves_simple_instance_unsolvable() {
        let start = State::new(&[1.0, 1.0, 1.0], 5.0);
        assert!(solve_plan(&start, 50_000).is_none());
    }

    #[test]
    fn uses_unary_chain_when_binary_only_fails() {
        let start = State::new(&[16.0, 20.0], 24.0);
        let plan = solve_plan(&start, 20_000).expect("sqrt(16)+20 reaches 24");
        assert!(plan.iter().any(|a| matches!(
            a,
            Action::Unary {
                op: UnaryOp::Sqrt,
                ..
            }
        )));
        assert!(plan_solves(&start, &plan));
    }

    #[test]
    fn respects_node_budget() {
        let start = State::new(&[2.0, 3.0, 5.0], 30.0);
        assert!(solve_plan(&start, 1).is_none());
        assert!(solve_plan(&start, 2).is_some());
    }

    #[test]
    fn memoization_preserves_completeness_on_symmetric_instances() {
        let start = State::new(&[4.0, 6.0, 6.0, 1.0], 25.0);
        let plan = solve_plan(&start, 200_000).expect("solvable via 6*4+1");
        assert!(plan_solves(&start, &plan));
    }
}
