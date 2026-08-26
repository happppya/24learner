use crate::ops::Action;

#[derive(Debug, Default, Clone, Copy)]
pub struct EdgeStats {
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
}

impl EdgeStats {
    pub fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Debug, Default)]
pub struct Node {
    pub visits: u32,
    pub children: Vec<Child>,
}

#[derive(Debug)]
pub struct Child {
    pub action: Action,
    pub stats: EdgeStats,
    pub subtree: Option<Box<Node>>,
}

pub fn puct_score(parent_visits: u32, stats: &EdgeStats, c_puct: f32) -> f32 {
    let exploitation = stats.mean_value();
    let exploration =
        c_puct * stats.prior * ((parent_visits as f32 + 1.0).sqrt() / (1.0 + stats.visits as f32));
    exploitation + exploration
}

pub fn select_child(node: &Node, c_puct: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, child) in node.children.iter().enumerate() {
        let score = puct_score(node.visits, &child.stats, c_puct);
        if best.is_none_or(|(_, top)| score > top) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::BinaryOp;

    fn sample_child(prior: f32, visits: u32, value_sum: f32) -> Child {
        Child {
            action: Action::Binary {
                i: 0,
                j: 1,
                op: BinaryOp::Add,
            },
            stats: EdgeStats {
                prior,
                visits,
                value_sum,
            },
            subtree: None,
        }
    }

    #[test]
    fn unvisited_children_rank_by_prior() {
        let node = Node {
            visits: 8,
            children: vec![sample_child(0.1, 0, 0.0), sample_child(0.9, 0, 0.0)],
        };
        assert_eq!(select_child(&node, 1.25), Some(1));
    }

    #[test]
    fn exploration_bonus_decays_with_visits() {
        let parent_visits = 100;
        let hot = EdgeStats {
            prior: 0.5,
            visits: 50,
            value_sum: 0.0,
        };
        let fresh = EdgeStats {
            prior: 0.05,
            visits: 0,
            value_sum: 0.0,
        };
        assert!(puct_score(parent_visits, &fresh, 1.0) > puct_score(parent_visits, &hot, 1.0));
    }

    #[test]
    fn empty_node_selects_nothing() {
        assert_eq!(select_child(&Node::default(), 1.0), None);
    }
}
