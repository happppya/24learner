use std::hint::black_box;
use std::time::Instant;

use core24::state::State;

fn pseudo_random_values(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sample = ((*seed >> 33) % 200_001) as f64 - 100_000.0;
            sample / 1_000.0
        })
        .collect()
}

fn main() {
    let mut seed: u64 = 20_260_825;
    println!("{:>4} {:>14} {:>16}", "N", "actions", "actions/sec");
    for n in [2usize, 4, 8, 16, 32, 64, 100] {
        let values = pseudo_random_values(n, &mut seed);
        let state = State::new(&values, 24.0);
        let iterations = (200_000 / (n * n)).max(4);
        let clock = Instant::now();
        let mut total_actions = 0u64;
        for _ in 0..iterations {
            let actions = state.legal_actions();
            total_actions += actions.len() as u64;
            for action in &actions {
                black_box(state.apply(action));
            }
        }
        let elapsed = clock.elapsed().as_secs_f64();
        println!(
            "{n:>4} {total_actions:>14} {:>16.0}",
            total_actions as f64 / elapsed
        );
    }
}
