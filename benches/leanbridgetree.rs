use std::collections::HashSet;
use std::rc::Rc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use leanbridgetree::BridgeTree;

fn remove_mark(c: &mut Criterion) {
    c.bench_function("remove marks in batch and gc", |b| {
        let mut tree = BridgeTree::<String, 32>::new();

        // prefill tree
        for _ in 0..10000 {
            tree.append("a".to_string()).unwrap();
            tree.mark().unwrap();
        }

        b.iter_batched(
            || tree.clone(),
            |mut tree| {
                tree.remove_multiple_marks(
                    (0u64..10000).map(incrementalmerkletree::Position::from),
                )
                .unwrap();
                tree.garbage_collect_ommers();
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("remove marks individually and gc", |b| {
        let mut tree = BridgeTree::<String, 32>::new();

        // prefill tree
        for _ in 0..10000 {
            tree.append("a".to_string()).unwrap();
            tree.mark().unwrap();
        }

        b.iter_batched(
            || tree.clone(),
            |mut tree| {
                for position in (0u64..10000).map(incrementalmerkletree::Position::from) {
                    tree.remove_mark(position).unwrap();
                }
                tree.garbage_collect_ommers();
            },
            BatchSize::SmallInput,
        )
    });
    // NOTE: not worth benchmarking, this sucks
    //c.bench_function("remove marks and gc individually", |b| {
    //    let mut tree = BridgeTree::<String, 32>::new();

    //    // prefill tree
    //    for _ in 0..10000 {
    //        tree.append("a".to_string()).unwrap();
    //        tree.mark().unwrap();
    //    }

    //    b.iter_batched(
    //        || tree.clone(),
    //        |mut tree| {
    //            for position in (0u64..10000).map(incrementalmerkletree::Position::from) {
    //                tree.remove_mark_and_gc(position).unwrap();
    //            }
    //        },
    //        BatchSize::SmallInput,
    //    )
    //});
    c.bench_function("remove marks (zcash crate)", |b| {
        let mut tree = bridgetree::BridgeTree::<String, (), 32>::new(1);

        // prefill tree
        for _ in 0..10000 {
            assert!(tree.append("a".to_string()));
            tree.mark().unwrap();
        }

        b.iter_batched(
            || tree.clone(),
            |mut tree| {
                for position in (0u64..10000).map(incrementalmerkletree::Position::from) {
                    assert!(tree.remove_mark(position));
                }
                tree.garbage_collect();
            },
            BatchSize::SmallInput,
        )
    });
}

fn append(c: &mut Criterion) {
    c.bench_function("bench tree appending with marked leaves", |b| {
        let mut tree = BridgeTree::<String, 32>::new();
        let mut iter = 0usize;

        // prefill tree
        for _ in 0..10000 {
            tree.append("a".to_string()).unwrap();

            if iter & 0b11 == 0 {
                tree.mark().unwrap();
            }

            iter += 1;
        }

        b.iter(|| {
            tree.append("a".to_string()).unwrap();

            if iter & 0b11 == 0 {
                tree.mark().unwrap();
            }
            if iter & 0b1111 == 0 {
                let pos = (iter - 4) as u64;
                tree.remove_mark(pos.into()).unwrap();
            }

            iter += 1;
        })
    });
    c.bench_function(
        "bench tree appending with marked leaves (zcash crate)",
        |b| {
            let mut tree = bridgetree::BridgeTree::<String, (), 32>::new(1);
            let mut iter = 0usize;

            // prefill tree
            for _ in 0..10000 {
                assert!(tree.append("a".to_string()));

                if iter & 0b11 == 0 {
                    tree.mark().unwrap();
                }

                iter += 1;
            }

            b.iter(|| {
                assert!(tree.append("a".to_string()));

                if iter & 0b11 == 0 {
                    tree.mark().unwrap();
                }
                if iter & 0b1111 == 0 {
                    let pos = (iter - 4) as u64;
                    assert!(tree.remove_mark(pos.into()));
                }

                iter += 1;
            })
        },
    );
}

fn update(c: &mut Criterion) {
    for tree_size in [1000u64, 10_000, 50_000, 100_000] {
        c.bench_function(
            &format!("build the tree ({tree_size} leaves) by calling append"),
            |b| {
                let marked_pos = Rc::new(
                    (0u64..tree_size)
                        .filter(|pos| pos & 0b11 == 0)
                        .map(incrementalmerkletree::Position::from)
                        .collect::<HashSet<_>>(),
                );

                b.iter_batched(
                    || marked_pos.clone(),
                    |marked_pos| {
                        let mut tree = BridgeTree::<String, 32>::new();

                        for position in (0u64..tree_size).map(incrementalmerkletree::Position::from)
                        {
                            tree.append("a".to_string()).unwrap();
                            if marked_pos.contains(&position) {
                                tree.mark().unwrap();
                            }
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        c.bench_function(
            &format!("build the tree ({tree_size} leaves) by calling update"),
            |b| {
                let marked_pos = Rc::new(
                    (0u64..tree_size)
                        .filter(|pos| pos & 0b11 == 0)
                        .map(incrementalmerkletree::Position::from)
                        .collect::<Vec<_>>(),
                );
                let mut frontiers = Vec::with_capacity(tree_size.try_into().unwrap());
                let mut frontier = incrementalmerkletree::frontier::NonEmptyFrontier::from_parts(
                    0u64.into(),
                    "a".to_string(),
                    vec![],
                )
                .unwrap();

                // prefill tree
                for _ in 1..tree_size {
                    frontiers.push(frontier.clone());
                    frontier.append("a".to_string());
                }

                drop(frontier);
                let frontiers = Rc::new(frontiers);

                b.iter_batched(
                    || (frontiers.clone(), marked_pos.clone()),
                    |(frontiers, marked_pos)| {
                        let mut tree = BridgeTree::<String, 32>::new();

                        unsafe {
                            tree.update(&frontiers, &marked_pos).unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(10));
    targets = append, remove_mark, update
);
criterion_main!(benches);
