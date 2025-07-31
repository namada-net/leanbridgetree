use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use leanbridgetree::BridgeTree;

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

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(10));
    targets = append
);
criterion_main!(benches);
