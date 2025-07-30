# `leanbridgetree`

This is a Rust crate that provides an implementation of an append-only Merkle
tree structure. Individual leaves of the merkle tree may be marked such that
witnesses will be maintained for the marked leaves as additional nodes are
appended to the tree, but leaf and node data not specifically required to
maintain these witnesses is not retained, for space efficiency.

The crate also supports using "bridges" containing the minimal possible amount
of data to advance witnesses for marked leaves data up to the latest state of
the tree without having to append each intermediate leaf individually, given
a bridge between the desired states computed by an outside source. The state
of the tree is internally represented as a set of such bridges.

## [`Documentation`](https://docs.rs/leanbridgetree)

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

## Original work

This crate is a heavily modified fork of the excellent [`bridgetree`] crate from
Zcash Zebra.

[`bridgetree`]: https://github.com/zcash/incrementalmerkletree/tree/main/bridgetree
