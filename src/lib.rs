//! # `leanbridgetree`
//!
//! This crate provides an implementation of an append-only Merkle tree structure. Individual
//! leaves of the merkle tree may be marked such that witnesses will be maintained for the marked
//! leaves as additional nodes are appended to the tree, but leaf and node data not specifically
//! required to maintain these witnesses is not retained, for space efficiency.
//!
//! The crate also supports using "bridges" containing the minimal possible amount of data to
//! advance witnesses for marked leaves data up to recent checkpoints or the the latest state of
//! the tree without having to append each intermediate leaf individually, given a bridge between
//! the desired states computed by an outside source. The state of the tree is internally
//! represented as a set of such bridges.
//!
//! ## Marking
//!
//! Merkle trees can be used to show that a value exists in the tree by providing a witness
//! to a leaf value. We provide an API that allows us to mark the current leaf as a value we wish
//! to compute witnesses for even after the tree has been appended to in the future; this is called
//! maintaining a witness. When we're later no longer in a leaf, we can remove the mark and drop
//! the now unnecessary information from the structure.
//!
//! ## Glossary
//!
//! - Frontier – leading leaf node in a binary tree.
//! - Ommer – sibling of a parent node in a binary tree.
//! - Bridge - past frontier and ommer data necessary to witness leaves before the latest frontier.

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::Debug;

pub use incrementalmerkletree::{
    Address, Hashable, Level, Position, Source, frontier::NonEmptyFrontier,
};

#[cfg(feature = "display-error")]
use thiserror::Error;

/// Sparse representation of a Merkle tree with linear appending of leaves that contains enough
/// information to produce a witness for any [marked](BridgeTree::mark) leaf.
#[derive(Default, Clone, PartialEq, Eq)]
pub struct BridgeTree<H, const DEPTH: u8> {
    /// The current (mutable) frontier of the tree.
    frontier: Option<NonEmptyFrontier<H>>,
    /// Ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf.
    prior_bridges: Vec<NonEmptyFrontier<H>>,
    /// Set of addresses for which we are waiting to discover the ommers.  The values of this
    /// set and the keys of the `ommers` map should always be disjoint. Also, this set should
    /// never contain an address for which the sibling value has been discovered; at that point,
    /// the address is replaced in this set with its parent and the address/sibling pair is stored
    /// in `ommers`.
    ///
    /// Another way to consider the contents of this set is that the values that exist in
    /// `ommers`, combined with the values in previous bridges' `ommers` and an original leaf
    /// node, already contain all the values needed to compute the value at the given address.
    /// Therefore, we are tracking that address as we do not yet have enough information to compute
    /// its sibling without filling the sibling subtree with empty nodes.
    tracking: BTreeSet<Address>,
    /// Map from addresses that were being tracked to the values of their ommers that have been
    /// discovered while scanning this bridge's range by adding leaves to the bridge's frontier.
    ommers: BTreeMap<Address, H>,
}

impl<H: Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        // NB: We destructure `self` in order to catch compiler errors, should
        // its internal structure ever change. See note below.
        let Self {
            frontier,
            prior_bridges,
            tracking,
            ommers,
        } = self;

        // XXX: Keep me up to date!
        f.debug_struct(stringify!(BridgeTree))
            .field("max_depth", &DEPTH) // virtual field
            .field("frontier", frontier)
            .field("prior_bridges", prior_bridges)
            .field("tracking", tracking)
            .field("ommers", ommers)
            .finish()
    }
}

/// [`BridgeTree`] related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "display-error", derive(Error))]
pub enum BridgeTreeError {
    /// The tree is full (i.e. has reached is maximum depth) and can no longer acccept new leaves.
    #[cfg_attr(
        feature = "display-error",
        error("Bridge tree has reached its maximum depth and can no longer acccept new leaves")
    )]
    FullTree,
    /// The bridges aren't contiguous (i.e. their frontier positions aren't monotonically
    /// increasing).
    #[cfg_attr(
        feature = "display-error",
        error("Non-monotonically increasing bridge leaf positions")
    )]
    Discontinuity,
    /// The requested position is not tracked, therefore we can't generate its witnesses.
    #[cfg_attr(
        feature = "display-error",
        error("{0:?} is not tracked in any of the tree's bridges")
    )]
    PositionNotMarked(Position),
    /// The tree is missing the data of an ommer at the given address.
    #[cfg_attr(feature = "display-error", error("Missing ommer data of {0:?}"))]
    MissingOmmer(Address),
}

impl<H, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct an empty [`BridgeTree`].
    pub const fn new() -> Self {
        Self {
            prior_bridges: Vec::new(),
            tracking: BTreeSet::new(),
            ommers: BTreeMap::new(),
            frontier: None,
        }
    }

    /// Return the prior bridges that make up this tree
    pub fn prior_bridges(&self) -> &[NonEmptyFrontier<H>] {
        &self.prior_bridges
    }

    /// Return the bridge's frontier.
    pub fn frontier(&self) -> Option<&NonEmptyFrontier<H>> {
        self.frontier.as_ref()
    }

    /// Returns the most recently appended leaf value's position.
    pub fn current_position(&self) -> Option<Position> {
        self.frontier().map(|f| f.position())
    }

    /// Returns the most recently appended leaf value.
    pub fn current_leaf(&self) -> Option<&H> {
        self.frontier().map(|f| f.leaf())
    }

    /// Returns the size of the Merkle tree that this frontier corresponds to.
    ///
    /// The returned values corresponds to the number of leaves appended to the
    /// tree.
    pub fn tree_size(&self) -> u64 {
        self.frontier
            .as_ref()
            .map_or(0, |f| u64::from(f.position()) + 1)
    }

    /// Construct a new [`BridgeTree`] that will start recording changes from the state of
    /// the specified frontier.
    pub fn from_frontier(frontier: NonEmptyFrontier<H>) -> Result<Self, BridgeTreeError> {
        if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
            return Err(BridgeTreeError::FullTree);
        }

        Ok(Self {
            frontier: Some(frontier),
            prior_bridges: Vec::new(),
            tracking: BTreeSet::new(),
            ommers: BTreeMap::new(),
        })
    }

    /// Construct a new [`BridgeTree`] from its constituent parts, checking for internal
    /// consistency.
    pub fn from_parts(
        frontier: Option<NonEmptyFrontier<H>>,
        mut prior_bridges: Vec<NonEmptyFrontier<H>>,
        tracking: BTreeSet<Address>,
        ommers: BTreeMap<Address, H>,
    ) -> Result<Self, BridgeTreeError> {
        Self::check_consistency_internal(&prior_bridges, frontier.as_ref())?;

        // Remove duplicated entries in the marked leaves.
        prior_bridges.dedup_by_key(|bridge_frontier| bridge_frontier.position());

        Ok(Self {
            frontier,
            prior_bridges,
            tracking,
            ommers,
        })
    }

    /// Return an iterator over all the marked leaf positions.
    pub fn marked_positions(&self) -> impl Iterator<Item = Position> + '_ {
        self.prior_bridges
            .iter()
            .map(|bridge_frontier| bridge_frontier.position())
    }

    /// Returns the leaf at the specified position if the tree can produce
    /// a witness for it.
    pub fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        let index = self.lookup_prior_bridge(position).ok()?;
        Some(self.prior_bridges[index].leaf())
    }

    /// Verify the integrity of the [`BridgeTree`].
    fn check_consistency_internal(
        prior_bridges: &[NonEmptyFrontier<H>],
        current_bridge: Option<&NonEmptyFrontier<H>>,
    ) -> Result<(), BridgeTreeError> {
        // Make sure the frontier hasn't reached the maximum depth.
        if let Some(frontier) = current_bridge {
            if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
                return Err(BridgeTreeError::FullTree);
            }
        }

        // Make sure the bridges are ordered in ascending order,
        // keyed by their frontier leaf's position.
        for (prev, next) in prior_bridges.iter().zip(prior_bridges.iter().skip(1)) {
            if prev.position() > next.position() {
                return Err(BridgeTreeError::Discontinuity);
            }
        }
        if let Some((prev, next)) = prior_bridges.last().zip(current_bridge) {
            if prev.position() > next.position() {
                return Err(BridgeTreeError::Discontinuity);
            }
        }

        Ok(())
    }

    /// Look-up a prior bridge, given the position of a marked leaf.
    ///
    /// Returns the index of the bridge if `Ok`, or where a new bridge
    /// could be inserted, if `Err`.
    #[inline]
    fn lookup_prior_bridge(&self, position: Position) -> Result<usize, usize> {
        self.prior_bridges
            .binary_search_by_key(&position, |bridge_frontier| bridge_frontier.position())
    }
}

impl<H: Hashable + Clone, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Append a new leaf to the tree at the next available slot.
    ///
    /// Returns an error if the tree would exceed the maximum allowed depth.
    pub fn append(&mut self, leaf: H) -> Result<(), BridgeTreeError> {
        let frontier = if let Some(frontier) = self.frontier.as_mut() {
            if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
                return Err(unlikely(|| BridgeTreeError::FullTree));
            }
            frontier
        } else {
            self.frontier = unlikely(|| Some(NonEmptyFrontier::new(leaf)));
            return Ok(());
        };

        frontier.append(leaf);

        let mut found = Vec::new();
        for address in self.tracking.iter() {
            // We know that there will only ever be one address that we're
            // tracking at a given level, because as soon as we find a
            // value for the sibling of the address we're tracking, we
            // remove the tracked address and replace it the next parent
            // of that address for which we need to find a sibling.
            if frontier.position().is_complete_subtree(address.level()) {
                let digest = frontier.root(Some(address.level()));
                self.ommers.insert(address.sibling(), digest);
                found.push(*address);
            }
        }

        for address in found {
            self.tracking.remove(&address);

            // The address of the next incomplete parent note for which
            // we need to find a sibling.
            let parent = address.next_incomplete_parent();
            assert!(!self.ommers.contains_key(&parent));
            self.tracking.insert(parent);
        }

        Ok(())
    }

    /// Obtain the root of the Merkle tree at the maximum depth.
    #[inline]
    pub fn root(&self) -> H {
        self.root_at_depth(DEPTH.into())
    }

    /// Obtain the root of the Merkle tree at the specified level.
    pub fn root_at_depth(&self, depth: Level) -> H {
        self.frontier.as_ref().map_or_else(
            || unlikely(|| H::empty_root(depth)),
            |frontier| frontier.root(Some(depth)),
        )
    }

    /// Marks the current leaf as one for which we're interested in producing a witness.
    ///
    /// Returns an optional value containing the current position if successful or if the current
    /// value was already marked, or None if the tree is empty.
    pub fn mark(&mut self) -> Option<Position> {
        let frontier = self.frontier.as_ref()?;

        let last_leaf_already_marked = self
            .prior_bridges
            .last()
            .is_some_and(|bridge_frontier| bridge_frontier.position() == frontier.position());

        if !last_leaf_already_marked {
            self.prior_bridges.push(frontier.clone());
            self.tracking
                .insert(Address::from(frontier.position()).current_incomplete());
        } else {
            // NB: Simply mark the cold path.
            unlikely(|| {});
        }

        Some(frontier.position())
    }

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for.
    ///
    /// Returns an error if we were already not maintaining a mark at this position.
    pub fn remove_mark(&mut self, position: Position) -> Result<(), BridgeTreeError> {
        // Figure out where the marked leaf is in the vector of bridges.
        let index_of_marked_leaf_bridge = self
            .lookup_prior_bridge(position)
            .map_err(|_| unlikely(|| BridgeTreeError::PositionNotMarked(position)))?;

        // Then remove it. This shifts all bridges following `index_of_marked_leaf_bridge`
        // in the vector in O(n) time.
        self.prior_bridges.remove(index_of_marked_leaf_bridge);

        // Let's also get rid of the tracking data.
        let ommer_addrs: BTreeSet<_> = self
            .prior_bridges
            .iter()
            .flat_map(|prior_bridge_frontier| {
                prior_bridge_frontier
                    .position()
                    .witness_addrs(Level::from(DEPTH))
                    .filter_map(|(addr, source)| {
                        if source == Source::Future {
                            Some(addr)
                        } else {
                            None
                        }
                    })
            })
            .collect();
        self.tracking
            .retain(|addr| ommer_addrs.contains(&addr.sibling()));
        self.ommers.retain(|addr, _| ommer_addrs.contains(addr));

        Ok(())
    }

    /// Obtains a witness for the value at the specified leaf position.
    ///
    /// Returns an error if there is no witness information for the requested
    /// position.
    pub fn witness(&self, position: Position) -> Result<Vec<H>, BridgeTreeError> {
        let current_frontier = self
            .frontier
            .as_ref()
            .ok_or(BridgeTreeError::PositionNotMarked(position))?;

        let saved_idx = self
            .lookup_prior_bridge(position)
            .map_err(|_| unlikely(|| BridgeTreeError::PositionNotMarked(position)))?;

        let prior_frontier = &self.prior_bridges[saved_idx];

        prior_frontier
            .witness(DEPTH, |addr| {
                let r = addr.position_range();

                if current_frontier.position() < r.start {
                    Some(H::empty_root(addr.level()))
                } else if r.contains(&current_frontier.position()) {
                    Some(current_frontier.root(Some(addr.level())))
                } else {
                    // the frontier's position is after the end of the requested
                    // range, so the requested value should exist in a stored
                    // fragment
                    self.ommers.get(&addr).cloned()
                }
            })
            .map_err(BridgeTreeError::MissingOmmer)
    }
}

/// Utility that can be used to mark cold paths in
/// if-branches, in order to optimize codegen.
#[cold]
#[inline(never)]
fn unlikely<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use incrementalmerkletree::Retention;
    use incrementalmerkletree_testing::{
        self as testing, SipHashable, TestHashable, apply_operation, arb_operation,
        check_operations,
    };
    use proptest::prelude::*;

    use super::*;

    enum DynDepthBridgeTree<H> {
        Depth0(BridgeTree<H, 0>),
        Depth1(BridgeTree<H, 1>),
        Depth2(BridgeTree<H, 2>),
        Depth3(BridgeTree<H, 3>),
        Depth4(BridgeTree<H, 4>),
        Depth5(BridgeTree<H, 5>),
        Depth6(BridgeTree<H, 6>),
        Depth7(BridgeTree<H, 7>),
        Depth8(BridgeTree<H, 8>),
        Depth9(BridgeTree<H, 9>),
        Depth10(BridgeTree<H, 10>),
        Depth11(BridgeTree<H, 11>),
        Depth12(BridgeTree<H, 12>),
        Depth13(BridgeTree<H, 13>),
        Depth14(BridgeTree<H, 14>),
        Depth15(BridgeTree<H, 15>),
    }

    impl<H> DynDepthBridgeTree<H> {
        const fn new(depth: u8) -> Self {
            match depth & 0xf {
                0 => Self::Depth0(BridgeTree::new()),
                1 => Self::Depth1(BridgeTree::new()),
                2 => Self::Depth2(BridgeTree::new()),
                3 => Self::Depth3(BridgeTree::new()),
                4 => Self::Depth4(BridgeTree::new()),
                5 => Self::Depth5(BridgeTree::new()),
                6 => Self::Depth6(BridgeTree::new()),
                7 => Self::Depth7(BridgeTree::new()),
                8 => Self::Depth8(BridgeTree::new()),
                9 => Self::Depth9(BridgeTree::new()),
                10 => Self::Depth10(BridgeTree::new()),
                11 => Self::Depth11(BridgeTree::new()),
                12 => Self::Depth12(BridgeTree::new()),
                13 => Self::Depth13(BridgeTree::new()),
                14 => Self::Depth14(BridgeTree::new()),
                15 => Self::Depth15(BridgeTree::new()),
                _ => unreachable!(),
            }
        }
    }

    #[allow(dead_code)]
    impl<H: Clone + Hashable> DynDepthBridgeTree<H> {
        fn get(&self, max_depth: u8) -> &dyn testing::Tree<H, ()> {
            match (max_depth & 0xf, self) {
                (0, DynDepthBridgeTree::Depth0(t)) => t,
                (1, DynDepthBridgeTree::Depth1(t)) => t,
                (2, DynDepthBridgeTree::Depth2(t)) => t,
                (3, DynDepthBridgeTree::Depth3(t)) => t,
                (4, DynDepthBridgeTree::Depth4(t)) => t,
                (5, DynDepthBridgeTree::Depth5(t)) => t,
                (6, DynDepthBridgeTree::Depth6(t)) => t,
                (7, DynDepthBridgeTree::Depth7(t)) => t,
                (8, DynDepthBridgeTree::Depth8(t)) => t,
                (9, DynDepthBridgeTree::Depth9(t)) => t,
                (10, DynDepthBridgeTree::Depth10(t)) => t,
                (11, DynDepthBridgeTree::Depth11(t)) => t,
                (12, DynDepthBridgeTree::Depth12(t)) => t,
                (13, DynDepthBridgeTree::Depth13(t)) => t,
                (14, DynDepthBridgeTree::Depth14(t)) => t,
                (15, DynDepthBridgeTree::Depth15(t)) => t,
                _ => panic!("called get on tree of invalid depth"),
            }
        }

        fn get_mut(&mut self, max_depth: u8) -> &mut dyn testing::Tree<H, ()> {
            match (max_depth & 0xf, self) {
                (0, DynDepthBridgeTree::Depth0(t)) => t,
                (1, DynDepthBridgeTree::Depth1(t)) => t,
                (2, DynDepthBridgeTree::Depth2(t)) => t,
                (3, DynDepthBridgeTree::Depth3(t)) => t,
                (4, DynDepthBridgeTree::Depth4(t)) => t,
                (5, DynDepthBridgeTree::Depth5(t)) => t,
                (6, DynDepthBridgeTree::Depth6(t)) => t,
                (7, DynDepthBridgeTree::Depth7(t)) => t,
                (8, DynDepthBridgeTree::Depth8(t)) => t,
                (9, DynDepthBridgeTree::Depth9(t)) => t,
                (10, DynDepthBridgeTree::Depth10(t)) => t,
                (11, DynDepthBridgeTree::Depth11(t)) => t,
                (12, DynDepthBridgeTree::Depth12(t)) => t,
                (13, DynDepthBridgeTree::Depth13(t)) => t,
                (14, DynDepthBridgeTree::Depth14(t)) => t,
                (15, DynDepthBridgeTree::Depth15(t)) => t,
                _ => panic!("called get on tree of invalid depth"),
            }
        }
    }

    impl<H: Hashable + Clone, const DEPTH: u8> testing::Tree<H, ()> for BridgeTree<H, DEPTH> {
        fn append(&mut self, value: H, retention: Retention<()>) -> bool {
            let appended = BridgeTree::append(self, value).is_ok();
            if appended && retention.is_marked() {
                BridgeTree::mark(self).unwrap();
            }
            appended
        }

        fn depth(&self) -> u8 {
            DEPTH
        }

        fn current_position(&self) -> Option<Position> {
            BridgeTree::current_position(self)
        }

        fn get_marked_leaf(&self, position: Position) -> Option<H> {
            BridgeTree::get_marked_leaf(self, position).cloned()
        }

        fn marked_positions(&self) -> BTreeSet<Position> {
            BridgeTree::marked_positions(self).collect()
        }

        fn root(&self, _checkpoint_depth: usize) -> Option<H> {
            Some(BridgeTree::root(self))
        }

        fn witness(&self, position: Position, _checkpoint_depth: usize) -> Option<Vec<H>> {
            BridgeTree::witness(self, position).ok()
        }

        fn remove_mark(&mut self, position: Position) -> bool {
            BridgeTree::remove_mark(self, position).is_ok()
        }

        fn checkpoint(&mut self, _id: ()) -> bool {
            false
        }

        fn rewind(&mut self) -> bool {
            false
        }
    }

    fn arb_bridgetree<G: Strategy + Clone>(
        item_gen: G,
        max_count: usize,
    ) -> impl Strategy<Value = BridgeTree<G::Value, 8>>
    where
        G::Value: Hashable + Clone + Debug + 'static,
    {
        let pos_gen = (0..max_count).prop_map(|p| Position::try_from(p).unwrap());
        proptest::collection::vec(arb_operation(item_gen, pos_gen), 0..max_count).prop_map(|ops| {
            let mut tree: BridgeTree<G::Value, 8> = BridgeTree::new();
            for op in ops {
                apply_operation(&mut tree, op.map_checkpoint_id(|_| ()));
            }
            tree
        })
    }

    proptest! {
        #[test]
        fn bridgetree_from_parts(
            tree in arb_bridgetree((97u8..123).prop_map(|c| char::from(c).to_string()), 100)
        ) {
            let tree_from_parts = BridgeTree::from_parts(
                tree.frontier.as_ref().cloned(),
                tree.prior_bridges.clone(),
                tree.tracking.clone(),
                tree.ommers.clone(),
            )
            .unwrap();

            assert_eq!(tree_from_parts, tree);
        }

        #[test]
        fn tree_depth(max_depth in 0u8..16) {
            let mut t = DynDepthBridgeTree::<String>::new(max_depth);
            let max_leaves = 1u64 << max_depth;
            for i in 0..max_leaves {
                assert!(t.get_mut(max_depth).append(format!("{i}"), Retention::Ephemeral));
            }
            assert!(!t.get_mut(max_depth).append(format!("{max_depth}"), Retention::Ephemeral));
        }

        #[test]
        fn compare_against_upstream_bridgetree(
            values in proptest::collection::vec((0..=25u64, any::<bool>()), 1..256)
        )
        {
            let mut upstream = bridgetree::BridgeTree::<String, (), 8>::new(1);
            let mut forked = BridgeTree::<String, 8>::new();

            let mut marked_pos = vec![];

            for (pos, (value, mark)) in values.into_iter().enumerate() {
                assert!(upstream.append(String::from_u64(value)));
                assert!(forked.append(String::from_u64(value)).is_ok());

                if mark {
                    assert!(upstream.mark().is_some());
                    assert!(forked.mark().is_some());

                    marked_pos.push(pos as u64);
                }
            }

            // check the roots are identical
            assert_eq!(
                upstream.root(0).unwrap(),
                forked.root(),
            );

            // check that witnesses are identical
            for pos in marked_pos {
                assert_eq!(
                    upstream.witness(pos.into(), 0).unwrap(),
                    forked.witness(pos.into()).unwrap(),
                );
            }
        }
    }

    fn new_tree<H: Hashable + Clone + Debug>() -> BridgeTree<H, 4> {
        BridgeTree::new()
    }

    proptest! {
        //#![proptest_config(ProptestConfig::with_cases(100000))]

        #[test]
        fn check_randomized_u64_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (0..32u64).prop_map(SipHashable),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_tree();
            let ops = ops
                .into_iter()
                .map(|op| {
                    match op {
                        incrementalmerkletree_testing::Operation::Witness(pos, _)
                            => incrementalmerkletree_testing::Operation::Witness(pos, 0),
                        op => op,
                    }
                })
                .collect::<Vec<_>>();
            check_operations(tree, &ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (97u8..123).prop_map(|c| char::from(c).to_string()),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_tree();
            let ops = ops
                .into_iter()
                .map(|op| {
                    match op {
                        incrementalmerkletree_testing::Operation::Witness(pos, _)
                            => incrementalmerkletree_testing::Operation::Witness(pos, 0),
                        op => op,
                    }
                })
                .collect::<Vec<_>>();
            check_operations(tree, &ops)?;
        }
    }

    #[test]
    fn root_hashes() {
        {
            let mut tree = new_tree::<String>();
            tree.assert_root(&[]);
            tree.assert_append(0, incrementalmerkletree::Retention::Ephemeral);
            tree.assert_root(&[0]);
            tree.assert_append(1, incrementalmerkletree::Retention::Ephemeral);
            tree.assert_root(&[0, 1]);
            tree.assert_append(2, incrementalmerkletree::Retention::Ephemeral);
            tree.assert_root(&[0, 1, 2]);
        }

        {
            let mut t = new_tree::<String>();
            t.assert_append(0, Retention::Marked);
            for _ in 0..3 {
                t.assert_append(0, incrementalmerkletree::Retention::Ephemeral);
            }
            t.assert_root(&[0, 0, 0, 0]);
        }
    }

    #[test]
    fn witness() {
        use Retention::*;
        use incrementalmerkletree_testing::Operation;
        use incrementalmerkletree_testing::Operation::Append;
        use incrementalmerkletree_testing::Operation::Witness;

        {
            let mut tree = new_tree::<String>();
            tree.assert_append(0, Ephemeral);
            tree.assert_append(1, Marked);
            assert_eq!(testing::Tree::witness(&tree, Position::from(0), 0), None);
        }

        {
            let mut tree = new_tree::<String>();
            tree.assert_append(0, Marked);
            assert_eq!(
                testing::Tree::witness(&tree, Position::from(0), 0),
                Some(vec![
                    String::empty_root(0.into()),
                    String::empty_root(1.into()),
                    String::empty_root(2.into()),
                    String::empty_root(3.into())
                ])
            );

            tree.assert_append(1, Ephemeral);
            assert_eq!(
                testing::Tree::witness(&tree, 0.into(), 0),
                Some(vec![
                    String::from_u64(1),
                    String::empty_root(1.into()),
                    String::empty_root(2.into()),
                    String::empty_root(3.into())
                ])
            );

            tree.assert_append(2, Marked);
            assert_eq!(
                testing::Tree::witness(&tree, Position::from(2), 0),
                Some(vec![
                    String::empty_root(0.into()),
                    String::combine_all(1, &[0, 1]),
                    String::empty_root(2.into()),
                    String::empty_root(3.into())
                ])
            );

            tree.assert_append(3, Ephemeral);
            assert_eq!(
                testing::Tree::witness(&tree, Position::from(2), 0),
                Some(vec![
                    String::from_u64(3),
                    String::combine_all(1, &[0, 1]),
                    String::empty_root(2.into()),
                    String::empty_root(3.into())
                ])
            );

            tree.assert_append(4, Ephemeral);
            assert_eq!(
                testing::Tree::witness(&tree, Position::from(2), 0),
                Some(vec![
                    String::from_u64(3),
                    String::combine_all(1, &[0, 1]),
                    String::combine_all(2, &[4]),
                    String::empty_root(3.into())
                ])
            );
        }

        {
            let mut tree = new_tree::<String>();
            tree.assert_append(0, Marked);
            for i in 1..6 {
                tree.assert_append(i, Ephemeral);
            }
            tree.assert_append(6, Marked);
            tree.assert_append(7, Ephemeral);

            assert_eq!(
                testing::Tree::witness(&tree, 0.into(), 0),
                Some(vec![
                    String::from_u64(1),
                    String::combine_all(1, &[2, 3]),
                    String::combine_all(2, &[4, 5, 6, 7]),
                    String::empty_root(3.into())
                ])
            );
        }

        {
            let mut tree = new_tree::<String>();
            tree.assert_append(0, Marked);
            tree.assert_append(1, Ephemeral);
            tree.assert_append(2, Ephemeral);
            tree.assert_append(3, Marked);
            tree.assert_append(4, Marked);
            tree.assert_append(5, Marked);
            tree.assert_append(6, Ephemeral);

            assert_eq!(
                testing::Tree::witness(&tree, Position::from(5), 0),
                Some(vec![
                    String::from_u64(4),
                    String::combine_all(1, &[6]),
                    String::combine_all(2, &[0, 1, 2, 3]),
                    String::empty_root(3.into())
                ])
            );
        }

        {
            let mut tree = new_tree::<String>();
            for i in 0..10 {
                tree.assert_append(i, Ephemeral);
            }
            tree.assert_append(10, Marked);
            tree.assert_append(11, Ephemeral);

            assert_eq!(
                testing::Tree::witness(&tree, Position::from(10), 0),
                Some(vec![
                    String::from_u64(11),
                    String::combine_all(1, &[8, 9]),
                    String::empty_root(2.into()),
                    String::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7])
                ])
            );
        }

        {
            let mut tree = new_tree::<String>();
            for i in 0..12 {
                tree.assert_append(i, Ephemeral);
            }
            tree.assert_append(12, Marked);
            tree.assert_append(13, Marked);
            tree.assert_append(14, Ephemeral);
            tree.assert_append(15, Ephemeral);

            assert_eq!(
                testing::Tree::witness(&tree, Position::from(12), 0),
                Some(vec![
                    String::from_u64(13),
                    String::combine_all(1, &[14, 15]),
                    String::combine_all(2, &[8, 9, 10, 11]),
                    String::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7]),
                ])
            );
        }

        {
            let ops = (0..=11)
                .map(|i| Append(String::from_u64(i), Marked))
                .chain(Some(Append(String::from_u64(12), Ephemeral)))
                .chain(Some(Append(String::from_u64(13), Ephemeral)))
                .chain(Some(Witness(11u64.into(), 0)))
                .collect::<Vec<_>>();

            let mut tree = new_tree::<String>();
            assert_eq!(
                Operation::apply_all(&ops, &mut tree),
                Some((
                    Position::from(11),
                    vec![
                        String::from_u64(10),
                        String::combine_all(1, &[8, 9]),
                        String::combine_all(2, &[12, 13]),
                        String::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7]),
                    ]
                ))
            );
        }

        {
            let ops = vec![
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(1), Ephemeral),
                Append(String::from_u64(2), Ephemeral),
                Append(String::from_u64(3), Marked),
                Append(String::from_u64(4), Marked),
                Append(String::from_u64(5), Ephemeral),
                Append(String::from_u64(6), Ephemeral),
                Append(String::from_u64(7), Ephemeral),
                Witness(3u64.into(), 4),
            ];
            let mut tree = new_tree::<String>();
            assert_eq!(
                Operation::apply_all(&ops, &mut tree),
                Some((
                    Position::from(3),
                    vec![
                        String::from_u64(2),
                        String::combine_all(1, &[0, 1]),
                        String::combine_all(2, &[4, 5, 6, 7]),
                        String::combine_all(3, &[]),
                    ]
                ))
            );
        }

        {
            let ops = vec![
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Marked),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Ephemeral),
                Witness(Position::from(3), 0),
            ];
            let mut tree = new_tree::<String>();
            assert_eq!(
                Operation::apply_all(&ops, &mut tree),
                Some((
                    Position::from(3),
                    vec![
                        String::from_u64(0),
                        String::combine_all(1, &[0, 0]),
                        String::combine_all(2, &[0, 0, 0, 0]),
                        String::combine_all(3, &[0, 0]),
                    ]
                ))
            );
        }

        {
            let ops = vec![
                Append(String::from_u64(0), Marked),
                Append(String::from_u64(0), Ephemeral),
                Append(String::from_u64(0), Marked),
                Append(String::from_u64(0), Ephemeral),
                Witness(Position::from(2), 1),
            ];
            let mut tree = new_tree::<String>();
            assert_eq!(
                Operation::apply_all(&ops, &mut tree),
                Some((
                    Position::from(2),
                    vec![
                        String::from_u64(0),
                        String::combine_all(1, &[0, 0]),
                        String::combine_all(2, &[]),
                        String::combine_all(3, &[]),
                    ]
                ))
            );
        }
    }

    trait TestTree<H: TestHashable> {
        fn assert_root(&self, values: &[u64]);

        fn assert_append(&mut self, value: u64, retention: incrementalmerkletree::Retention<()>);
    }

    impl<H: TestHashable, T: testing::Tree<H, ()>> TestTree<H> for T {
        fn assert_root(&self, values: &[u64]) {
            assert_eq!(self.root(0).unwrap(), H::combine_all(self.depth(), values));
        }

        fn assert_append(&mut self, value: u64, retention: incrementalmerkletree::Retention<()>) {
            assert!(
                self.append(H::from_u64(value), retention),
                "append failed for value {value}",
            );
        }
    }
}
