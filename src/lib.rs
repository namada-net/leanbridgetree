//! # `leanbridgetree`
//!
//! This crate provides an implementation of an append-only Merkle tree structure. Individual
//! leaves of the merkle tree may be marked such that witnesses will be maintained for the marked
//! leaves as additional nodes are appended to the tree, but leaf and node data not specifically
//! required to maintain these witnesses is not retained, for space efficiency. The data structure
//! also supports checkpointing of the tree state such that the tree may be reset to a previously
//! checkpointed state, up to a fixed number of checkpoints.
//!
//! The crate also supports using "bridges" containing the minimal possible amount of data to
//! advance witnesses for marked leaves data up to recent checkpoints or the the latest state of
//! the tree without having to append each intermediate leaf individually, given a bridge between
//! the desired states computed by an outside source. The state of the tree is internally
//! represented as a set of such bridges, and the data structure supports fusing and splitting of
//! bridges.
//!
//! ## Marking
//!
//! Merkle trees can be used to show that a value exists in the tree by providing a witness
//! to a leaf value. We provide an API that allows us to mark the current leaf as a value we wish
//! to compute witnesses for even after the tree has been appended to in the future; this is called
//! maintaining a witness. When we're later no longer in a leaf, we can remove the mark and drop
//! the now unnecessary information from the structure.
//!
//! In this module, the term "ommer" is used as for the sibling of a parent node in a binary tree.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::Debug;

pub use incrementalmerkletree::{
    frontier::NonEmptyFrontier, Address, Hashable, Level, Position, Retention, Source,
};

/// A sparse representation of a Merkle tree with linear appending of leaves that contains enough
/// information to produce a witness for any `mark`ed leaf.
#[derive(Default, Clone, PartialEq, Eq)]
pub struct BridgeTree<H, const DEPTH: u8> {
    /// The current (mutable) frontier of the tree.
    frontier: Option<NonEmptyFrontier<H>>,
    /// The ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf.
    prior_bridges: Vec<NonEmptyFrontier<H>>,
    /// The set of addresses for which we are waiting to discover the ommers.  The values of this
    /// set and the keys of the `need` map should always be disjoint. Also, this set should
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
    /// A map from addresses that were being tracked to the values of their ommers that have been
    /// discovered while scanning this bridge's range by adding leaves to the bridge's frontier.
    ommers: BTreeMap<Address, H>,
}

impl<H: Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        let Self {
            frontier,
            prior_bridges,
            tracking,
            ommers,
        } = self;

        f.debug_struct(stringify!(BridgeTree))
            .field("max_depth", &DEPTH)
            .field("frontier", frontier)
            .field("prior_bridges", prior_bridges)
            .field("tracking", tracking)
            .field("ommers", ommers)
            .finish()
    }
}

/// Errors that can appear when validating the internal consistency of a `[BridgeTree]`
/// value when constructing a tree from its constituent parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeTreeError {
    FullTree,
    Discontinuity,
    PositionNotMarked(Position),
    BridgeAddressInvalid(Address),
}

impl<H, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct an empty BridgeTree value.
    pub const fn new() -> Self {
        Self {
            prior_bridges: Vec::new(),
            tracking: BTreeSet::new(),
            ommers: BTreeMap::new(),
            frontier: None,
        }
    }

    /// Returns the prior bridges that make up this tree
    pub fn prior_bridges(&self) -> &[NonEmptyFrontier<H>] {
        &self.prior_bridges
    }

    /// Returns the bridge's frontier.
    pub fn frontier(&self) -> Option<&NonEmptyFrontier<H>> {
        self.frontier.as_ref()
    }
}

impl<H: Hashable + Clone + Ord, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct a new BridgeTree that will start recording changes from the state of
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

    /// Construct a new BridgeTree from its constituent parts, checking for internal
    /// consistency.
    pub fn from_parts(
        frontier: Option<NonEmptyFrontier<H>>,
        prior_bridges: Vec<NonEmptyFrontier<H>>,
        tracking: BTreeSet<Address>,
        ommers: BTreeMap<Address, H>,
    ) -> Result<Self, BridgeTreeError> {
        Self::check_consistency_internal(&prior_bridges, frontier.as_ref())?;

        Ok(Self {
            frontier,
            prior_bridges,
            tracking,
            ommers,
        })
    }

    /*
    fn check_consistency(&self) -> Result<(), BridgeTreeError> {
        Self::check_consistency_internal(&self.prior_bridges, &self.current_bridge)
    }
    */

    fn check_consistency_internal(
        prior_bridges: &[NonEmptyFrontier<H>],
        current_bridge: Option<&NonEmptyFrontier<H>>,
    ) -> Result<(), BridgeTreeError> {
        if let Some(frontier) = current_bridge {
            if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
                return Err(BridgeTreeError::FullTree);
            }
        }

        for (prev, next) in prior_bridges.iter().zip(prior_bridges.iter().skip(1)) {
            if prev.position() >= next.position() {
                return Err(BridgeTreeError::Discontinuity);
            }
        }

        if let Some((prev, next)) = prior_bridges.last().zip(current_bridge) {
            if prev.position() >= next.position() {
                return Err(BridgeTreeError::Discontinuity);
            }
        }

        Ok(())
    }

    /// Appends a new value to the tree at the next available slot.
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed depth.
    pub fn append(&mut self, value: H) -> Result<(), BridgeTreeError> {
        let frontier = if let Some(frontier) = self.frontier.as_mut() {
            if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
                return Err(BridgeTreeError::FullTree);
            }
            frontier
        } else {
            self.frontier = Some(NonEmptyFrontier::new(value));
            return Ok(());
        };

        frontier.append(value);

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

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    pub fn root(&self) -> H {
        self.frontier.as_ref().map_or_else(
            || H::empty_root(DEPTH.into()),
            |frontier| frontier.root(Some(DEPTH.into())),
        )
    }

    /// Returns the size of the Merkle tree that this frontier corresponds to.
    pub fn tree_size(&self) -> u64 {
        self.frontier
            .as_ref()
            .map_or(0, |f| u64::from(f.position()) + 1)
    }

    /// Returns the most recently appended leaf value's position.
    pub fn current_position(&self) -> Option<Position> {
        self.frontier.as_ref().map(|f| f.position())
    }

    /// Returns the most recently appended leaf value.
    pub fn current_leaf(&self) -> Option<&H> {
        self.frontier.as_ref().map(|f| f.leaf())
    }

    /// Marks the current leaf as one for which we're interested in producing a witness.
    ///
    /// Returns an optional value containing the current position if successful or if the current
    /// value was already marked, or None if the tree is empty.
    pub fn mark(&mut self) -> Option<Position> {
        let frontier = self.frontier.as_ref()?;

        if self.lookup_prior_bridge(frontier.position()).is_err() {
            self.prior_bridges.push(frontier.clone());
            self.tracking
                .insert(Address::from(frontier.position()).current_incomplete());
        }

        Some(frontier.position())
    }

    /// Return a set of all the positions for which we have marked.
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

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for. Returns true if successful and
    /// false if we were already not maintaining a mark at this position.
    pub fn remove_mark(&mut self, position: Position) -> Result<(), BridgeTreeError> {
        // Figure out where the marked leaf is in the vector of bridges.
        let index_of_marked_leaf_bridge = self
            .lookup_prior_bridge(position)
            .map_err(|_| BridgeTreeError::PositionNotMarked(position))?;

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
    /// Returns an error if there is no witness information for the requested
    /// position.
    pub fn witness(&self, position: Position) -> Result<Vec<H>, BridgeTreeError> {
        let current_frontier = self
            .frontier
            .as_ref()
            .ok_or(BridgeTreeError::PositionNotMarked(position))?;

        let saved_idx = self
            .lookup_prior_bridge(position)
            .map_err(|_| BridgeTreeError::PositionNotMarked(position))?;

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
            .map_err(BridgeTreeError::BridgeAddressInvalid)
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

#[cfg(test)]
#[cfg(DISABLED)]
mod tests {
    use proptest::prelude::*;
    use std::fmt::Debug;

    use super::*;
    use incrementalmerkletree::Hashable;
    use incrementalmerkletree_testing::{
        self as testing, apply_operation, arb_operation, check_checkpoint_rewind, check_operations,
        check_remove_mark, check_rewind_remove_mark, check_root_hashes, check_witnesses,
        complete_tree::CompleteTree, CombinedTree, SipHashable,
    };

    impl<H: Hashable + Clone + Ord, const DEPTH: u8> testing::Tree<H, usize>
        for BridgeTree<H, usize, DEPTH>
    {
        fn append(&mut self, value: H, retention: Retention<usize>) -> bool {
            let appended = BridgeTree::append(self, value);
            if appended {
                if retention.is_marked() {
                    BridgeTree::mark(self);
                }
                if let Retention::Checkpoint { id, .. } = retention {
                    BridgeTree::checkpoint(self, id);
                }
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
            BridgeTree::marked_positions(self)
        }

        fn root(&self, checkpoint_depth: usize) -> Option<H> {
            BridgeTree::root(self, checkpoint_depth)
        }

        fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>> {
            BridgeTree::witness(self, position, checkpoint_depth).ok()
        }

        fn remove_mark(&mut self, position: Position) -> bool {
            BridgeTree::remove_mark(self, position)
        }

        fn checkpoint(&mut self, id: usize) -> bool {
            BridgeTree::checkpoint(self, id)
        }

        fn rewind(&mut self) -> bool {
            BridgeTree::rewind(self)
        }
    }

    #[test]
    fn tree_depth() {
        let mut tree = BridgeTree::<String, usize, 3>::new(100);
        for c in 'a'..'i' {
            assert!(tree.append(c.to_string()))
        }
        assert!(!tree.append('i'.to_string()));
    }

    fn check_garbage_collect<H: Hashable + Clone + Ord + Debug, const DEPTH: u8>(
        mut tree: BridgeTree<H, usize, DEPTH>,
    ) {
        // Add checkpoints until we're sure everything that can be gc'ed will be gc'ed
        for i in 0..tree.max_checkpoints {
            tree.checkpoint(i + 1);
        }

        let mut tree_mut = tree.clone();
        tree_mut.garbage_collect();

        for pos in tree.saved.keys() {
            assert_eq!(tree.witness(*pos, 0), tree_mut.witness(*pos, 0));
        }
    }

    fn arb_bridgetree<G: Strategy + Clone>(
        item_gen: G,
        max_count: usize,
    ) -> impl Strategy<Value = BridgeTree<G::Value, usize, 8>>
    where
        G::Value: Hashable + Clone + Ord + Debug + 'static,
    {
        let pos_gen = (0..max_count).prop_map(|p| Position::try_from(p).unwrap());
        proptest::collection::vec(arb_operation(item_gen, pos_gen), 0..max_count).prop_map(|ops| {
            let mut tree: BridgeTree<G::Value, usize, 8> = BridgeTree::new(10);
            for (i, op) in ops.into_iter().enumerate() {
                apply_operation(&mut tree, op.map_checkpoint_id(|_| i));
            }
            tree
        })
    }

    proptest! {
        #[test]
        fn bridgetree_from_parts(
            tree in arb_bridgetree((97u8..123).prop_map(|c| char::from(c).to_string()), 100)
        ) {
            assert_eq!(
                BridgeTree::from_parts(
                    tree.prior_bridges.clone(),
                    tree.current_bridge.clone(),
                    tree.saved.clone(),
                    tree.checkpoints.clone(),
                    tree.max_checkpoints
                ),
                Ok(tree),
            );
        }

        #[test]
        fn prop_garbage_collect(
            tree in arb_bridgetree((97u8..123).prop_map(|c| char::from(c).to_string()), 100)
        ) {
            check_garbage_collect(tree);
        }
    }

    #[test]
    fn root_hashes() {
        check_root_hashes(BridgeTree::<String, usize, 4>::new);
    }

    #[test]
    fn witness() {
        check_witnesses(BridgeTree::<String, usize, 4>::new);
    }

    #[test]
    fn checkpoint_rewind() {
        check_checkpoint_rewind(|max_checkpoints| {
            BridgeTree::<String, usize, 4>::new(max_checkpoints)
        });
    }

    #[test]
    fn rewind_remove_mark() {
        check_rewind_remove_mark(|max_checkpoints| {
            BridgeTree::<String, usize, 4>::new(max_checkpoints)
        });
    }

    #[test]
    fn garbage_collect() {
        let mut tree: BridgeTree<String, usize, 7> = BridgeTree::new(1000);
        let empty_root = tree.root(0);
        tree.append("a".to_string());
        for i in 0..100 {
            tree.checkpoint(i + 1);
        }
        tree.garbage_collect();
        assert!(tree.root(0) != empty_root);
        tree.rewind();
        assert!(tree.root(0) != empty_root);

        let mut t = BridgeTree::<String, usize, 7>::new(10);
        let mut to_unmark = vec![];
        let mut has_witness = vec![];
        for i in 0u64..100 {
            let elem: String = format!("{},", i);
            assert!(t.append(elem), "Append should succeed.");
            if i % 5 == 0 {
                t.checkpoint(usize::try_from(i).unwrap() + 1);
            }
            if i % 7 == 0 {
                t.mark();
                if i > 0 && i % 2 == 0 {
                    to_unmark.push(Position::from(i));
                } else {
                    has_witness.push(Position::from(i));
                }
            }
            if i % 11 == 0 && !to_unmark.is_empty() {
                let pos = to_unmark.remove(0);
                t.remove_mark(pos);
            }
        }
        // 32 = 20 (checkpointed) + 14 (marked) - 2 (marked & checkpointed)
        assert_eq!(t.prior_bridges().len(), 20 + 14 - 2);
        let witness = has_witness
            .iter()
            .map(|pos| match t.witness(*pos, 0) {
                Ok(path) => path,
                Err(e) => panic!("Failed to get auth path: {:?}", e),
            })
            .collect::<Vec<_>>();
        t.garbage_collect();
        // 20 = 32 - 10 (removed checkpoints) + 1 (not removed due to mark) - 3 (removed marks)
        assert_eq!(t.prior_bridges().len(), 32 - 10 + 1 - 3);
        let retained_witness = has_witness
            .iter()
            .map(|pos| t.witness(*pos, 0).expect("Must be able to get auth path"))
            .collect::<Vec<_>>();
        assert_eq!(witness, retained_witness);
    }

    // Combined tree tests
    fn new_combined_tree<H: Hashable + Clone + Ord + Debug>(
        max_checkpoints: usize,
    ) -> CombinedTree<H, usize, CompleteTree<H, usize, 4>, BridgeTree<H, usize, 4>> {
        CombinedTree::new(
            CompleteTree::<H, usize, 4>::new(max_checkpoints),
            BridgeTree::<H, usize, 4>::new(max_checkpoints),
        )
    }

    #[test]
    fn combined_remove_mark() {
        check_remove_mark(new_combined_tree);
    }

    #[test]
    fn combined_rewind_remove_mark() {
        check_rewind_remove_mark(new_combined_tree);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]

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
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i + 1)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
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
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i + 1)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
        }
    }
}
