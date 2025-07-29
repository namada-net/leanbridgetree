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
pub use incrementalmerkletree::{
    frontier::{Frontier, NonEmptyFrontier},
    Address, Hashable, Level, Position, Retention, Source,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::ops::Range;

/// Errors that can be discovered during checks that verify the compatibility of adjacent bridges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContinuityError {
    /// Returned when a bridge with no prior position information is
    PriorPositionNotFound,
    /// Returned when the subsequent bridge's prior position does not match the position of the
    /// prior bridge's frontier.
    PositionMismatch(Position, Position),
}

/// Errors that can be discovered during the process of attempting to create
/// the witness for a leaf node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessingError {
    PositionNotMarked(Position),
    BridgeFusionError(ContinuityError),
    BridgeAddressInvalid(Address),
}

/// The information required to "update" witnesses from one state of a Merkle tree to another.
///
/// The witness for a particular leaf of a Merkle tree consists of the siblings of that leaf, plus
/// the siblings of the parents of that leaf in a path to the root of the tree. When considering a
/// Merkle tree where leaves are appended to the tree in a linear fashion (rather than being
/// inserted at arbitrary positions), we often wish to produce a witness for a leaf that was
/// appended to the tree at some point in the past. A [`MerkleBridge`] from one position in the
/// tree to another position in the tree contains the minimal amount of information necessary to
/// produce a witness for the leaf at the former position, given that leaves have been subsequently
/// appended to reach the current position.
///
/// [`MerkleBridge`] values have a semigroup, such that the sum (`fuse`d) value of two successive
/// bridges, along with a [`NonEmptyFrontier`] with its tip at the prior position of the first bridge
/// being fused, can be used to produce a witness for the leaf at the tip of the prior frontier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleBridge<H> {
    /// The position of the final leaf in the frontier of the bridge that this bridge is the
    /// successor of, or None if this is the first bridge in a tree.
    prior_position: Option<Position>,
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
    /// The leading edge of the bridge.
    frontier: NonEmptyFrontier<H>,
}

impl<H> MerkleBridge<H> {
    /// Construct a new Merkle bridge containing only the specified
    /// leaf.
    pub fn new(value: H) -> Self {
        Self {
            prior_position: None,
            tracking: BTreeSet::new(),
            ommers: BTreeMap::new(),
            frontier: NonEmptyFrontier::new(value),
        }
    }

    /// Construct a new Merkle bridge from its constituent parts.
    pub fn from_parts(
        prior_position: Option<Position>,
        tracking: BTreeSet<Address>,
        ommers: BTreeMap<Address, H>,
        frontier: NonEmptyFrontier<H>,
    ) -> Self {
        Self {
            prior_position,
            tracking,
            ommers,
            frontier,
        }
    }

    /// Returns the position of the final leaf in the frontier of the
    /// bridge that this bridge is the successor of, or None
    /// if this is the first bridge in a tree.
    pub fn prior_position(&self) -> Option<Position> {
        self.prior_position
    }

    /// Returns the position of the most recently appended leaf.
    pub fn position(&self) -> Position {
        self.frontier.position()
    }

    /// Returns the set of internal node addresses that we're searching
    /// for the ommers for.
    pub fn tracking(&self) -> &BTreeSet<Address> {
        &self.tracking
    }

    /// Returns the set of internal node addresses that we're searching
    /// for the ommers for.
    pub fn ommers(&self) -> &BTreeMap<Address, H> {
        &self.ommers
    }

    /// Returns the non-empty frontier of this Merkle bridge.
    pub fn frontier(&self) -> &NonEmptyFrontier<H> {
        &self.frontier
    }

    /// Returns the value of the most recently appended leaf.
    pub fn current_leaf(&self) -> &H {
        self.frontier.leaf()
    }

    /// Checks whether this bridge is a valid successor for the specified
    /// bridge.
    pub fn check_continuity(&self, next: &Self) -> Result<(), ContinuityError> {
        if let Some(pos) = next.prior_position {
            if pos == self.frontier.position() {
                Ok(())
            } else {
                Err(ContinuityError::PositionMismatch(
                    self.frontier.position(),
                    pos,
                ))
            }
        } else {
            Err(ContinuityError::PriorPositionNotFound)
        }
    }

    /// Returns the range of positions observed by this bridge.
    pub fn position_range(&self) -> Range<Position> {
        Range {
            start: self.prior_position.unwrap_or_else(|| Position::from(0)),
            end: self.position() + 1,
        }
    }
}

impl<'a, H: Hashable + Clone + Ord + 'a> MerkleBridge<H> {
    /// Constructs a new bridge to follow this one. If `mark_current_leaf` is true, the successor
    /// will track the information necessary to create a witness for the leaf most
    /// recently appended to this bridge's frontier.
    #[must_use]
    pub fn successor(&self, track_current_leaf: bool) -> Self {
        let mut result = Self {
            prior_position: Some(self.frontier.position()),
            tracking: self.tracking.clone(),
            ommers: BTreeMap::new(),
            frontier: self.frontier.clone(),
        };

        if track_current_leaf {
            result.track_current_leaf();
        }

        result
    }

    fn track_current_leaf(&mut self) {
        self.tracking
            .insert(Address::from(self.frontier.position()).current_incomplete());
    }

    /// Advances this bridge's frontier by appending the specified node,
    /// and updates any auth path ommers being tracked if necessary.
    pub fn append(&mut self, value: H) {
        self.frontier.append(value);

        let mut found = vec![];
        for address in self.tracking.iter() {
            // We know that there will only ever be one address that we're
            // tracking at a given level, because as soon as we find a
            // value for the sibling of the address we're tracking, we
            // remove the tracked address and replace it the next parent
            // of that address for which we need to find a sibling.
            if self
                .frontier()
                .position()
                .is_complete_subtree(address.level())
            {
                let digest = self.frontier.root(Some(address.level()));
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
    }

    /// Returns a single MerkleBridge that contains the aggregate information
    /// of this bridge and `next`, or None if `next` is not a valid successor
    /// to this bridge. The resulting Bridge will have the same state as though
    /// `self` had had every leaf used to construct `next` appended to it.
    /// directly.
    pub fn fuse(&self, next: &Self) -> Result<Self, ContinuityError> {
        self.check_continuity(next)?;

        Ok(Self {
            prior_position: self.prior_position,
            tracking: next.tracking.clone(),
            ommers: self
                .ommers
                .iter()
                .chain(next.ommers.iter())
                .map(|(k, v)| (*k, v.clone()))
                .collect(),
            frontier: next.frontier.clone(),
        })
    }

    /// Returns a single MerkleBridge that contains the aggregate information
    /// of all the provided bridges (discarding internal frontiers) or None
    /// if the provided iterator is empty. Returns a continuity error if
    /// any of the bridges are not valid successors to one another.
    fn fuse_all<T: Iterator<Item = &'a Self>>(
        mut iter: T,
    ) -> Result<Option<Self>, ContinuityError> {
        let mut fused = iter.next().cloned();
        for next in iter {
            fused = Some(fused.unwrap().fuse(next)?);
        }
        Ok(fused)
    }

    /// If this bridge contains sufficient auth fragment information, construct an authentication
    /// path for the specified position by interleaving with values from the prior frontier. This
    /// method will panic if the position of the prior frontier does not match this bridge's prior
    /// position.
    fn witness(
        &self,
        depth: u8,
        prior_frontier: &NonEmptyFrontier<H>,
    ) -> Result<Vec<H>, WitnessingError> {
        assert!(Some(prior_frontier.position()) == self.prior_position);

        prior_frontier
            .witness(depth, |addr| {
                let r = addr.position_range();
                if self.frontier.position() < r.start {
                    Some(H::empty_root(addr.level()))
                } else if r.contains(&self.frontier.position()) {
                    Some(self.frontier.root(Some(addr.level())))
                } else {
                    // the frontier's position is after the end of the requested
                    // range, so the requested value should exist in a stored
                    // fragment
                    self.ommers.get(&addr).cloned()
                }
            })
            .map_err(WitnessingError::BridgeAddressInvalid)
    }

    /*
    fn retain(&mut self, ommer_addrs: &BTreeSet<Address>) {
        // Prune away any ommers & tracking addresses we don't need
        self.tracking
            .retain(|addr| ommer_addrs.contains(&addr.sibling()));
        self.ommers.retain(|addr, _| ommer_addrs.contains(addr));
    }
    */
}

/// A sparse representation of a Merkle tree with linear appending of leaves that contains enough
/// information to produce a witness for any `mark`ed leaf.
#[derive(Default, Clone, PartialEq, Eq)]
pub struct BridgeTree<H, const DEPTH: u8> {
    /// The ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf.
    prior_bridges: Vec<MerkleBridge<H>>,
    /// The current (mutable) bridge at the tip of the tree.
    current_bridge: Option<MerkleBridge<H>>,
    /// A map from positions for which we wish to be able to compute a
    /// witness to index in the bridges vector.
    saved: BTreeMap<Position, usize>,
}

impl<H: Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let Self {
            prior_bridges,
            current_bridge,
            saved,
        } = self;

        f.debug_struct(stringify!(BridgeTree))
            .field("depth", &DEPTH)
            .field("prior_bridges", prior_bridges)
            .field("current_bridge", current_bridge)
            .field("saved", saved)
            .finish()
    }
}

/// Errors that can appear when validating the internal consistency of a `[BridgeTree]`
/// value when constructing a tree from its constituent parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeTreeError {
    IncorrectIncompleteIndex,
    InvalidMarkIndex(usize),
    PositionMismatch { expected: Position, found: Position },
    InvalidSavePoints,
    Discontinuity(ContinuityError),
}

impl<H, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct an empty BridgeTree value.
    pub const fn new() -> Self {
        Self {
            prior_bridges: Vec::new(),
            current_bridge: None,
            saved: BTreeMap::new(),
        }
    }

    /// Returns the prior bridges that make up this tree
    pub fn prior_bridges(&self) -> &[MerkleBridge<H>] {
        &self.prior_bridges
    }

    /// Returns the current bridge at the tip of this tree
    pub fn current_bridge(&self) -> &Option<MerkleBridge<H>> {
        &self.current_bridge
    }

    /// Returns the map from leaf positions that have been marked to the index of
    /// the bridge whose tip is at that position in this tree's list of bridges.
    pub fn marked_indices(&self) -> &BTreeMap<Position, usize> {
        &self.saved
    }

    /// Returns the bridge's frontier.
    pub fn frontier(&self) -> Option<&NonEmptyFrontier<H>> {
        self.current_bridge.as_ref().map(|b| b.frontier())
    }
}

impl<H: Hashable + Clone + Ord, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct a new BridgeTree that will start recording changes from the state of
    /// the specified frontier.
    pub fn from_frontier(frontier: NonEmptyFrontier<H>) -> Self {
        Self {
            prior_bridges: vec![],
            current_bridge: Some(MerkleBridge::from_parts(
                None,
                BTreeSet::new(),
                BTreeMap::new(),
                frontier,
            )),
            saved: BTreeMap::new(),
        }
    }

    /// Construct a new BridgeTree from its constituent parts, checking for internal
    /// consistency.
    pub fn from_parts(
        prior_bridges: Vec<MerkleBridge<H>>,
        current_bridge: Option<MerkleBridge<H>>,
        saved: BTreeMap<Position, usize>,
    ) -> Result<Self, BridgeTreeError> {
        Self::check_consistency_internal(&prior_bridges, &current_bridge, &saved)?;
        Ok(Self {
            prior_bridges,
            current_bridge,
            saved,
        })
    }

    /*
    fn check_consistency(&self) -> Result<(), BridgeTreeError> {
        Self::check_consistency_internal(&self.prior_bridges, &self.current_bridge, &self.saved)
    }
    */

    fn check_consistency_internal(
        prior_bridges: &[MerkleBridge<H>],
        current_bridge: &Option<MerkleBridge<H>>,
        saved: &BTreeMap<Position, usize>,
    ) -> Result<(), BridgeTreeError> {
        // check that saved values correspond to bridges
        for (pos, i) in saved {
            if i >= &prior_bridges.len() {
                return Err(BridgeTreeError::InvalidMarkIndex(*i));
            }
            let found = prior_bridges[*i].position();
            if &found != pos {
                return Err(BridgeTreeError::PositionMismatch {
                    expected: *pos,
                    found,
                });
            }
        }

        for (prev, next) in prior_bridges.iter().zip(prior_bridges.iter().skip(1)) {
            prev.check_continuity(next)
                .map_err(BridgeTreeError::Discontinuity)?;
        }

        if let Some((prev, next)) = prior_bridges.last().zip(current_bridge.as_ref()) {
            prev.check_continuity(next)
                .map_err(BridgeTreeError::Discontinuity)?;
        }

        Ok(())
    }

    /// Appends a new value to the tree at the next available slot.
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed depth.
    pub fn append(&mut self, value: H) -> bool {
        if let Some(bridge) = self.current_bridge.as_mut() {
            if bridge
                .frontier
                .position()
                .is_complete_subtree(Level::from(DEPTH))
            {
                false
            } else {
                bridge.append(value);
                true
            }
        } else {
            self.current_bridge = Some(MerkleBridge::new(value));
            true
        }
    }

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    pub fn root(&self) -> H {
        let root_level = Level::from(DEPTH);

        self.current_bridge.as_ref().map_or_else(
            || H::empty_root(root_level),
            |bridge| bridge.frontier().root(Some(root_level)),
        )
    }

    /// Returns the most recently appended leaf value.
    pub fn current_position(&self) -> Option<Position> {
        self.current_bridge.as_ref().map(|b| b.position())
    }

    /// Returns the most recently appended leaf value.
    pub fn current_leaf(&self) -> Option<&H> {
        self.current_bridge.as_ref().map(|b| b.current_leaf())
    }

    /// Marks the current leaf as one for which we're interested in producing a witness.
    ///
    /// Returns an optional value containing the current position if successful or if the current
    /// value was already marked, or None if the tree is empty.
    pub fn mark(&mut self) -> Option<Position> {
        match self.current_bridge.take() {
            Some(cur_b) => {
                let pos = cur_b.position();

                // the successor(true) call will ensure that the marked leaf is tracked
                let successor = cur_b.successor(true);
                self.prior_bridges.push(cur_b);
                self.current_bridge = Some(successor);

                // save the position of the marked leaf
                if let std::collections::btree_map::Entry::Vacant(e) = self.saved.entry(pos) {
                    e.insert(self.prior_bridges.len() - 1);
                }

                Some(pos)
            }
            None => None,
        }
    }

    /// Return a set of all the positions for which we have marked.
    pub fn marked_positions(&self) -> BTreeSet<Position> {
        self.saved.keys().cloned().collect()
    }

    /// Returns the leaf at the specified position if the tree can produce
    /// a witness for it.
    pub fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        self.saved
            .get(&position)
            .and_then(|idx| self.prior_bridges.get(*idx).map(|b| b.current_leaf()))
    }

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for. Returns true if successful and
    /// false if we were already not maintaining a mark at this position.
    pub fn remove_mark(&mut self, position: Position) -> bool {
        if self.saved.contains_key(&position) {
            self.saved.remove(&position);
            true
        } else {
            false
        }
    }

    /// Obtains a witness for the value at the specified leaf position.
    /// Returns an error if there is no witness information for the requested
    /// position.
    pub fn witness(&self, position: Position) -> Result<Vec<H>, WitnessingError> {
        let saved_idx = self
            .saved
            .get(&position)
            .ok_or(WitnessingError::PositionNotMarked(position))?;

        let prior_frontier = &self.prior_bridges[*saved_idx].frontier;

        // Fuse the following bridges to obtain a bridge that has all
        // of the data to the right of the selected value in the tree.
        let fuse_from = saved_idx + 1;
        let successor = MerkleBridge::fuse_all(
            // fuse all the way up to the current tip
            self.prior_bridges[fuse_from..]
                .iter()
                .chain(&self.current_bridge),
        )
        .map(|fused| fused.unwrap()) // safe as the iterator being fused is nonempty
        .map_err(WitnessingError::BridgeFusionError)?;

        successor.witness(DEPTH, prior_frontier)
    }

    /// Remove state from the tree that no longer needs to be maintained
    /// because it is associated with marks that have been removed from
    /// the tree.
    pub fn garbage_collect(&mut self) {
        /*
        let gc_len = self.checkpoints.front().unwrap().bridges_len;
        let mut cur: Option<MerkleBridge<H>> = None;
        let mut merged = 0;
        let mut ommer_addrs: BTreeSet<Address> = BTreeSet::new();

        for (i, next_bridge) in std::mem::take(&mut self.prior_bridges)
            .into_iter()
            .enumerate()
        {
            if let Some(cur_bridge) = cur {
                let pos = cur_bridge.position();
                let mut new_cur = if self.saved.contains_key(&pos) || i > gc_len {
                    // We need to remember cur_bridge; update its save index & put next_bridge
                    // on the chopping block
                    if let Some(idx) = self.saved.get_mut(&pos) {
                        *idx -= merged;
                    }

                    // Add the elements of the auth path to the set of addresses we should
                    // continue to track and retain information for
                    for (addr, source) in cur_bridge
                        .frontier
                        .position()
                        .witness_addrs(Level::from(DEPTH))
                    {
                        if source == Source::Future {
                            ommer_addrs.insert(addr);
                        }
                    }

                    self.prior_bridges.push(cur_bridge);
                    next_bridge
                } else {
                    // We can fuse these bridges together because we don't need to
                    // remember next_bridge.
                    merged += 1;
                    cur_bridge.fuse(&next_bridge).unwrap()
                };

                new_cur.retain(&ommer_addrs);
                cur = Some(new_cur);
            } else {
                // this case will only occur for the first bridge
                cur = Some(next_bridge);
            }
        }

        // unwrap is safe because we know that prior_bridges was nonempty.
        if let Some(last_bridge) = cur {
            if let Some(idx) = self.saved.get_mut(&last_bridge.position()) {
                *idx -= merged;
            }
            self.prior_bridges.push(last_bridge);
        }

        for c in self.checkpoints.iter_mut() {
            c.rewrite_indices(|idx| idx - merged);
        }
        if let Err(e) = self.check_consistency() {
            panic!(
                "Consistency check failed after garbage collection with {:?}",
                e
            );
        }
        */
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
