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

use slab::Slab;

pub use incrementalmerkletree::{
    Address, Hashable, Level, Position, Source, frontier::NonEmptyFrontier,
};

#[cfg(feature = "display-error")]
use thiserror::Error;

#[cfg(feature = "std")]
use ::std::marker::Send as MaybeSend;

#[cfg(not(feature = "std"))]
#[doc(hidden)]
pub trait MaybeSend {}

#[cfg(not(feature = "std"))]
impl<T> MaybeSend for T {}

/// Abstract tree frontier.
pub trait AbstractFrontier<H> {
    /// The leaf at the frontier.
    fn leaf(&self) -> &H;

    /// The set of ommers of this frontier.
    fn ommers(&self) -> &[H];

    /// Get a [`NonEmptyFrontier`] from this [`AbstractFrontier`].
    ///
    /// Panics if the number of ommers is incorrect, or the position
    /// is invalid.
    fn to_non_empty_frontier(frontiers: &[Self], position: Position) -> NonEmptyFrontier<H>
    where
        Self: Sized,
        H: Clone,
    {
        let frontier_index: usize = position.try_into().unwrap();
        to_non_empty_frontier(position, &frontiers[frontier_index])
    }
}

fn to_non_empty_frontier<F, H>(position: Position, frontier: &F) -> NonEmptyFrontier<H>
where
    H: Clone,
    F: AbstractFrontier<H>,
{
    NonEmptyFrontier::from_parts(
        position,
        frontier.leaf().clone(),
        frontier.ommers().to_vec(),
    )
    .unwrap()
}

impl<H> AbstractFrontier<H> for NonEmptyFrontier<H> {
    fn leaf(&self) -> &H {
        Self::leaf(self)
    }

    fn ommers(&self) -> &[H] {
        Self::ommers(self)
    }
}

impl<H> AbstractFrontier<H> for (H, Vec<H>) {
    fn leaf(&self) -> &H {
        &self.0
    }

    fn ommers(&self) -> &[H] {
        &self.1
    }
}

/// Struct used to debug the true (i.e. unmodified) inner
/// representation of a [`BridgeTree`].
pub struct DebugBridgeTree<'tree, H, const DEPTH: u8> {
    tree: &'tree BridgeTree<H, DEPTH>,
}

impl<H: Debug, const DEPTH: u8> Debug for DebugBridgeTree<'_, H, DEPTH> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        // NB: We destructure `self` in order to catch compiler errors, should
        // its internal structure ever change. See note below.
        let Self {
            tree:
                BridgeTree {
                    frontier,
                    tracking,
                    ommers,
                    prior_bridges_slab,
                    prior_bridges_slab_keys,
                },
        } = self;

        // XXX: Keep me up to date!
        f.debug_struct(stringify!(DebugBridgeTree))
            .field("frontier", frontier)
            .field("tracking", tracking)
            .field("ommers", ommers)
            .field("prior_bridges_slab", prior_bridges_slab)
            .field("prior_bridges_slab_keys", prior_bridges_slab_keys)
            .finish()
    }
}

/// Sparse representation of a Merkle tree with linear appending of leaves that contains enough
/// information to produce a witness for any [marked](BridgeTree::mark) leaf.
#[derive(Default, Clone)]
pub struct BridgeTree<H, const DEPTH: u8> {
    /// The current (mutable) frontier of the tree.
    frontier: Option<NonEmptyFrontier<H>>,
    /// Storage of Merkle bridges.
    prior_bridges_slab: Slab<NonEmptyFrontier<H>>,
    /// List of keys into `prior_bridges_slab`, representing the history
    /// of the tree.
    ///
    /// Keys are ordered by the bridges' frontier positions. There will
    /// be one bridge for each saved leaf.
    prior_bridges_slab_keys: Vec<usize>,
    /// Set of addresses for which we are waiting to discover the ommers.
    ///
    /// The values of this set and the keys of the `ommers` map should always
    /// be disjoint. Also, this set should never contain an address for which
    /// the sibling value has been discovered; at that point, the address is
    /// replaced in this set with its parent and the address/sibling pair
    /// is stored in `ommers`.
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

impl<const DEPTH: u8, H: Eq> Eq for BridgeTree<H, DEPTH> {}

impl<const DEPTH: u8, H: PartialEq> PartialEq for BridgeTree<H, DEPTH> {
    fn eq(&self, other: &Self) -> bool {
        // NB: We destructure `self` in order to catch compiler errors, should
        // its internal structure ever change. See note below.
        let Self {
            frontier,
            tracking,
            ommers,

            // NB: Marked as unused, because they get implicitly
            // used by calling `BridgeTree::prior_bridges`.
            prior_bridges_slab: _,
            prior_bridges_slab_keys: _,
        } = self;

        frontier.eq(&other.frontier)
            && tracking.eq(&other.tracking)
            && ommers.eq(&other.ommers)
            && self.prior_bridges().eq(other.prior_bridges())
    }
}

impl<H: Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        // NB: We destructure `self` in order to catch compiler errors, should
        // its internal structure ever change. See note below.
        let Self {
            frontier,
            tracking,
            ommers,

            // NB: Marked as unused, because they get implicitly
            // used by calling `BridgeTree::prior_bridges`.
            prior_bridges_slab: _,
            prior_bridges_slab_keys: _,
        } = self;

        // Use a friendlier display format for prior bridges.
        let prior_bridges: Vec<_> = self.prior_bridges().collect();

        // XXX: Keep me up to date!
        f.debug_struct(stringify!(BridgeTree))
            .field("max_depth", &DEPTH) // virtual field
            .field("frontier", frontier)
            .field("prior_bridges", &prior_bridges)
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
    /// The tree is missing the data of various ommers.
    #[cfg_attr(
        feature = "display-error",
        error("Missing ommer data, expected {expected} but got {got}")
    )]
    MissingOmmers { expected: u8, got: u8 },
    /// The trees have different frontiers, thus cannot be merged.
    #[cfg_attr(
        feature = "display-error",
        error("Attempted to merge two trees with different frontiers")
    )]
    MergeDifferentFrontier,
    /// Marked leaf position is outside the range of the latest frontier.
    #[cfg_attr(
        feature = "display-error",
        error("Leaf {position:?} greater than leaf frontier {frontier:?}")
    )]
    PositionOutsideFrontierRange {
        frontier: Position,
        position: Position,
    },
}

impl<H, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Construct an empty [`BridgeTree`].
    pub const fn new() -> Self {
        Self {
            prior_bridges_slab: Slab::new(),
            prior_bridges_slab_keys: Vec::new(),
            tracking: BTreeSet::new(),
            ommers: BTreeMap::new(),
            frontier: None,
        }
    }

    /// Get a debug representation of a [`BridgeTree`] with no
    /// sugaring of the inner fields.
    #[inline]
    pub fn unsugared_debug(&self) -> DebugBridgeTree<'_, H, DEPTH> {
        DebugBridgeTree { tree: self }
    }

    /// Return the prior bridges that make up this tree
    pub fn prior_bridges(&self) -> impl Iterator<Item = &NonEmptyFrontier<H>> + '_ {
        self.prior_bridges_slab_keys
            .iter()
            .map(|&key| unsafe { self.get_prior_bridge_by_slab_key_unchecked(key) })
    }

    /// Return a prior Merkle bridge, given its slab key.
    ///
    /// ## Safety
    ///
    /// This method doesn't check that the provided `key` exists in the slab of Merkle bridges.
    /// However, given that the slab is never exposed to users, its underlying structure is fully
    /// under the control for this crate. Therefore, we never build an invalid storage of
    /// Merkle bridges. In any case, we should mark this method as `unsafe`.
    #[inline]
    unsafe fn get_prior_bridge_by_slab_key_unchecked(&self, key: usize) -> &NonEmptyFrontier<H> {
        unsafe { get_prior_bridge_by_slab_key_unchecked(&self.prior_bridges_slab, key) }
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

    /// Check whether the tree is empty (i.e. has no leaves).
    pub fn is_empty(&self) -> bool {
        self.frontier.is_none()
    }

    /// Construct a new [`BridgeTree`] that will start recording changes from the state of
    /// the specified frontier.
    pub fn from_frontier(frontier: NonEmptyFrontier<H>) -> Result<Self, BridgeTreeError> {
        if frontier.position().is_complete_subtree(Level::from(DEPTH)) {
            return Err(BridgeTreeError::FullTree);
        }

        let mut tree = Self::new();
        tree.frontier = Some(frontier);

        Ok(tree)
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

        // Build the Merkle tree slab.
        let prior_bridges_slab_keys: Vec<_> = (0..prior_bridges.len()).collect();
        let prior_bridges_slab = {
            let mut slab = Slab::new();

            for bridge_frontier in prior_bridges {
                slab.insert(bridge_frontier);
            }

            slab
        };

        Ok(Self {
            frontier,
            prior_bridges_slab,
            prior_bridges_slab_keys,
            tracking,
            ommers,
        })
    }

    /// Return an iterator over all the marked leaf positions.
    pub fn marked_positions(&self) -> impl Iterator<Item = Position> + '_ {
        self.prior_bridges()
            .map(|bridge_frontier| bridge_frontier.position())
    }

    /// Returns the leaf at the specified position if the tree can produce
    /// a witness for it.
    pub fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        self.lookup_prior_bridge(position)
            .map(|bridge_frontier| bridge_frontier.leaf())
    }

    /// Clone the tree's frontier at a position before or equal to the
    /// provided `position`.
    ///
    /// The returned tree does not track any leaves.
    pub fn clone_from_frontier_at(&self, position: Position) -> Self
    where
        H: Clone,
    {
        let frontier = match self.lookup_prior_bridge_slab_index(position) {
            Ok(frontier_index) => {
                let key = unsafe { *self.prior_bridges_slab_keys.get_unchecked(frontier_index) };
                unsafe { self.get_prior_bridge_by_slab_key_unchecked(key) }
            }
            Err(0) => return Self::new(),
            Err(index_to_insert) => {
                let index = index_to_insert - 1;
                let key = unsafe { *self.prior_bridges_slab_keys.get_unchecked(index) };
                unsafe { self.get_prior_bridge_by_slab_key_unchecked(key) }
            }
        };

        Self {
            frontier: Some(frontier.clone()),
            ..Self::new()
        }
    }

    /// Merge the data in this tree with the data in another tree.
    ///
    /// This results in an error if both trees have different frontiers.
    pub fn merge_with(&mut self, mut other: Self) -> Result<(), BridgeTreeError>
    where
        H: PartialEq,
    {
        if self.frontier != other.frontier {
            return Err(BridgeTreeError::MergeDifferentFrontier);
        }

        // Move the tracking data from `other` into `self`.
        //
        // We should not be missing any intermediate nodes
        // required to build merkle proofs of existence of
        // prior leaves, since the merged trees have the
        // same frontier.
        self.ommers.append(&mut other.ommers);
        self.tracking.append(&mut other.tracking);

        // Move all the bridges from `other` into `self`,
        // unless they were already present.
        let Self {
            prior_bridges_slab_keys,
            mut prior_bridges_slab,
            ..
        } = other;

        let drain_other_prior_bridges = prior_bridges_slab_keys
            .into_iter()
            .map(move |key| prior_bridges_slab.remove(key));

        for bridge_frontier in drain_other_prior_bridges {
            let Err(index_of_marked_leaf_bridge) =
                self.lookup_prior_bridge_slab_index(bridge_frontier.position())
            else {
                unlikely(|| {});

                // Skip bridges already in `self`.
                continue;
            };

            let key = self.prior_bridges_slab.insert(bridge_frontier);
            self.prior_bridges_slab_keys
                .insert(index_of_marked_leaf_bridge, key);
        }

        Ok(())
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

    /// Return the last prior bridge.
    fn last_prior_bridge(&self) -> Option<&NonEmptyFrontier<H>> {
        self.prior_bridges_slab_keys
            .last()
            .map(|&key| unsafe { self.get_prior_bridge_by_slab_key_unchecked(key) })
    }

    /// Look-up a prior Merkle bridge, given the position of a marked leaf.
    #[inline]
    fn lookup_prior_bridge(&self, position: Position) -> Option<&NonEmptyFrontier<H>> {
        // NB: This is the index into the vector of slab keys.
        let index = self.lookup_prior_bridge_slab_index(position).ok()?;

        // NB: This is the key into the slab.
        let key = {
            #[cfg(debug_assertions)]
            {
                self.prior_bridges_slab_keys[index]
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                *self.prior_bridges_slab_keys.get_unchecked(index)
            }
        };

        // Don't mix up these values, which are both `usize`! It's important
        // to understand the difference between them, otherwise none of these
        // unsafes will work, and we will encounter UB.
        Some(unsafe { self.get_prior_bridge_by_slab_key_unchecked(key) })
    }

    /// Look-up a prior bridge's key in `prior_bridges_slab_keys`, given the position of a marked leaf.
    ///
    /// Returns the index of the bridge key if `Ok`, or where a new bridge
    /// could be inserted, if `Err`.
    #[inline]
    fn lookup_prior_bridge_slab_index(&self, position: Position) -> Result<usize, usize> {
        lookup_prior_bridge_slab_index(
            &self.prior_bridges_slab,
            &self.prior_bridges_slab_keys,
            position,
        )
    }
}

impl<H: Hashable + Clone + MaybeSend, const DEPTH: u8> BridgeTree<H, DEPTH> {
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
            debug_assert!(!self.ommers.contains_key(&parent));
            if parent.level() < DEPTH.into() {
                self.tracking.insert(parent);
            }
        }

        Ok(())
    }

    /// Update this Merkle tree with the provided frontiers.
    ///
    /// Each position in `frontiers` represents a position of a leaf
    /// in the Merkle tree, with the first index matching the first
    /// leaf in the tree, and so on. New markings will be added for
    /// each position in `new_marked_leaf_positions`.
    ///
    /// ## Safety
    ///
    /// This method does not validate the history of the provided frontiers.
    /// It assumes the frontiers belong to the same Merkle tree as the one
    /// in `self`.
    pub unsafe fn update<F>(
        &mut self,
        frontiers: &[F],
        new_marked_leaf_positions: &[Position],
    ) -> Result<(), BridgeTreeError>
    where
        F: AbstractFrontier<H>,
    {
        // Validate the state changes before we apply them.
        let latest_position = if !frontiers.is_empty() {
            Position::from((frontiers.len() - 1) as u64)
        } else {
            // If no frontier is provided, we don't have anything to update...
            return Ok(());
        };
        if Some(latest_position) <= self.current_position() {
            // Nothing to do if the provided data precedes our
            // frontier.
            return Ok(());
        }
        for position in new_marked_leaf_positions.iter().copied() {
            if position > latest_position {
                return Err(BridgeTreeError::PositionOutsideFrontierRange {
                    frontier: latest_position,
                    position,
                });
            }
        }
        for (position, frontier) in frontiers.iter().enumerate() {
            let position = Position::from(position as u64);

            let ommers = frontier.ommers().len() as u8;
            let expected = position.past_ommer_count();

            if ommers != expected {
                return Err(BridgeTreeError::MissingOmmers {
                    got: ommers,
                    expected,
                });
            }
        }

        // Update the frontier of the tree with the latest frontier.
        self.frontier = Some(F::to_non_empty_frontier(frontiers, latest_position));

        // Add all the new leaves we wish to track.
        for position in new_marked_leaf_positions.iter().copied() {
            let Err(index_of_marked_leaf_bridge) = self.lookup_prior_bridge_slab_index(position)
            else {
                unlikely(|| {});

                // Skip bridges already in `self`.
                continue;
            };

            self.tracking
                .insert(Address::from(position).current_incomplete());

            let frontier = F::to_non_empty_frontier(frontiers, position);

            let key = self.prior_bridges_slab.insert(frontier);
            self.prior_bridges_slab_keys
                .insert(index_of_marked_leaf_bridge, key);
        }

        // Find all the missing ommers within the provided data.
        let mut found = Vec::new();
        loop {
            for address in self.tracking.iter() {
                if let Some(digest) = find_precomputed_ommer(address, frontiers)
                    .cloned()
                    .or_else(|| recompute_hash_at_address(&address.sibling(), frontiers))
                {
                    self.ommers.insert(address.sibling(), digest);
                    found.push(*address);
                }
            }

            if found.is_empty() {
                break;
            }

            for address in found.drain(..) {
                self.tracking.remove(&address);
                let parent = address.next_incomplete_parent();
                debug_assert!(!self.ommers.contains_key(&parent));
                if parent.level() < DEPTH.into() {
                    self.tracking.insert(parent);
                }
            }
        }

        Ok(())
    }

    /// Obtain the root of the Merkle tree at the maximum depth.
    #[inline]
    pub fn root(&self) -> H {
        self.root_at_level(DEPTH.into())
    }

    /// Obtain the root of the Merkle tree at the specified level.
    pub fn root_at_level(&self, level: Level) -> H {
        self.frontier.as_ref().map_or_else(
            || unlikely(|| H::empty_root(level)),
            |frontier| frontier.root(Some(level)),
        )
    }

    /// Marks the current leaf as one for which we're interested in producing a witness.
    ///
    /// Returns an optional value containing the current position if successful or if the current
    /// value was already marked, or None if the tree is empty.
    pub fn mark(&mut self) -> Option<Position> {
        let frontier = self.frontier.as_ref()?;

        let last_leaf_already_marked = self
            .last_prior_bridge()
            .is_some_and(|bridge_frontier| bridge_frontier.position() == frontier.position());

        if !last_leaf_already_marked {
            let key = self.prior_bridges_slab.insert(frontier.clone());
            self.prior_bridges_slab_keys.push(key);
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
    ///
    /// ## Warning
    ///
    /// This method does not remove the tracking data associated with
    /// `position`. Use [`BridgeTree::remove_mark_and_gc`] if you want
    /// to automate this process.
    pub fn remove_mark(&mut self, position: Position) -> Result<(), BridgeTreeError> {
        // Figure out where the marked leaf is in the vector of Merkle bridge slab indices.
        let index_of_marked_leaf_bridge = self
            .lookup_prior_bridge_slab_index(position)
            .map_err(|_| unlikely(|| BridgeTreeError::PositionNotMarked(position)))?;

        // Remove the index from the vector. This shifts all bridge indices following
        // `index_of_marked_leaf_bridge` in the vector in O(n) time, and returns the
        // key of the removed bridge.
        let key = self
            .prior_bridges_slab_keys
            .remove(index_of_marked_leaf_bridge);

        // Remove the actual bridge. This completes in O(1) time.
        self.prior_bridges_slab.remove(key);

        Ok(())
    }

    /// Remove multiple marked leaves in a batch.
    ///
    /// Unlike [`BridgeTree::remove_mark`], this is optimized for batch
    /// processing of many marks to be removed.
    ///
    /// ## Warning
    ///
    /// This method does not remove the tracking data associated with
    /// `positions`. Use [`BridgeTree::garbage_collect_ommers`] to get rid of
    /// this data.
    pub fn remove_multiple_marks<I>(&mut self, positions: I) -> Result<(), BridgeTreeError>
    where
        I: IntoIterator<Item = Position>,
    {
        // Set of leaves which we wish to remove. Well, more like
        // the keys to those leaves' bridges, in the slab.
        let keys_of_marked_leaf_bridges = positions
            .into_iter()
            .map(|position| {
                let index = self
                    .lookup_prior_bridge_slab_index(position)
                    .map_err(|_| unlikely(|| BridgeTreeError::PositionNotMarked(position)))?;

                Ok(unsafe { *self.prior_bridges_slab_keys.get_unchecked(index) })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;

        // Remove the keys from the vector of slab keys.
        self.prior_bridges_slab_keys.retain(|key| {
            let leaf_in_remove_set = keys_of_marked_leaf_bridges.contains(key);

            // NB: We wish to retain the leaves **not** in the set.
            !leaf_in_remove_set
        });

        // Remove the bridges from the slab.
        for key in keys_of_marked_leaf_bridges {
            self.prior_bridges_slab.remove(key);
        }

        Ok(())
    }

    /// Convenience method for calling [`BridgeTree::remove_mark`] and
    /// [`BridgeTree::garbage_collect_ommers`]
    /// in sequence.
    ///
    /// ## Performance
    ///
    /// In general, it is preferred to call [`BridgeTree::remove_multiple_marks`], then have
    /// a single call to [`BridgeTree::garbage_collect_ommers`], as this will be **significantly**
    /// more efficient.
    pub fn remove_mark_and_gc(&mut self, position: Position) -> Result<(), BridgeTreeError> {
        self.remove_mark(position)?;
        self.garbage_collect_ommers();
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

        let prior_frontier = self
            .lookup_prior_bridge(position)
            .ok_or_else(|| unlikely(|| BridgeTreeError::PositionNotMarked(position)))?;

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

#[cfg(feature = "std")]
impl<H: Hashable + Clone + MaybeSend, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Remove data that is not necessary for the currently tracked leaves.
    pub fn garbage_collect_ommers(&mut self) {
        use std::thread;

        // TODO: Optimize this
        let ommer_addrs: BTreeSet<_> = self
            .prior_bridges()
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

        let Self {
            tracking, ommers, ..
        } = self;

        let combined_len = tracking.len() + ommers.len();

        const THREADING_THRESHOLD: usize = 10_000;

        if combined_len < THREADING_THRESHOLD {
            tracking.retain(|addr| ommer_addrs.contains(&addr.sibling()));
            ommers.retain(|addr, _| ommer_addrs.contains(addr));
        } else {
            unlikely(|| {
                // Only spawn threads if there are a ton of ommers to remove.
                thread::scope(|s| {
                    s.spawn(|| {
                        tracking.retain(|addr| ommer_addrs.contains(&addr.sibling()));
                    });
                    s.spawn(|| {
                        ommers.retain(|addr, _| ommer_addrs.contains(addr));
                    });
                });
            });
        }
    }
}

#[cfg(not(feature = "std"))]
impl<H: Hashable + Clone, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Remove data that is not necessary for the currently tracked leaves.
    pub fn garbage_collect_ommers(&mut self) {
        let ommer_addrs: BTreeSet<_> = self
            .prior_bridges()
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
    }
}

/// Utility that can be used to mark cold paths in
/// if-branches, in order to optimize codegen.
#[cold]
#[inline(never)]
fn unlikely<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

#[inline]
fn lookup_prior_bridge_slab_index<H>(
    prior_bridges_slab: &Slab<NonEmptyFrontier<H>>,
    prior_bridges_slab_keys: &[usize],
    position: Position,
) -> Result<usize, usize> {
    prior_bridges_slab_keys.binary_search_by(|&key| {
        let bridge_frontier =
            unsafe { get_prior_bridge_by_slab_key_unchecked(prior_bridges_slab, key) };

        // Compare the probe position with the bridge frontier's position.
        bridge_frontier.position().cmp(&position)
    })
}

#[inline]
unsafe fn get_prior_bridge_by_slab_key_unchecked<H>(
    prior_bridges_slab: &Slab<NonEmptyFrontier<H>>,
    key: usize,
) -> &NonEmptyFrontier<H> {
    #[cfg(debug_assertions)]
    {
        prior_bridges_slab.get(key).unwrap()
    }
    #[cfg(not(debug_assertions))]
    {
        unsafe { prior_bridges_slab.get_unchecked(key) }
    }
}

/// Given an address of an ommer and a contiguous slice of historical
/// frontiers, return a reference to the pre-computed hash of the
/// ommer if it exists.
fn find_precomputed_ommer<'frontiers, H, F>(
    ommer_addr: &Address,
    frontiers: &'frontiers [F],
) -> Option<&'frontiers H>
where
    F: AbstractFrontier<H>,
{
    // An ommer at `ommer_addr` is the root of a completed subtree.
    // This hash is finalized and stored in the frontier's ommers
    // when the leaf immediately following the subtree is added.
    let next_leaf_pos = ommer_addr.position_range_end();

    // Retrieve the candidate frontier. If it doesn't exist in our history,
    // the ommer cannot have been computed yet.
    let frontier_index: usize = next_leaf_pos
        .try_into()
        .expect("32bit platform has usizes that cannot be converted from u64s");
    let candidate_frontier: &F = frontiers.get(frontier_index)?;

    let p: u64 = next_leaf_pos.into();
    let target_level: u64 = ommer_addr.level().into();

    // An address can only be a past ommer for position `p` if the path for `p`
    // at that level goes to the right. This corresponds to the bit for that level
    // being a '1' in the binary representation of `p`.
    let is_past_ommer = (p >> target_level) & 0x1 == 1;
    if !is_past_ommer {
        return None;
    }

    // The index of an ommer in the `ommers` vector is the count of past ommers
    // at all levels below it. This is equivalent to the number of set bits ('1's)
    // in the binary representation of `p` at positions less than `target_level`.
    let mask = (1u64 << target_level) - 1;
    let ommer_index = (p & mask).count_ones() as usize;

    candidate_frontier.ommers().get(ommer_index)
}

/// Recompute the hash for a given address by finding the exact frontier
/// that completed the subtree and calculating its root.
fn recompute_hash_at_address<H, F>(target_address: &Address, frontiers_vec: &[F]) -> Option<H>
where
    H: Clone + Hashable,
    F: AbstractFrontier<H>,
{
    // The state of the tree needed to compute the root of the target_address
    // is perfectly captured by the frontier at the position of the *last leaf*
    // within that address's subtree.
    let end_leaf_pos = target_address.max_position();

    // If we don't have the history for that final leaf, we can't compute the hash.
    let frontier_index: usize = u64::from(end_leaf_pos)
        .try_into()
        .expect("32bit platform has usizes that cannot be converted from u64s");
    let relevant_frontier = to_non_empty_frontier(end_leaf_pos, frontiers_vec.get(frontier_index)?);

    // Calculate the root of this historical frontier, but only up to the level
    // of the target address. This effectively gives us the hash of the subtree.
    Some(relevant_frontier.root(Some(target_address.level())))
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
    impl<H: Clone + Hashable + Send> DynDepthBridgeTree<H> {
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

    impl<H: Hashable + Clone + Send, const DEPTH: u8> testing::Tree<H, ()> for BridgeTree<H, DEPTH> {
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
            BridgeTree::remove_mark_and_gc(self, position).is_ok()
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
        G::Value: Hashable + Clone + Debug + Send + 'static,
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
                tree.prior_bridges().cloned().collect(),
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
                    forked.witness(pos.into()).unwrap_or_else(|_| panic!("Couldn't get merkle proof at {pos:?} with tree: {forked:#?}")),
                );
            }
        }

        #[test]
        fn clone_from_frontier_le_bridge_pos(past_bridges in proptest::collection::vec(any::<bool>(), 100)) {
            let mut tree = BridgeTree::<String, 7>::new();

            for mark in past_bridges.iter().copied() {
                tree.append("a".to_string()).unwrap();
                if mark {
                    tree.mark().unwrap();
                }
            }
            for _remaining in past_bridges.len()..100 {
                tree.append("a".to_string()).unwrap();
            }

            let marked_positions = past_bridges
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(position, mark)| {
                    mark.then_some(Position::from(position as u64))
                });

            for position in marked_positions {
                let cloned = tree.clone_from_frontier_at(position);
                prop_assert!(cloned.current_position() <= Some(position));
            }
        }

        #[test]
        fn clone_from_frontier_on_empty_tree(pos in any::<u64>()) {
            assert_eq!(
                new_tree::<String>().clone_from_frontier_at(Position::from(pos))
                    .current_position(),
                None,
            );
        }
    }

    fn new_tree<H: Hashable + Clone + Debug + Send>() -> BridgeTree<H, 4> {
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

    #[test]
    fn clone_from_frontier() {
        use Retention::*;
        use incrementalmerkletree_testing::Operation;
        use incrementalmerkletree_testing::Operation::Append;

        let ops = vec![
            /* 0 */ Append(String::from_u64(0), Marked),
            /* 1 */ Append(String::from_u64(0), Ephemeral),
            /* 2 */ Append(String::from_u64(0), Marked),
            /* 3 */ Append(String::from_u64(0), Ephemeral),
            /* 4 */ Append(String::from_u64(0), Ephemeral),
            /* 5 */ Append(String::from_u64(0), Marked),
        ];

        let mut tree = new_tree::<String>();
        Operation::apply_all(&ops, &mut tree);

        struct TestCase {
            clone_pos: u64,
            expected_frontier_pos: u64,
        }

        for test_case in [
            TestCase {
                clone_pos: 0,
                expected_frontier_pos: 0,
            },
            TestCase {
                clone_pos: 1,
                expected_frontier_pos: 0,
            },
            TestCase {
                clone_pos: 2,
                expected_frontier_pos: 2,
            },
            TestCase {
                clone_pos: 3,
                expected_frontier_pos: 2,
            },
            TestCase {
                clone_pos: 4,
                expected_frontier_pos: 2,
            },
            TestCase {
                clone_pos: 5,
                expected_frontier_pos: 5,
            },
        ] {
            assert_eq!(
                tree.clone_from_frontier_at(Position::from(test_case.clone_pos))
                    .current_position(),
                Some(Position::from(test_case.expected_frontier_pos))
            );
        }
    }

    #[test]
    fn merge() {
        use Retention::*;
        use incrementalmerkletree_testing::Operation;
        use incrementalmerkletree_testing::Operation::Append;

        // Start by marking leaf at positon 3.
        let ops = vec![
            /* 0 */ Append(String::from_u64(0), Ephemeral),
            /* 1 */ Append(String::from_u64(0), Ephemeral),
            /* 2 */ Append(String::from_u64(0), Ephemeral),
            /* 3 */ Append(String::from_u64(0), Marked),
        ];

        //    aaaa
        //   /    \
        //  aa    aa
        //  / |   | \
        // a  a   a  a
        //    ^      ^
        // 0  1   2  3

        let mut tree = BridgeTree::<String, 2>::new();
        Operation::apply_all(&ops, &mut tree);

        let original_root = tree.root();
        let original_proof_3 = tree.witness(3u64.into()).unwrap();
        assert_eq!(
            compute_root(String::from_u64(0), 3u64.into(), &original_proof_3),
            original_root
        );

        // Now let's mark the tree at position 1.
        let mut forked_tree = tree.clone_from_frontier_at(1u64.into());

        while forked_tree.current_position() != tree.current_position() {
            // Merging will always fail until the trees are caught up
            assert!(tree.merge_with(forked_tree.clone()).is_err());

            forked_tree.append(String::from_u64(0)).unwrap();

            if forked_tree.current_position().unwrap() == 1u64.into() {
                forked_tree.mark().unwrap();
            }
        }

        // Merge the trees.
        assert!(tree.merge_with(forked_tree).is_ok());
        assert_eq!(tree.root(), original_root);

        // Attempt to build merkle proofs with both marked leaves.
        let proof_3 = tree.witness(3u64.into()).unwrap();
        let proof_1 = tree.witness(1u64.into()).unwrap();

        assert_eq!(proof_3, original_proof_3);
        assert_eq!(
            compute_root(String::from_u64(0), 1u64.into(), &proof_1),
            original_root
        );
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

    fn compute_root<H: Hashable>(leaf: H, position: Position, proof: &[H]) -> H {
        let mut address: Address = position.into();

        proof.iter().fold(leaf, |accum_node, sibling| {
            let parent = if address.is_right_child() {
                H::combine(address.level(), sibling, &accum_node)
            } else {
                H::combine(address.level(), &accum_node, sibling)
            };

            address = address.parent();
            parent
        })
    }
}
