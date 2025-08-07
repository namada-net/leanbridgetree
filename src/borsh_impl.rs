use std::io::{self, Read, Write};

use borsh::{BorshDeserialize, BorshSerialize};

use super::*;

impl<H, const DEPTH: u8> BorshDeserialize for BridgeTree<H, DEPTH>
where
    H: BorshDeserialize,
{
    fn deserialize_reader<R: Read>(reader: &mut R) -> io::Result<Self> {
        fn deserialize_u32_as_usize<R: Read>(reader: &mut R) -> io::Result<usize> {
            let x = u32::deserialize_reader(reader)?;
            usize::try_from(x).map_err(io::Error::other)
        }

        fn deserialize_frontier<H: BorshDeserialize, R: Read>(
            reader: &mut R,
        ) -> io::Result<NonEmptyFrontier<H>> {
            let frontier_position = Position::from(u64::deserialize_reader(reader)?);

            let frontier_leaf = H::deserialize_reader(reader)?;

            let frontier_ommers_len = deserialize_u32_as_usize(reader)?;
            let mut frontier_ommers = Vec::with_capacity(frontier_ommers_len);
            for _ in 0..frontier_ommers_len {
                frontier_ommers.push(H::deserialize_reader(reader)?);
            }

            NonEmptyFrontier::from_parts(frontier_position, frontier_leaf, frontier_ommers).map_err(
                |err| {
                    io::Error::other(format!(
                        "failed to rebuild NonEmptyFrontier from deserialized \
                     data: {err:?}"
                    ))
                },
            )
        }

        // `frontier`
        let has_leaves = bool::deserialize_reader(reader)?;
        if !has_leaves {
            return Ok(Self::new());
        }
        let frontier = deserialize_frontier(reader)?;

        // `tracking`
        let tracking_len = deserialize_u32_as_usize(reader)?;
        let mut tracking = BTreeSet::new();
        for _ in 0..tracking_len {
            let level = u8::deserialize_reader(reader)?;
            let index = u64::deserialize_reader(reader)?;
            tracking.insert(Address::from_parts(level.into(), index));
        }

        // `ommers`
        let ommers_len = deserialize_u32_as_usize(reader)?;
        let mut ommers = BTreeMap::new();
        for _ in 0..ommers_len {
            let level = u8::deserialize_reader(reader)?;
            let index = u64::deserialize_reader(reader)?;
            let ommer = H::deserialize_reader(reader)?;
            ommers.insert(Address::from_parts(level.into(), index), ommer);
        }

        // `bridges`
        let bridges_len = deserialize_u32_as_usize(reader)?;
        let mut bridges = Vec::with_capacity(bridges_len);
        for _ in 0..bridges_len {
            bridges.push(deserialize_frontier(reader)?);
        }

        Self::from_parts(Some(frontier), bridges, tracking, ommers).map_err(|err| {
            io::Error::other(format!(
                "failed to rebuild BridgeTree from deserialized \
                 data: {err:?}"
            ))
        })
    }
}

impl<H, const DEPTH: u8> BorshSerialize for BridgeTree<H, DEPTH>
where
    H: BorshSerialize,
{
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        fn serialize_usize_as_u32<W: Write>(x: usize, writer: &mut W) -> io::Result<()> {
            let x = u32::try_from(x).map_err(io::Error::other)?;
            BorshSerialize::serialize(&x, writer)
        }

        fn serialize_frontier<H: BorshSerialize, W: Write>(
            frontier: &NonEmptyFrontier<H>,
            writer: &mut W,
        ) -> io::Result<()> {
            BorshSerialize::serialize(&u64::from(frontier.position()), writer)?;
            BorshSerialize::serialize(frontier.leaf(), writer)?;

            serialize_usize_as_u32(frontier.ommers().len(), writer)?;
            for ommer in frontier.ommers() {
                BorshSerialize::serialize(ommer, writer)?;
            }

            Ok(())
        }

        let Self {
            frontier,
            tracking,
            ommers,
            prior_bridges_slab: _,
            prior_bridges_slab_keys,
        } = self;

        let Some(frontier) = frontier else {
            // `frontier`
            return BorshSerialize::serialize(&false, writer);
        };

        // `frontier`
        BorshSerialize::serialize(&true, writer)?;
        serialize_frontier(frontier, writer)?;

        // `tracking`
        serialize_usize_as_u32(tracking.len(), writer)?;
        for addr in tracking {
            BorshSerialize::serialize(&u8::from(addr.level()), writer)?;
            BorshSerialize::serialize(&addr.index(), writer)?;
        }

        // `ommers`
        serialize_usize_as_u32(ommers.len(), writer)?;
        for (addr, ommer) in ommers {
            BorshSerialize::serialize(&u8::from(addr.level()), writer)?;
            BorshSerialize::serialize(&addr.index(), writer)?;
            BorshSerialize::serialize(ommer, writer)?;
        }

        // `bridges`
        serialize_usize_as_u32(prior_bridges_slab_keys.len(), writer)?;
        for frontier in self.prior_bridges() {
            serialize_frontier(frontier, writer)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use incrementalmerkletree_testing::TestHashable;

    use super::*;

    #[test]
    fn test_bridge_tree_borsh_roundtrip() {
        let mut tree = BridgeTree::<String, 4>::new();

        // test empty tree
        let serialized = borsh::to_vec(&tree).unwrap();
        let deserialized = BridgeTree::try_from_slice(&serialized).unwrap();
        assert_eq!(BridgeTree::<String, 4>::new(), deserialized);

        // test non-empty tree
        tree.append(String::from_u64(0u64)).unwrap();
        tree.mark().unwrap();
        tree.append(String::from_u64(1u64)).unwrap();
        tree.append(String::from_u64(2u64)).unwrap();
        tree.append(String::from_u64(3u64)).unwrap();
        tree.mark().unwrap();
        tree.append(String::from_u64(4u64)).unwrap();
        tree.append(String::from_u64(5u64)).unwrap();
        tree.mark().unwrap();
        tree.append(String::from_u64(6u64)).unwrap();
        tree.append(String::from_u64(7u64)).unwrap();
        tree.append(String::from_u64(8u64)).unwrap();
        tree.append(String::from_u64(9u64)).unwrap();

        let serialized = borsh::to_vec(&tree).unwrap();
        let deserialized = BridgeTree::try_from_slice(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }
}
