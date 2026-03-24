use alloc::format;
use core::fmt;
use core::marker::PhantomData;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use super::*;

impl<H, const DEPTH: u8> Serialize for BridgeTree<H, DEPTH>
where
    H: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Collect the bridges via the public method before destructuring
        // to avoid any borrow-checker edge cases.
        let bridges_vec: Vec<_> = self
            .prior_bridges()
            .map(|f| (u64::from(f.position()), f.leaf(), f.ommers()))
            .collect();

        // Destructure to handle future codebase updates (compiler will catch changes)
        let Self {
            frontier,
            tracking,
            ommers,
            prior_bridges_slab: _,
            prior_bridges_slab_keys: _,
        } = self;

        let mut state = serializer.serialize_struct("BridgeTree", 4)?;

        // Map `NonEmptyFrontier` into a compact tuple: (position, leaf, ommers)
        let frontier_tuple = frontier
            .as_ref()
            .map(|f| (u64::from(f.position()), f.leaf(), f.ommers()));
        state.serialize_field("frontier", &frontier_tuple)?;

        // Map `Address` into a compact tuple: (level, index)
        let tracking_vec: Vec<(u8, u64)> = tracking
            .iter()
            .map(|addr| (u8::from(addr.level()), addr.index()))
            .collect();
        state.serialize_field("tracking", &tracking_vec)?;

        // Serialize maps as sequences of key-value tuples to avoid
        // issues with non-string keys in formats like JSON.
        let ommers_vec: Vec<((u8, u64), &H)> = ommers
            .iter()
            .map(|(addr, h)| ((u8::from(addr.level()), addr.index()), h))
            .collect();
        state.serialize_field("ommers", &ommers_vec)?;

        state.serialize_field("bridges", &bridges_vec)?;

        state.end()
    }
}

impl<'de, H, const DEPTH: u8> Deserialize<'de> for BridgeTree<H, DEPTH>
where
    H: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        enum Field {
            Frontier,
            Tracking,
            Ommers,
            Bridges,
        }

        struct FieldVisitor;
        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = Field;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("`frontier`, `tracking`, `ommers`, or `bridges`")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Field, E> {
                match value {
                    "frontier" => Ok(Field::Frontier),
                    "tracking" => Ok(Field::Tracking),
                    "ommers" => Ok(Field::Ommers),
                    "bridges" => Ok(Field::Bridges),
                    _ => Err(de::Error::unknown_field(value, FIELDS)),
                }
            }
        }

        impl<'de> Deserialize<'de> for Field {
            #[inline]
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct BridgeTreeVisitor<H, const DEPTH: u8>(PhantomData<H>);

        impl<'de, H, const DEPTH: u8> Visitor<'de> for BridgeTreeVisitor<H, DEPTH>
        where
            H: Deserialize<'de>,
        {
            type Value = BridgeTree<H, DEPTH>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct BridgeTree")
            }

            // Fallback for compact/binary sequences (like bincode)
            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let frontier_tuple = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let tracking_vec = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let ommers_vec = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let bridges_vec = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;

                build_tree::<DEPTH, V::Error, _>(
                    frontier_tuple,
                    tracking_vec,
                    ommers_vec,
                    bridges_vec,
                )
            }

            // Standard implementation for maps (like JSON)
            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut frontier_tuple: Option<Option<(u64, H, Vec<H>)>> = None;
                let mut tracking_vec: Option<Vec<(u8, u64)>> = None;
                let mut ommers_vec: Option<Vec<((u8, u64), H)>> = None;
                let mut bridges_vec: Option<Vec<(u64, H, Vec<H>)>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Frontier => {
                            if frontier_tuple.is_some() {
                                return Err(de::Error::duplicate_field("frontier"));
                            }
                            frontier_tuple = Some(map.next_value()?);
                        }
                        Field::Tracking => {
                            if tracking_vec.is_some() {
                                return Err(de::Error::duplicate_field("tracking"));
                            }
                            tracking_vec = Some(map.next_value()?);
                        }
                        Field::Ommers => {
                            if ommers_vec.is_some() {
                                return Err(de::Error::duplicate_field("ommers"));
                            }
                            ommers_vec = Some(map.next_value()?);
                        }
                        Field::Bridges => {
                            if bridges_vec.is_some() {
                                return Err(de::Error::duplicate_field("bridges"));
                            }
                            bridges_vec = Some(map.next_value()?);
                        }
                    }
                }

                let frontier_tuple = frontier_tuple.unwrap_or(None);
                let tracking_vec =
                    tracking_vec.ok_or_else(|| de::Error::missing_field("tracking"))?;
                let ommers_vec = ommers_vec.ok_or_else(|| de::Error::missing_field("ommers"))?;
                let bridges_vec = bridges_vec.ok_or_else(|| de::Error::missing_field("bridges"))?;

                build_tree::<DEPTH, V::Error, _>(
                    frontier_tuple,
                    tracking_vec,
                    ommers_vec,
                    bridges_vec,
                )
            }
        }

        const FIELDS: &[&str] = &["frontier", "tracking", "ommers", "bridges"];
        deserializer.deserialize_struct("BridgeTree", FIELDS, BridgeTreeVisitor(PhantomData))
    }
}

// Helper block to rebuild the logic from parts so we don't repeat code
// in both `visit_seq` and `visit_map`.
fn build_tree<const DEPTH: u8, E: de::Error, H>(
    frontier_tuple: Option<(u64, H, Vec<H>)>,
    tracking_vec: Vec<(u8, u64)>,
    ommers_vec: Vec<((u8, u64), H)>,
    bridges_vec: Vec<(u64, H, Vec<H>)>,
) -> Result<BridgeTree<H, DEPTH>, E> {
    let frontier = match frontier_tuple {
        Some((pos, leaf, f_ommers)) => Some(
            NonEmptyFrontier::from_parts(Position::from(pos), leaf, f_ommers).map_err(|err| {
                de::Error::custom(format!("failed to rebuild NonEmptyFrontier: {err:?}"))
            })?,
        ),
        None => None,
    };

    let mut tracking = BTreeSet::new();
    for (level, index) in tracking_vec {
        tracking.insert(Address::from_parts(level.into(), index));
    }

    let mut ommers = BTreeMap::new();
    for ((level, index), h) in ommers_vec {
        ommers.insert(Address::from_parts(level.into(), index), h);
    }

    let mut bridges = Vec::with_capacity(bridges_vec.len());
    for (pos, leaf, f_ommers) in bridges_vec {
        bridges.push(
            NonEmptyFrontier::from_parts(Position::from(pos), leaf, f_ommers).map_err(|err| {
                de::Error::custom(format!(
                    "failed to rebuild NonEmptyFrontier for bridge: {err:?}"
                ))
            })?,
        );
    }

    BridgeTree::from_parts(frontier, bridges, tracking, ommers)
        .map_err(|err| de::Error::custom(format!("failed to rebuild BridgeTree: {err:?}")))
}

#[cfg(test)]
mod tests {
    use incrementalmerkletree_testing::TestHashable;

    use super::*;

    #[test]
    fn test_bridge_tree_serde_json_roundtrip() {
        let mut tree = BridgeTree::<String, 4>::new();

        let serialized = serde_json::to_string(&tree).unwrap();
        println!("{serialized}");
        let deserialized: BridgeTree<String, 4> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(BridgeTree::<String, 4>::new(), deserialized);

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

        let serialized = serde_json::to_string(&tree).unwrap();
        println!("{serialized}");
        let deserialized: BridgeTree<String, 4> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }
}
