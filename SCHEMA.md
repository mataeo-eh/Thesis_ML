# SCHEMA.md

Derived from the five parquet fixtures in `tests/fixtures/`:

- `match_4745721_game_state.parquet`
- `match_4745722_game_state.parquet`
- `match_4745723_game_state.parquet`
- `match_4745724_game_state.parquet`
- `match_4745725_game_state.parquet`

This document records the actual extractor output shape for prompt 002. It is a schema description only; tokenization code must not be implemented against it until owner approval.

## Top-Level Structure

Each fixture is one replay stored as a wide parquet table.

- One row is one sampled game-state snapshot.
- Rows are ordered by `game_loop` ascending.
- Snapshot time is represented by:
  - `game_loop`: `int64`, SC2 game loop/frame index.
  - `timestamp_seconds`: `double`, elapsed game time in seconds.
- `Messages`: `large_string`, optional replay/chat/tag metadata. Values may be null, plain text, or serialized list-looking strings.
- Entity records are not nested rows. They are encoded as groups of wide columns.

Fixture dimensions:

| File | Rows | Columns |
|---|---:|---:|
| `match_4745721_game_state.parquet` | 1136 | 23110 |
| `match_4745722_game_state.parquet` | 397 | 3820 |
| `match_4745723_game_state.parquet` | 537 | 9927 |
| `match_4745724_game_state.parquet` | 426 | 3897 |
| `match_4745725_game_state.parquet` | 457 | 4672 |

## Entity Column Grammar

Entity-instance fields use this column pattern:

```text
{player}_{bot_name}_{entity_type}_{instance_id}_{attribute}
```

Observed examples:

```text
p1_jimmybott_scv_001_pos_(X,Y,Z)
p1_jimmybott_scv_001_health
p2_who_zergling_084_pos_(X,Y,Z)
```

Parsed fields:

| Schema field | Source | Example | SPEC role |
|---|---|---|---|
| Owner/player | Column prefix | `p1`, `p2` | Raw allegiance input field. Not token identity. Later training can map one player to self and the other to enemy per training perspective. |
| Bot name | Column middle before entity type | `jimmybott`, `who`, `phobos` | Metadata only. Not token identity. |
| Entity type | Last segment before `instance_id` | `scv`, `marine`, `zergling`, `unknown(1943)` | Token identity. Content vocabulary is raw entity-type tokens only. |
| Instance id | Three digit segment | `001`, `084`, `201` | Deterministic within-type tiebreak candidate for canonical ordering. The fixtures do not expose a separate persistent own-unit game tag field. |
| Attribute | Column suffix after instance id | `pos_(X,Y,Z)`, `health`, `is_flying` | Raw input-only contextual/stat field. Not token identity. |

Observed player prefixes: `p1`, `p2`.

Observed bot names in fixtures: `avocados`, `caninana`, `jimmybott`, `phobos`, `sharkbot`, `who`.

Observed instance-id range in fixtures: `001` through `201`.

## Entity Attributes

All entity attribute columns are stored as parquet `large_string`, even when the logical value is numeric or boolean.

Observed attributes:

```text
add_on_tag
armor_upgrade_level
assigned_harvesters
attack_upgrade_level
buff_duration_max
buff_duration_remain
buff_ids
build_progress
cargo_space_max
cargo_space_taken
cloak
detect_range
display_type
energy
engaged_target_tag
facing
health
ideal_harvesters
is_active
is_burrowed
is_flying
is_hallucination
is_powered
order_count
pos_(X,Y,Z)
radar_range
radius
rally_tag
rally_x
rally_y
shield_upgrade_level
shields
weapon_cooldown
```

Representative values:

| Attribute | Example values | SPEC role |
|---|---|---|
| `pos_(X,Y,Z)` | `(68.481689453125, 176.035888671875, 11.990320205688477)` | Exact map position. Input-only contextual value. Never token identity. |
| `health` | `45.0/45.0`, `6.0/45.0` | Unit stat. Input-only raw value. |
| `energy` | `51.23046875/200.0` | Unit stat. Input-only raw value. |
| `shields` | shield values where present | Unit stat. Input-only raw value. |
| `facing` | `0.22607851028442383` | Unit stat/context. Input-only raw value. |
| `radius` | `0.375` | Unit stat/context. Input-only raw value. |
| `build_progress` | `1.0` | Unit/building stat. Input-only raw value. |
| `is_flying`, `is_burrowed`, `is_hallucination`, `is_active`, `is_powered` | `True`, `False` | Unit flags. Input-only raw values. |
| `attack_upgrade_level`, `armor_upgrade_level`, `shield_upgrade_level` | `0`, `1`, `2` | Unit stat. Input-only raw value. |
| `cargo_space_taken`, `cargo_space_max`, `order_count` | `0`, `1` | Unit stat. Input-only raw value. |
| `buff_ids` | `[]`, `[271]`, `[271, 33]` | Unit stat/list. Input-only raw value. |
| `engaged_target_tag`, `rally_tag`, `add_on_tag` | integer-looking tag strings | Relationship fields. Input-only raw value. Not entity identity. |
| `rally_x`, `rally_y` | `67.0`, `176.5` | Rally-point coordinates. Not token identity. |

Observed lifecycle/status sentinels in entity attribute columns:

```text
completed
destroyed
building_started
inside refinery
inside orbitalcommand
```

Some rows contain these sentinel strings instead of numeric/boolean/tuple values. Owner-approved Phase 2 rule: every non-null entity attribute value indicates that the entity instance is present for that snapshot. `destroyed` is the final valid frame of an entity life, and following rows should be null for that entity. `inside ...` values also indicate that the unit still exists even if it is not literally visible on the map.

## Entity Types

Across the five fixtures, observed entity types are:

```text
adept
adeptphaseshift
armory
assimilator
banshee
barracks
barracksreactor
barrackstechlab
bunker
changeling
cocoon
commandcenter
cyberneticscore
cyclone
drone
engineeringbay
evolutionchamber
extractor
factory
factorytechlab
forge
fusioncore
gateway
hatchery
immortal
infestationpit
kd8charge
larva
marauder
marine
medivac
mule
nexus
overlord
photoncannon
probe
pylon
queen
reaper
refinery
roach
roachwarren
roboticsfacility
scv
shieldbattery
siegetank
spawningpool
spire
stalker
starport
starportreactor
starporttechlab
supplydepot
unknown(1943)
vikingfighter
warpprism
zealot
zergling
```

These names are raw content-token vocabulary candidates observed in the fixtures. The full prompt 002 vocabulary should be seeded from `data/Token_Dictionary.json`, which was built from the broader processed replay dataset and therefore contains many entity and upgrade names not visible in the five fixtures. Token identity must be location-agnostic and must not include positions, regions, counts, frame numbers, timestamps, or player ownership.

## Snapshot Ordering

Snapshots are represented by rows.

- Primary row order field: `game_loop`.
- Secondary time field: `timestamp_seconds`.
- The fixtures appear sorted by `game_loop`.
- `timestamp_seconds` is absolute elapsed game time. It may remain source metadata for ordering and post-sampling evaluation, but must not enter model-facing features, embeddings, attention inputs, targets, or the vocabulary.

## Player Ownership And Training Perspective

The extractor output is neutral. It has no explicit `self` or `enemy` field.

Ownership is encoded in column prefixes:

- `p1_...`: entity/resource/feature belongs to player 1.
- `p2_...`: entity/resource/feature belongs to player 2.

For model training, self/enemy allegiance must be derived later by selecting a perspective:

- Training example from player 1 perspective: `p1` is self, `p2` is enemy.
- Training example from player 2 perspective: `p2` is self, `p1` is enemy.

This derived allegiance is a raw per-token field for the later team-flag embedding. It is not part of token identity.

## Aggregate And Non-Entity Columns

The fixtures also contain resource, upgrade, and derived feature columns.

Resource/upgrade columns:

```text
p1_minerals
p1_vespene
p1_supply_used
p1_supply_cap
p1_collection_rate_minerals
p1_collection_rate_vespene
p1_upgrades
p2_minerals
p2_vespene
p2_supply_used
p2_supply_cap
p2_collection_rate_minerals
p2_collection_rate_vespene
p2_upgrades
```

Derived feature/count columns include:

```text
p1_<entity_type>_count
p2_<entity_type>_count
p1_total_unit_types
p2_total_unit_types
p1_production_building_count
p2_production_building_count
p1_has_air_units
p2_has_air_units
```

These fields are present in the parquet. Counts must emerge from repeated entity tokens, not from count columns. Resource and aggregate columns may be useful for later research or diagnostics, but prompt 002 tokenization should derive entity tokens from entity-instance columns, not from aggregate counts.

Upgrade handling:

- `p1_upgrades` and `p2_upgrades` are cumulative per-player upgrade-list fields.
- The list column names themselves must not enter the vocabulary.
- Each individual upgrade name from the processed-dataset token dictionary should be included as a location-agnostic content token.
- Resource fields (`minerals`, `vespene`, `supply`, collection rates) must not enter the vocabulary.

## Fields Not Used In Vocabulary

These parquet values must not enter the vocabulary:

- `game_loop`
- `timestamp_seconds`
- `Messages`
- exact `(X,Y,Z)` positions
- `rally_x`, `rally_y`
- resource counts and rates
- upgrade-list column names
- aggregate unit-count/feature columns
- player ownership prefix (`p1`/`p2`)
- bot names
- unit stats and flags
- entity lifecycle/status sentinel strings

## SPEC Mapping

| SPEC concept | Derived parquet source |
|---|---|
| Entity type token identity | Entity type embedded in column name: `{player}_{bot_name}_{entity_type}_{instance_id}_{attribute}` |
| Unit instance for deterministic ordering | Three-digit `instance_id` embedded in column name; owner-approved as the within-type tiebreak |
| Map position | `pos_(X,Y,Z)` attribute value, exact tuple string |
| Unit stats | Remaining entity attributes such as `health`, `energy`, `build_progress`, flags, upgrades, cargo, buffs |
| Absolute game clock (non-model metadata only) | `timestamp_seconds` row field; excluded by the model-facing feature allowlist |
| Snapshot/timestep | One parquet row, ordered by `game_loop` |
| Self/enemy allegiance | Derived later from `p1`/`p2` owner prefix plus selected training perspective |
| Vocabulary exclusions | Coordinates, timestamps, frame numbers, aggregate counts, resources, ownership, bot names, stats |

## Schema Approval Decisions

Owner approval received before Phase 2 implementation:

1. The three-digit `instance_id` is acceptable for tiebreaking.
2. Every non-null entity attribute value indicates an active/present entity for that snapshot. `destroyed` is the last valid frame before disappearance, and `inside ...` statuses still represent existing units.
3. Units, buildings, and individual upgrades belong in the content vocabulary. Resource fields, derived features, count columns, frame/time fields, coordinates, and upgrade-list column names do not.
