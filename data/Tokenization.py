"""
Tokenization.py: Master entity list builder for the SC2 ML tokenization schema.

PURPOSE
-------
Derives the canonical set of SC2 entity names (units + buildings + upgrades)
from two complementary sources, then writes them to two output files:

  1. Dataset scan  -- reads every parquet file in the project data directory.
                      For unit/building columns, parses column names with
                      ENTITY_COL_RE and collects every entity type name observed.
                      For upgrades, reads the p1_upgrades / p2_upgrades columns
                      (cumulative chronological lists per row) and gathers every
                      upgrade name observed across all games.

  2. pysc2 enums   -- iterates pysc2.lib.units.Terran / Protoss / Zerg for
                      units and buildings, and pysc2.lib.upgrades.Upgrades for
                      upgrades, to catch entities that exist in the game but
                      haven't appeared in the current dataset yet.

Entities found in EITHER source are included in both output files.  Entities
already in the JSONs that disappear from both sources are NOT removed -- both
files are append-only so token IDs remain stable across dataset updates.

UPGRADE TOKEN ID OFFSET
-----------------------
SC2's unit-type ID space and upgrade ID space are independent enums in the
engine and overlap heavily (e.g. unit ID 64 = FleetBeacon, upgrade ID 64 =
Burrow).  To merge both into a single flat Token_Dictionary.json without
collisions, every upgrade token ID is stored as `pysc2_upgrade_id + 100000`
(the engine ID is recoverable as `token_id - 100000`).  Random fallback IDs
for upgrade names that pysc2 does not recognise are also drawn from the
[100000, 199999] range, so the entire range is reserved for upgrades and
unit/building IDs (always < 2000) never overlap.

UPDATING THE LIST
-----------------
Re-run this file directly whenever:
  - New parquet files have been added to the dataset.
  - pysc2 has been updated with renamed or new unit entries.

  python ML_PoC/Tokenization.py [--data-dir path/to/data]

The script merges new findings without disturbing existing entries, then prints
a summary of what was added.

OUTPUT SCHEMA (Entity_List.json)
---------------------------------
Human-readable entity list organised by category and race.  Useful as a
reference and for downstream code that needs race-aware lookups.

{
  "metadata": {
    "generated_at":        "<ISO timestamp of last update>",
    "parquet_files_scanned": <int>,
    "total_entities":        <int>
  },
  "buildings": {
    "terran":  [...sorted list of terran building names...],
    "protoss": [...],
    "zerg":    [...],
    "unclassified": [...]   // in data/pysc2 but not mappable to a race
  },
  "units": {
    "terran":  [...sorted list of terran unit names...],
    "protoss": [...],
    "zerg":    [...],
    "unclassified": [...]
  },
  "upgrades": {
    "terran":  [...sorted list of terran upgrade names...],
    "protoss": [...],
    "zerg":    [...],
    "unclassified": [...]   // upgrades whose race could not be inferred
  }
}

OUTPUT SCHEMA (Token_Dictionary.json)
--------------------------------------
Flat vocabulary mapping integer token IDs to their canonical string names.

For units and buildings: token ID == pysc2 unit-type ID == game engine ID
(e.g. "48": "marine"). Unknown IDs encountered by the extractor are stored
as "unknown_<ID>" so the token is still uniquely identifiable.

For upgrades: token ID == pysc2 upgrade ID + 100000 (e.g. Burrow whose
engine ID is 64 is stored as "100064": "burrow"). The +100000 offset keeps
the upgrade ID space disjoint from the unit/building ID space, since the
engine reuses small integers across both. Upgrade names that pysc2 does
not recognise are assigned a deterministic-random unused integer in
[100000, 199999] (seeded by name so the same upgrade always gets the same
ID across re-runs).

{
  "metadata": {
    "generated_at":        "<ISO timestamp of last update>",
    "total_tokens":          <int>
  },
  "tokens": {
    "48":     "marine",          // unit/building: engine ID directly
    "1943":   "unknown_1943",    // unit IDs not in pysc2 enums
    "100064": "burrow",          // upgrade: engine ID + 100000 (Burrow)
    "100015": "stimpack",        // upgrade: engine ID + 100000 (Stimpack)
    "187234": "myodd_upgrade"    // upgrade not in pysc2: random unused ID
  }
}

All string names are lowercase.

Dependencies:
  - pyarrow       (reads parquet column schemas without loading row data)
  - pandas        (reads upgrade column row data for dataset upgrade scan)
  - pysc2.lib.units
  - pysc2.lib.upgrades
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pysc2.lib import units as sc2_units
from pysc2.lib import upgrades as sc2_upgrades


# ---------------------------------------------------------------------------
# Upgrade token ID offset
# ---------------------------------------------------------------------------
# Upgrade IDs from pysc2 are stored in Token_Dictionary.json as
# `pysc2_upgrade_id + UPGRADE_TOKEN_ID_OFFSET`. The unit/building ID space
# tops out around 2000, so an offset of 100000 leaves a clean disjoint
# window [100000, 199999] for all upgrade tokens (both pysc2-mapped and
# random fallbacks).
UPGRADE_TOKEN_ID_OFFSET: int = 100000
UPGRADE_RANDOM_ID_LO:    int = 100000
UPGRADE_RANDOM_ID_HI:    int = 199999


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels up from ML_PoC/Tokenization.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directory to search for parquet files (all subdirs under data/)
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"

# Output JSONs live alongside this file
ENTITY_LIST_PATH   = Path(__file__).resolve().parent / "Entity_List.json"
TOKEN_DICT_PATH    = Path(__file__).resolve().parent / "Token_Dictionary.json"

# Pattern that matches unknown entity names produced by the extractor when it
# encounters a game-engine unit-type ID not present in the pysc2 enums.
# Example: "unknown(1943)" -> captured group = "1943"
_UNKNOWN_NAME_RE = re.compile(r"^unknown\((\d+)\)$")

# Column name regex -- matches {player}_{botname}_{entitytype}_{seq_id}_{attr}
# Group 2 is the combined botname+entitytype "middle"; entity type is the last
# underscore-delimited segment: middle.rsplit('_', 1)[-1]
_ENTITY_COL_RE = re.compile(r"^(p[12])_(.+)_(\d+)_(.+)$")


# ---------------------------------------------------------------------------
# BUILDING_TYPES
# ---------------------------------------------------------------------------
# Authoritative frozenset of all SC2 building names across all three races.
# Copied from SC2-gamestate-extractor/src_new/shared_constants.py.
#
# Role here: used as the building/unit classifier. Any entity name in this set
# is a building; everything else is treated as a unit.
#
# Building ID cross-reference notes (from original shared_constants.py):
#   ID 133 --> WarpGate (Protoss)      [NOT TechLab; TechLab = ID 5]
#   ID 138 --> CreepTumorQueen (Zerg)  [ability pseudo-entity, not a content token]
#   ID 142 --> NydusCanal (Zerg)       [NOT StarportReactor (ID 42)]

BUILDING_TYPES: frozenset = frozenset({

    # -----------------------------------------------------------------------
    # TERRAN BUILDINGS
    # -----------------------------------------------------------------------
    "commandcenter",          # ID 18
    "orbitalcommand",         # ID 132
    "planetaryfortress",      # ID 130
    "commandcenterflying",    # ID 36
    "orbitalcommandflying",   # ID 134
    "barracksflying",         # ID 46
    "factoryflying",          # ID 43
    "starportflying",         # ID 44
    "supplydepot",            # ID 19
    "supplydepotlowered",     # ID 47
    "refinery",               # ID 20
    "refineryrich",           # ID 1960
    "barracks",               # ID 21
    "factory",                # ID 27
    "starport",               # ID 28
    "engineeringbay",         # ID 22
    "missileturret",          # ID 23
    "bunker",                 # ID 24
    "sensortower",            # ID 25
    "ghostacademy",           # ID 26
    "armory",                 # ID 29
    "fusioncore",             # ID 30
    "techlab",                # ID 5
    "reactor",                # ID 6
    "barrackstechlab",        # ID 37
    "barracksreactor",        # ID 38
    "factorytechlab",         # ID 39
    "factoryreactor",         # ID 40
    "starporttechlab",        # ID 41
    "starportreactor",        # ID 42

    # -----------------------------------------------------------------------
    # PROTOSS BUILDINGS
    # -----------------------------------------------------------------------
    "nexus",                  # ID 59
    "pylon",                  # ID 60
    "assimilator",            # ID 61
    "assimilatorrich",        # ID 1955
    "gateway",                # ID 62
    "warpgate",               # ID 133
    "forge",                  # ID 63
    "fleetbeacon",            # ID 64
    "twilightcouncil",        # ID 65
    "photoncannon",           # ID 66
    "shieldbattery",          # ID 1910
    "stargate",               # ID 67
    "templararchive",         # ID 68
    "darkshrine",             # ID 69
    "roboticsbay",            # ID 70
    "roboticsfacility",       # ID 71
    "cyberneticscore",        # ID 72

    # -----------------------------------------------------------------------
    # ZERG BUILDINGS
    # -----------------------------------------------------------------------
    "hatchery",               # ID 86
    "lair",                   # ID 100
    "hive",                   # ID 101
    "extractor",              # ID 88
    "extractorrich",          # ID 1956
    "spawningpool",           # ID 89
    "evolutionchamber",       # ID 90
    "hydraliskden",           # ID 91
    "spire",                  # ID 92
    "greaterspire",           # ID 102
    "ultraliskcavern",        # ID 93
    "infestationpit",         # ID 94
    "banelingnest",           # ID 96
    "roachwarren",            # ID 97
    "lurkerden",              # ID 504
    "nydusnetwork",           # ID 95
    "nyduscanal",             # ID 142
    "spinecrawler",           # ID 98
    "spinecrawleruprooted",   # ID 139
    "sporecrawler",           # ID 99
    "sporecrawleruprooted",   # ID 140
})


# Entity types that the extractor deliberately suppresses because they are
# engine-created ability pseudo-entities, not durable game-state content tokens.
UNTRACKED_ENTITY_TYPES: frozenset = frozenset({
    "kd8charge",
    "creeptumor",
    "creeptumorburrowed",
    "creeptumorqueen",
})


# ---------------------------------------------------------------------------
# Race lookup tables (built once from pysc2 enums)
# ---------------------------------------------------------------------------
# Maps lowercase unit/building name -> race string.
# Used to classify entities found in the dataset by race.

def _build_race_map() -> dict[str, str]:
    """
    Build a name -> race mapping from pysc2 enums.

    Iterates Terran, Protoss, and Zerg enums and lowercases each member name.
    If a name appears in multiple races (shouldn't happen in SC2 but defensive),
    the last race wins -- this is intentional since the mapping is used purely
    for classification and collisions are not expected.

    Returns:
        dict[str, str]: {lowercase_name: "terran" | "protoss" | "zerg"}

    Called by: module-level _RACE_MAP constant.
    """
    race_map = {}
    for enum, race_label in (
        (sc2_units.Terran,  "terran"),
        (sc2_units.Protoss, "protoss"),
        (sc2_units.Zerg,    "zerg"),
    ):
        for member in enum:
            race_map[member.name.lower()] = race_label
    return race_map


_RACE_MAP: dict[str, str] = _build_race_map()


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------

def scan_parquet_entities(data_dir: Path) -> tuple[set[str], int]:
    """
    Scan all parquet files under data_dir for entity type names.

    Reads only the column schema of each file (no row data loaded) for speed.
    Applies ENTITY_COL_RE to each column name and extracts entity type from
    the "middle" group via middle.rsplit('_', 1)[-1].

    Args:
        data_dir: Root directory to search recursively for *.parquet files.

    Returns:
        (entity_names, file_count): set of lowercase entity name strings found
        in the dataset, and the number of parquet files scanned.

    Called by: build_entity_list(), update_entity_list()
    """
    entity_names: set[str] = set()
    parquet_files = list(data_dir.rglob("*.parquet"))

    for path in parquet_files:
        try:
            schema = pq.read_schema(path)
        except Exception as exc:
            print(f"  [WARN] Could not read schema of {path.name}: {exc}", file=sys.stderr)
            continue

        for col in schema.names:
            match = _ENTITY_COL_RE.match(col)
            if match:
                middle = match.group(2)
                entity_name = middle.rsplit("_", 1)[-1]
                if entity_name not in UNTRACKED_ENTITY_TYPES:
                    entity_names.add(entity_name)

    return entity_names, len(parquet_files)


# ---------------------------------------------------------------------------
# pysc2 entity scan
# ---------------------------------------------------------------------------

def scan_pysc2_entities() -> set[str]:
    """
    Collect all entity names from pysc2 Terran, Protoss, and Zerg enums.

    Returns:
        set[str]: Lowercase member names from all three race enums.

    Called by: build_entity_list(), update_entity_list()
    """
    names: set[str] = set()
    for enum in (sc2_units.Terran, sc2_units.Protoss, sc2_units.Zerg):
        for member in enum:
            name = member.name.lower()
            if name not in UNTRACKED_ENTITY_TYPES:
                names.add(name)
    return names


# ---------------------------------------------------------------------------
# Upgrade scans (pysc2 + dataset)
# ---------------------------------------------------------------------------

def scan_pysc2_upgrades() -> dict[str, int]:
    """
    Build a name -> engine-ID map from the pysc2 Upgrades enum.

    Each pysc2.lib.upgrades.Upgrades member exposes:
      member.name  -- the engine's PascalCase upgrade name (e.g. "Burrow")
      member.value -- the engine's integer upgrade ID (e.g. 64)

    The returned dict is keyed by lowercase name so it matches the
    lowercase-everywhere convention used by the unit/building scans.

    Returns:
        dict[str, int]: {lowercase_upgrade_name: pysc2_upgrade_id}

    Called by: scan_pysc2_upgrade_names(), build_token_dictionary()
    """
    return {member.name.lower(): member.value for member in sc2_upgrades.Upgrades}


def scan_pysc2_upgrade_names() -> set[str]:
    """
    Convenience: just the lowercase names from pysc2's Upgrades enum.

    Returns:
        set[str]: All lowercase upgrade names known to pysc2.

    Called by: build_entity_list()
    """
    return set(scan_pysc2_upgrades().keys())


def scan_parquet_upgrade_names(data_dir: Path) -> set[str]:
    """
    Collect every upgrade name observed in the dataset's upgrade columns.

    For each parquet file under data_dir, reads only the p1_upgrades and
    p2_upgrades columns (skipping the file silently if neither column is
    present) and unions every upgrade name from every cell that contains a
    list. Names are lowercased for consistency with the rest of the token
    schema.

    The upgrade columns are expected to follow the post-migration format:
    each cell is either None / NaN or a list[str] of cumulative upgrade
    names completed at or before that row's game_loop. Pre-migration files
    (legacy comma-string format) are also handled defensively -- a string
    cell is split on commas before being added.

    Args:
        data_dir: Root directory to search recursively for *.parquet files.

    Returns:
        set[str]: All lowercase upgrade names observed across the dataset.

    Called by: build_entity_list(), build_token_dictionary()
    """
    names: set[str] = set()
    parquet_files = list(data_dir.rglob("*.parquet"))

    for path in parquet_files:
        # Read only the column schema first to decide which upgrade columns
        # actually exist on this file -- older parquets may pre-date the
        # column entirely.
        try:
            schema = pq.read_schema(path)
        except Exception as exc:
            print(f"  [WARN] Could not read schema of {path.name}: {exc}", file=sys.stderr)
            continue

        cols_present = [c for c in ("p1_upgrades", "p2_upgrades") if c in schema.names]
        if not cols_present:
            continue

        try:
            df = pd.read_parquet(path, columns=cols_present)
        except Exception as exc:
            print(f"  [WARN] Could not read upgrades from {path.name}: {exc}", file=sys.stderr)
            continue

        for col in cols_present:
            for value in df[col].dropna():
                # Post-migration: list-like (numpy ndarray or list) of strings.
                if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, str):
                    for name in value:
                        if isinstance(name, str) and name:
                            names.add(name.lower())
                # Defensive: legacy comma-string format.
                elif isinstance(value, str):
                    for name in value.split(","):
                        name = name.strip()
                        if name:
                            names.add(name.lower())

    return names


# ---------------------------------------------------------------------------
# Upgrade race classification + deterministic-random ID fallback
# ---------------------------------------------------------------------------

# Build a lowercase-name -> race lookup from pysc2 upgrade names. The pysc2
# Upgrades enum is a single flat enum (not per-race), so race is inferred
# from the PascalCase name prefix where possible. Names that don't match a
# race prefix fall through to the keyword pass below.
def _build_upgrade_race_map() -> dict[str, str]:
    """
    Build a name -> race mapping for pysc2 upgrades.

    Race inference rules, applied in order:
      1. If the original PascalCase name starts with "Terran"/"Protoss"/"Zerg",
         use that prefix.
      2. Otherwise fall back to a small keyword pass against the lowercase
         name to catch race-implicit upgrades (e.g. "Burrow" -> zerg).
      3. If neither rule fires, classify as "unclassified" so the upgrade
         still appears in Entity_List.json without losing it.

    Returns:
        dict[str, str]: {lowercase_upgrade_name: "terran" | "protoss" | "zerg"
                        | "unclassified"}

    Called by: module-level _UPGRADE_RACE_MAP constant.
    """
    race_keywords = {
        "terran":  ("stimpack", "combatshield", "concussiveshells", "hisecautotracking",
                    "neosteelframe", "infernalpreigniter", "drillingclaws",
                    "smartservos", "weaponrefit", "personalcloaking", "hyperflightrotors",
                    "raveninterference", "ravenrecalibratedexplosives", "ravenenhancedmunitions",
                    "ravencorvidreactor", "yamatocannon", "behemothreactor",
                    "advancedballistics", "medivachighcapacityfueltanks",
                    "medivacincreasespeedboost", "magfieldlaunchers",
                    "shieldwall", "moebiusreactor", "cloakingfield", "tactical_jump",
                    "tacticaljump", "centrificalhooks"),
        "protoss": ("blink", "charge", "warpgateresearch", "psistormtech",
                    "extendedthermallance", "gravitonbeam", "gravitoncatapult",
                    "graviticbooster", "graviticdrive", "anionpulsecrystals",
                    "voidraysspeedupgrade", "voidrayspeedupgrade",
                    "fluxvanes", "tectonicdestabilizers", "interceptorlaunchspeedupgrade",
                    "interceptorgravitoncatapult", "phoenixrangeupgrade",
                    "darktemplarblinkupgrade", "shadowstrike",
                    "psionicstorm", "khaydarinamulet", "argusjewel"),
        "zerg":    ("burrow", "metabolicboost", "adrenalglands", "muscularaugments",
                    "glialreconstitution", "tunnelingclaws", "centrifugalhooks",
                    "pneumatizedcarapace", "ventralsacs", "groovedspines",
                    "pathogenglands", "neuralparasite", "chitinousplating",
                    "anabolicsynthesis", "adaptivetalons", "lurkerrange",
                    "lurkerspeed", "drilling_claws", "viperconsumeenergyupgrade"),
    }

    name_to_id = scan_pysc2_upgrades()
    race_map: dict[str, str] = {}

    for member in sc2_upgrades.Upgrades:
        lower = member.name.lower()

        # Rule 1: PascalCase prefix
        if member.name.startswith("Terran"):
            race_map[lower] = "terran"
        elif member.name.startswith("Protoss"):
            race_map[lower] = "protoss"
        elif member.name.startswith("Zerg"):
            race_map[lower] = "zerg"
        else:
            # Rule 2: keyword fallback
            classified = None
            for race, keywords in race_keywords.items():
                if lower in keywords:
                    classified = race
                    break
            race_map[lower] = classified if classified is not None else "unclassified"

    return race_map


_UPGRADE_RACE_MAP: dict[str, str] = _build_upgrade_race_map()


def _classify_upgrade(name: str) -> str:
    """
    Return the race for an upgrade name, or "unclassified" if unknown.

    Args:
        name: Lowercase upgrade name.

    Returns:
        "terran" | "protoss" | "zerg" | "unclassified".

    Called by: build_entity_list().
    """
    return _UPGRADE_RACE_MAP.get(name, "unclassified")


def _stable_random_upgrade_token_id(
    upgrade_name: str,
    taken: set[int],
    lo: int = UPGRADE_RANDOM_ID_LO,
    hi: int = UPGRADE_RANDOM_ID_HI,
) -> int:
    """
    Pick a deterministic-random token ID for an upgrade pysc2 doesn't know.

    The ID is drawn from the [lo, hi] range using an MD5 hash of the upgrade
    name as the starting offset, then linear-probed upward (wrapping within
    the range) until a free slot is found. This guarantees:
      - the ID lies in the upgrade-only window so it cannot collide with
        unit/building tokens (which are < 2000),
      - re-running this script with the same dataset always yields the same
        ID for the same upgrade (no Python hash randomisation involved),
      - collisions with already-assigned IDs are avoided even though the
        starting point is "random-looking".

    Args:
        upgrade_name: Lowercase upgrade name to derive the seed from.
        taken: Set of token IDs that are already claimed and must not be reused.
        lo: Inclusive lower bound for the random range.
        hi: Inclusive upper bound for the random range.

    Returns:
        int: A free token ID in [lo, hi].

    Raises:
        RuntimeError: If every ID in the range is already taken.

    Called by: build_token_dictionary().
    """
    span = hi - lo + 1
    digest = hashlib.md5(upgrade_name.encode("utf-8")).hexdigest()
    start_offset = int(digest, 16) % span

    for i in range(span):
        candidate = lo + ((start_offset + i) % span)
        if candidate not in taken:
            return candidate

    raise RuntimeError(
        f"No free upgrade token ID available in [{lo}, {hi}] -- range exhausted."
    )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify(name: str) -> tuple[str, str]:
    """
    Classify an entity name into (category, race).

    Args:
        name: Lowercase entity name string.

    Returns:
        (category, race):
          category -- "buildings" if name is in BUILDING_TYPES, else "units"
          race     -- "terran", "protoss", "zerg", or "unclassified"

    Called by: build_entity_list(), update_entity_list()
    """
    category = "buildings" if name in BUILDING_TYPES else "units"
    race = _RACE_MAP.get(name, "unclassified")
    return category, race


# ---------------------------------------------------------------------------
# JSON read / write
# ---------------------------------------------------------------------------

def _empty_entity_list() -> dict:
    """Return the skeleton structure for a new Entity_List.json."""
    return {
        "metadata": {
            "generated_at": "",
            "parquet_files_scanned": 0,
            "total_entities": 0,
        },
        "buildings": {"terran": [], "protoss": [], "zerg": [], "unclassified": []},
        "units":     {"terran": [], "protoss": [], "zerg": [], "unclassified": []},
        "upgrades":  {"terran": [], "protoss": [], "zerg": [], "unclassified": []},
    }


def _ensure_upgrades_section(data: dict) -> None:
    """
    Backfill the "upgrades" section on an Entity_List.json loaded from disk.

    Older versions of this script wrote Entity_List.json with only
    "buildings" and "units" keys. When a pre-existing file like that is
    loaded, this helper adds an empty "upgrades" section in place so the
    rest of the merge logic can treat all three categories uniformly.

    Args:
        data: The dict returned by load_entity_list().

    Called by: build_entity_list().
    """
    if "upgrades" not in data:
        data["upgrades"] = {
            "terran": [], "protoss": [], "zerg": [], "unclassified": [],
        }


def load_entity_list() -> dict:
    """
    Load Entity_List.json from disk.

    Returns:
        dict: Parsed JSON structure.  Returns a fresh empty structure if the
        file does not yet exist.

    Called by: update_entity_list(), and by external consumers that need the
    entity lists as Python data structures.
    """
    if ENTITY_LIST_PATH.exists():
        with open(ENTITY_LIST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return _empty_entity_list()


def _save_entity_list(data: dict) -> None:
    """
    Write entity list dict to Entity_List.json (pretty-printed, sorted keys).

    Args:
        data: The entity list dict to serialise.

    Called by: build_entity_list(), update_entity_list()
    """
    with open(ENTITY_LIST_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Token dictionary helpers
# ---------------------------------------------------------------------------

def _build_pysc2_id_to_name() -> dict[int, str]:
    """
    Build a game-engine-ID -> lowercase-string-name mapping from pysc2 enums.

    Each pysc2 enum member stores:
      member.name  -- the human-readable string the engine uses (e.g. "Marine")
      member.value -- the integer unit-type ID the game engine uses internally

    Reversing that gives us the token vocabulary: ID -> name.

    If two enum members share the same value (aliases exist in pysc2 for a few
    units), the last one wins -- this is acceptable because the string names
    for aliases are equivalent for tokenisation purposes.

    Returns:
        dict[int, str]: {game_engine_id: lowercase_name}

    Called by: build_token_dictionary()
    """
    id_to_name: dict[int, str] = {}
    for enum in (sc2_units.Terran, sc2_units.Protoss, sc2_units.Zerg):
        for member in enum:
            id_to_name[member.value] = member.name.lower()
    return id_to_name


def load_token_dictionary() -> dict:
    """
    Load Token_Dictionary.json from disk.

    Returns:
        dict: Parsed JSON structure.  Returns a fresh empty structure if the
        file does not yet exist.

    Called by: build_token_dictionary(), and by external consumers that need
    the flat token vocabulary.
    """
    if TOKEN_DICT_PATH.exists():
        with open(TOKEN_DICT_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"metadata": {"generated_at": "", "total_tokens": 0}, "tokens": {}}


def _save_token_dictionary(data: dict) -> None:
    """
    Write token dictionary dict to Token_Dictionary.json (pretty-printed).

    Tokens are sorted numerically by game-engine ID so the file is stable and
    easy to inspect.

    Args:
        data: The token dictionary dict to serialise.

    Called by: build_token_dictionary()
    """
    # Sort tokens numerically by game-engine ID before writing
    data["tokens"] = {
        str(k): data["tokens"][str(k)]
        for k in sorted(int(k) for k in data["tokens"])
    }
    with open(TOKEN_DICT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False, ensure_ascii=False)


def build_token_dictionary(
    dataset_entity_names: set[str],
    dataset_upgrade_names: set[str] | None = None,
) -> dict:
    """
    Build or update Token_Dictionary.json from pysc2 enums plus any unknowns
    observed in the dataset, including upgrade tokens.

    Three sources of token entries:
      1. Unit/building tokens from pysc2.lib.units enums -- the engine ID
         is used directly as the token ID (e.g. "48": "marine").
      2. Unknown unit/building tokens parsed out of "unknown(XXXX)" names
         in dataset_entity_names -- token ID == XXXX.
      3. Upgrade tokens. For each upgrade name (from pysc2.lib.upgrades AND
         from dataset_upgrade_names, unioned):
            - if pysc2 knows the name, token ID = engine_id + UPGRADE_TOKEN_ID_OFFSET
              (e.g. Burrow whose engine ID is 64 -> "100064": "burrow")
            - otherwise, a deterministic-random token ID is drawn from
              [UPGRADE_RANDOM_ID_LO, UPGRADE_RANDOM_ID_HI]; the seed is the
              upgrade name itself so re-runs produce the same ID, and the
              chosen ID is checked against the existing vocabulary to avoid
              collisions.

    The dictionary is append-only: any token ID already present in the
    on-disk file is left untouched so trained models retain a stable vocab.

    Args:
        dataset_entity_names: Set of unit/building entity name strings found
            in the dataset (output of scan_parquet_entities). Expected to
            contain any "unknown(XXXX)" strings produced by the extractor.
        dataset_upgrade_names: Set of lowercase upgrade names observed in
            the dataset's p1_upgrades / p2_upgrades columns (output of
            scan_parquet_upgrade_names). May be None or empty if the caller
            does not want to seed the upgrade vocabulary from the dataset.

    Returns:
        dict: The updated token dictionary structure (also written to disk).

    Called by: build_entity_list() (so both files are always updated together),
    and can be called standalone if only the token dictionary needs refreshing.
    """
    if dataset_upgrade_names is None:
        dataset_upgrade_names = set()

    # Step 1: reverse pysc2 unit/building enums
    pysc2_map = _build_pysc2_id_to_name()

    # Step 2: parse unknowns from the dataset scan
    unknown_map: dict[int, str] = {}
    for name in dataset_entity_names:
        m = _UNKNOWN_NAME_RE.match(name)
        if m:
            uid = int(m.group(1))
            unknown_map[uid] = f"unknown_{uid}"

    # Step 3: merge unit/building entries into existing dictionary (append-only)
    existing = load_token_dictionary()
    tokens: dict[str, str] = existing["tokens"]

    added: list[tuple[int, str]] = []

    for uid, name in pysc2_map.items():
        key = str(uid)
        if key not in tokens:
            tokens[key] = name
            added.append((uid, name))

    for uid, name in unknown_map.items():
        key = str(uid)
        if key not in tokens:
            tokens[key] = name
            added.append((uid, name))

    # Step 4: build the union of upgrade names from pysc2 + dataset and add
    # them with the +UPGRADE_TOKEN_ID_OFFSET scheme. pysc2-known names get
    # deterministic IDs (engine_id + offset); names pysc2 does not recognise
    # get a deterministic-random ID drawn from [LO, HI].
    pysc2_upgrade_ids: dict[str, int] = scan_pysc2_upgrades()
    all_upgrade_names: set[str] = set(pysc2_upgrade_ids) | set(dataset_upgrade_names)

    # Track which token IDs are already taken so the random fallback can
    # avoid collisions with both pre-existing entries and freshly-assigned
    # pysc2-derived IDs added in this same run.
    taken_ids: set[int] = {int(k) for k in tokens}

    # Track names already represented in the dictionary so we don't re-add
    # an upgrade that's already there under some prior ID.
    name_to_token_id: dict[str, int] = {v: int(k) for k, v in tokens.items()}

    upgrades_added: list[tuple[int, str]] = []

    # Process pysc2-known upgrades first so their offset IDs are claimed
    # before we fall back to random IDs for unknowns.
    for name in sorted(pysc2_upgrade_ids):
        if name in name_to_token_id:
            continue  # already in the vocab under a stable ID -- leave alone

        token_id = pysc2_upgrade_ids[name] + UPGRADE_TOKEN_ID_OFFSET
        key = str(token_id)
        if key in tokens:
            # Extremely rare: the offset ID is somehow already used (e.g. the
            # file was hand-edited). Fall back to a random unused ID rather
            # than overwrite.
            token_id = _stable_random_upgrade_token_id(name, taken_ids)
            key = str(token_id)

        tokens[key] = name
        taken_ids.add(token_id)
        name_to_token_id[name] = token_id
        upgrades_added.append((token_id, name))

    # Then process dataset-only upgrades pysc2 does not know about.
    dataset_only = sorted(set(dataset_upgrade_names) - set(pysc2_upgrade_ids))
    for name in dataset_only:
        if name in name_to_token_id:
            continue

        token_id = _stable_random_upgrade_token_id(name, taken_ids)
        key = str(token_id)
        tokens[key] = name
        taken_ids.add(token_id)
        name_to_token_id[name] = token_id
        upgrades_added.append((token_id, name))

    # Step 5: update metadata and write
    existing["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    existing["metadata"]["total_tokens"]  = len(tokens)
    existing["tokens"] = tokens

    _save_token_dictionary(existing)

    total_added = added + upgrades_added
    if total_added:
        if added:
            print(f"\n  Added {len(added)} new unit/building tokens "
                  f"to Token_Dictionary.json:")
            for uid, name in sorted(added):
                print(f"    + {uid}: \"{name}\"")
        if upgrades_added:
            print(f"\n  Added {len(upgrades_added)} new upgrade tokens "
                  f"to Token_Dictionary.json:")
            for uid, name in sorted(upgrades_added):
                pysc2_known = name in pysc2_upgrade_ids
                tag = "pysc2" if pysc2_known else "random"
                print(f"    + {uid}: \"{name}\"  ({tag})")
    else:
        print("\n  No new tokens found -- Token_Dictionary.json is already up to date.")

    print(f"\nToken_Dictionary.json written to: {TOKEN_DICT_PATH}")
    return existing


# ---------------------------------------------------------------------------
# Core build / update logic
# ---------------------------------------------------------------------------

def build_entity_list(data_dir: Path = _DEFAULT_DATA_DIR) -> dict:
    """
    Derive the full entity list from the dataset and pysc2 enums, then write
    Entity_List.json.

    If Entity_List.json already exists this function merges new findings into
    it without removing any existing entries (append-only to keep token IDs
    stable).  Call this function directly when you want a fresh or updated list.

    Steps:
      1. Scan all parquet files under data_dir for observed entity names.
      2. Scan pysc2 enums for all known entity names.
      3. Union both sets.
      4. Classify each name by category (building/unit) and race.
      5. Merge into existing Entity_List.json (or create fresh if absent).
      6. Write updated JSON to disk.

    Args:
        data_dir: Root directory to search for *.parquet files.
                  Defaults to <project_root>/data.

    Returns:
        dict: The updated entity list structure (also written to disk).

    Called by: __main__ block, and can be called programmatically by other
    ML pipeline scripts to ensure the list is current before consuming it.
    """
    print(f"Scanning dataset: {data_dir}")
    dataset_names, file_count = scan_parquet_entities(data_dir)
    print(f"  {file_count} parquet files scanned, {len(dataset_names)} entity types observed in data")

    pysc2_names = scan_pysc2_entities()
    print(f"  {len(pysc2_names)} entity types known to pysc2 enums")

    all_names = dataset_names | pysc2_names
    print(f"  {len(all_names)} total unique entity names (union)")

    # Upgrade scans: union of dataset-observed upgrade names and pysc2's
    # full Upgrades enum. Both feed the Entity_List "upgrades" section and
    # the Token_Dictionary upgrade tokens.
    print("Scanning upgrades:")
    dataset_upgrade_names = scan_parquet_upgrade_names(data_dir)
    print(f"  {len(dataset_upgrade_names)} upgrade names observed in dataset")

    pysc2_upgrade_names = scan_pysc2_upgrade_names()
    print(f"  {len(pysc2_upgrade_names)} upgrade names known to pysc2 enums")

    all_upgrade_names = dataset_upgrade_names | pysc2_upgrade_names
    print(f"  {len(all_upgrade_names)} total unique upgrade names (union)")

    # Load existing JSON to preserve any entries already there
    existing = load_entity_list()
    _ensure_upgrades_section(existing)

    # Collect existing names per bucket to detect what's new
    existing_names: set[str] = set()
    for cat in ("buildings", "units"):
        for race_list in existing[cat].values():
            existing_names.update(race_list)

    existing_upgrade_names: set[str] = set()
    for race_list in existing["upgrades"].values():
        existing_upgrade_names.update(race_list)

    # Insert new entities into the correct bucket
    added: list[str] = []
    for name in all_names:
        if name in existing_names:
            continue
        category, race = _classify(name)
        existing[category][race].append(name)
        added.append(name)

    # Insert new upgrades into the upgrades bucket, race-classified.
    upgrades_added: list[str] = []
    for name in all_upgrade_names:
        if name in existing_upgrade_names:
            continue
        race = _classify_upgrade(name)
        existing["upgrades"][race].append(name)
        upgrades_added.append(name)

    # Sort each bucket for readability / stable diffs
    for cat in ("buildings", "units", "upgrades"):
        for race in existing[cat]:
            existing[cat][race].sort()

    # Update metadata
    total = sum(
        len(lst)
        for cat in ("buildings", "units", "upgrades")
        for lst in existing[cat].values()
    )
    existing["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    existing["metadata"]["parquet_files_scanned"] = file_count
    existing["metadata"]["total_entities"] = total

    _save_entity_list(existing)

    if added:
        print(f"\n  Added {len(added)} new entities:")
        for name in sorted(added):
            category, race = _classify(name)
            print(f"    + {name}  ({category} / {race})")
    else:
        print("\n  No new entities found -- Entity_List.json is already up to date.")

    if upgrades_added:
        print(f"\n  Added {len(upgrades_added)} new upgrades to Entity_List.json:")
        for name in sorted(upgrades_added):
            race = _classify_upgrade(name)
            print(f"    + {name}  (upgrades / {race})")
    else:
        print("\n  No new upgrades found -- Entity_List.json upgrades section is up to date.")

    print(f"\nEntity_List.json written to: {ENTITY_LIST_PATH}")

    # Also keep Token_Dictionary.json in sync. Pass both the raw dataset
    # entity names (for unknown(XXXX) harvesting) and the upgrade names
    # (for the upgrade-token offset scheme).
    print("\n--- Token Dictionary ---")
    build_token_dictionary(dataset_names, all_upgrade_names)

    return existing


# ---------------------------------------------------------------------------
# Convenience accessors (for use by other ML pipeline modules)
# ---------------------------------------------------------------------------

def get_master_unit_frozenset() -> frozenset:
    """
    Load Entity_List.json and return all unit names as a single frozenset.

    Returns:
        frozenset[str]: Every unit name across all races.

    Called by: downstream ML pipeline modules that need a flat unit vocabulary.
    """
    data = load_entity_list()
    names: set[str] = set()
    for race_list in data["units"].values():
        names.update(race_list)
    return frozenset(names)


def get_master_building_frozenset() -> frozenset:
    """
    Load Entity_List.json and return all building names as a single frozenset.

    Returns:
        frozenset[str]: Every building name across all races.

    Called by: downstream ML pipeline modules that need a flat building vocabulary.
    """
    data = load_entity_list()
    names: set[str] = set()
    for race_list in data["buildings"].values():
        names.update(race_list)
    return frozenset(names)


def get_master_upgrade_frozenset() -> frozenset:
    """
    Load Entity_List.json and return all upgrade names as a single frozenset.

    Returns:
        frozenset[str]: Every upgrade name across all races. Empty frozenset
        if the on-disk Entity_List.json predates the upgrades section.

    Called by: downstream ML pipeline modules that need a flat upgrade vocabulary.
    """
    data = load_entity_list()
    names: set[str] = set()
    for race_list in data.get("upgrades", {}).values():
        names.update(race_list)
    return frozenset(names)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or update Entity_List.json from the SC2 dataset and pysc2 enums."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Root directory to search for *.parquet files (default: {_DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    result = build_entity_list(data_dir=args.data_dir)
    token_dict = load_token_dictionary()

    print("\n--- Summary ---")
    print(f"  Buildings  terran       : {len(result['buildings']['terran'])}")
    print(f"  Buildings  protoss      : {len(result['buildings']['protoss'])}")
    print(f"  Buildings  zerg         : {len(result['buildings']['zerg'])}")
    print(f"  Buildings  unclassified : {len(result['buildings']['unclassified'])}")
    print(f"  Units      terran       : {len(result['units']['terran'])}")
    print(f"  Units      protoss      : {len(result['units']['protoss'])}")
    print(f"  Units      zerg         : {len(result['units']['zerg'])}")
    print(f"  Units      unclassified : {len(result['units']['unclassified'])}")
    print(f"  Upgrades   terran       : {len(result['upgrades']['terran'])}")
    print(f"  Upgrades   protoss      : {len(result['upgrades']['protoss'])}")
    print(f"  Upgrades   zerg         : {len(result['upgrades']['zerg'])}")
    print(f"  Upgrades   unclassified : {len(result['upgrades']['unclassified'])}")
    print(f"  Total entities          : {result['metadata']['total_entities']}")
    print(f"  Token vocab size        : {token_dict['metadata']['total_tokens']}")
