import os
from collections import defaultdict
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


TOKEN_URL = "https://us.battle.net/oauth/token"
API_BASE = "https://us.api.blizzard.com"
NAMESPACE = "s2-client-replays"

PAGE_SIZE = 100


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"

    value = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{num_bytes:,} B"


def version_key(version: str):
    """
    Sort versions like:
        3.19.1
        4.0.1
        4.10.4
        5.0.2
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return (999999, version)


# ---------------------------------------------------------------------
# Blizzard authentication/API
# ---------------------------------------------------------------------

def get_access_token(session: requests.Session) -> str:
    response = session.post(
        TOKEN_URL,
        params={"grant_type": "client_credentials"},
        auth=(os.environ["BLIZZARD_CLIENT_ID"], os.environ["BLIZZARD_CLIENT_SECRET"]),
        timeout=30,
    )
    response.raise_for_status()

    return response.json()["access_token"]


def api_get(
    session: requests.Session,
    token: str,
    path_or_url: str,
    params: dict | None = None,
) -> dict:

    if path_or_url.startswith("http"):
        url = path_or_url
    else:
        url = urljoin(API_BASE, path_or_url)

    params = dict(params or {})
    params["namespace"] = NAMESPACE

    response = session.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
        },
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def get_archive_base_url(
    session: requests.Session,
    token: str,
) -> str:

    result = api_get(
        session,
        token,
        "/data/sc2/archive_url/base_url",
    )

    return result["base_url"]


# ---------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------

def discover_versions(
    session: requests.Session,
    token: str,
) -> list[str]:
    """
    Use the globally capped archive search only to discover which
    client_version values currently appear in Blizzard's replay index.

    We do NOT use this global query to count packs.
    """

    versions = set()
    page = 1

    print("Discovering replay versions from Blizzard's archive index...")

    while True:
        data = api_get(
            session,
            token,
            "/data/sc2/search/archive",
            params={
                "_pageSize": PAGE_SIZE,
                "_page": page,
            },
        )

        results = data.get("results", [])
        page_count = data.get("pageCount", 0)

        for result in results:
            version = (
                result
                .get("data", {})
                .get("client_version")
            )

            if version:
                versions.add(version)

        print(
            f"\rGlobal discovery: "
            f"page {page}/{page_count} | "
            f"{len(versions)} unique versions found",
            end="",
            flush=True,
        )

        if page >= page_count:
            break

        page += 1

    print()

    versions = sorted(versions, key=version_key)

    print(
        f"\nDiscovered {len(versions)} replay versions:\n"
    )

    print(", ".join(versions))
    print()

    return versions


# ---------------------------------------------------------------------
# Per-version archive enumeration
# ---------------------------------------------------------------------

def get_archives_for_version(
    session: requests.Session,
    token: str,
    version: str,
) -> list[dict]:
    """
    Query one client version at a time.

    This avoids the 1,000-result cap affecting the global search,
    provided the individual version itself has fewer than 1,000 packs.
    """

    results_all = []
    page = 1

    while True:
        data = api_get(
            session,
            token,
            "/data/sc2/search/archive",
            params={
                "client_version": version,
                "_pageSize": PAGE_SIZE,
                "_page": page,
            },
        )

        results = data.get("results", [])
        page_count = data.get("pageCount", 0)

        results_all.extend(results)

        if page >= page_count:
            break

        page += 1

    # Important warning:
    #
    # 10 pages * 100 results = exactly 1,000, which appears to be
    # Blizzard's search ceiling.
    if (
        len(results_all) >= 1000
        or (
            page_count == 10
            and len(results_all) == 1000
        )
    ):
        print(
            f"\nWARNING: {version} returned exactly "
            f"{len(results_all):,} packs across {page_count} pages."
        )
        print(
            "This version may itself be hitting Blizzard's "
            "1,000-result search limit."
        )

    return results_all


# ---------------------------------------------------------------------
# Pack metadata / size
# ---------------------------------------------------------------------

def get_archive_metadata(
    session: requests.Session,
    token: str,
    metadata_url: str,
) -> dict:

    return api_get(
        session,
        token,
        metadata_url,
    )


def get_remote_size(
    session: requests.Session,
    url: str,
) -> int | None:
    """
    Determine compressed archive size without downloading the ZIP.

    Try HEAD first. If necessary, fall back to a streamed GET while
    avoiding downloading the response body.
    """

    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=30,
        )

        if response.ok:
            length = response.headers.get("Content-Length")

            if length is not None:
                return int(length)

    except requests.RequestException:
        pass

    try:
        with session.get(
            url,
            stream=True,
            timeout=30,
        ) as response:

            response.raise_for_status()

            length = response.headers.get("Content-Length")

            if length is not None:
                return int(length)

    except requests.RequestException as exc:
        print(
            f"\nWARNING: Could not determine size for:\n"
            f"  {url}\n"
            f"  {exc}"
        )

    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    load_dotenv()
    with requests.Session() as session:

        # -------------------------------------------------------------
        # Authentication
        # -------------------------------------------------------------

        print("Authenticating with Blizzard...")

        token = get_access_token(session)

        print("Authentication successful.\n")

        # -------------------------------------------------------------
        # CDN
        # -------------------------------------------------------------

        print("Getting replay archive CDN base URL...")

        archive_base_url = get_archive_base_url(
            session,
            token,
        )

        print(
            f"Archive base URL: {archive_base_url}\n"
        )

        # -------------------------------------------------------------
        # Discover replay versions
        # -------------------------------------------------------------

        versions = discover_versions(
            session,
            token,
        )

        # -------------------------------------------------------------
        # Enumerate every version separately
        # -------------------------------------------------------------

        version_archives = {}

        print("=" * 72)
        print("ENUMERATING REPLAY PACKS BY VERSION")
        print("=" * 72)

        total_indexed_packs = 0

        for i, version in enumerate(versions, start=1):

            archives = get_archives_for_version(
                session,
                token,
                version,
            )

            version_archives[version] = archives
            total_indexed_packs += len(archives)

            print(
                f"[{i:>2}/{len(versions)}] "
                f"{version:<10} "
                f"{len(archives):>5,} packs"
            )

        print("-" * 72)

        print(
            f"Total indexed packs across version queries: "
            f"{total_indexed_packs:,}\n"
        )

        # -------------------------------------------------------------
        # De-duplicate
        # -------------------------------------------------------------

        #
        # This probably won't remove much, but it's cheap insurance.
        # We identify an archive by its metadata href.
        #

        unique_archives = {}

        for version, archives in version_archives.items():

            for archive in archives:

                href = archive["key"]["href"]

                unique_archives[href] = {
                    "version": version,
                    "result": archive,
                }

        duplicate_count = (
            total_indexed_packs
            - len(unique_archives)
        )

        print(
            f"Unique archive packs: "
            f"{len(unique_archives):,}"
        )

        print(
            f"Duplicate archive records removed: "
            f"{duplicate_count:,}\n"
        )

        # -------------------------------------------------------------
        # Inspect compressed archive sizes
        # -------------------------------------------------------------

        stats = defaultdict(
            lambda: {
                "packs": 0,
                "nonempty": 0,
                "empty": 0,
                "unknown": 0,
                "bytes": 0,
            }
        )

        total = len(unique_archives)

        print("=" * 72)
        print("INSPECTING COMPRESSED ARCHIVE SIZES")
        print("=" * 72)

        for i, (href, archive_info) in enumerate(
            unique_archives.items(),
            start=1,
        ):

            version = archive_info["version"]

            size = None

            try:
                metadata = get_archive_metadata(
                    session,
                    token,
                    href,
                )

                archive_path = metadata["path"]

                archive_url = urljoin(
                    archive_base_url,
                    archive_path,
                )

                size = get_remote_size(
                    session,
                    archive_url,
                )

            except Exception as exc:

                print(
                    f"\nWARNING: Failed to inspect "
                    f"{version} archive:\n"
                    f"  {exc}"
                )

            s = stats[version]

            s["packs"] += 1

            if size is None:
                s["unknown"] += 1

            # Blizzard's official downloader considers <=22 bytes
            # effectively an empty ZIP/archive.
            elif size <= 22:
                s["empty"] += 1

            else:
                s["nonempty"] += 1
                s["bytes"] += size

            print(
                f"\rPack {i:,}/{total:,} "
                f"| version {version:<8} "
                f"| {format_bytes(size):>12}",
                end="",
                flush=True,
            )

        print("\n")

        # -------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------

        print("=" * 82)
        print("BLIZZARD SC2 REPLAY ARCHIVE SUMMARY")
        print("=" * 82)

        print(
            f"{'Version':<12}"
            f"{'Packs':>8}"
            f"{'Nonempty':>12}"
            f"{'Empty':>8}"
            f"{'Unknown':>10}"
            f"{'Compressed Size':>18}"
        )

        print("-" * 82)

        totals = {
            "packs": 0,
            "nonempty": 0,
            "empty": 0,
            "unknown": 0,
            "bytes": 0,
        }

        for version in sorted(
            stats,
            key=version_key,
        ):

            s = stats[version]

            print(
                f"{version:<12}"
                f"{s['packs']:>8,}"
                f"{s['nonempty']:>12,}"
                f"{s['empty']:>8,}"
                f"{s['unknown']:>10,}"
                f"{format_bytes(s['bytes']):>18}"
            )

            for key in totals:
                totals[key] += s[key]

        print("-" * 82)

        print(
            f"{'TOTAL':<12}"
            f"{totals['packs']:>8,}"
            f"{totals['nonempty']:>12,}"
            f"{totals['empty']:>8,}"
            f"{totals['unknown']:>10,}"
            f"{format_bytes(totals['bytes']):>18}"
        )

        print("=" * 82)

        print(
            "\nCompressed size is the amount of storage required "
            "to download the usable ZIP archives."
        )

        if totals["unknown"]:
            print(
                f"\nNOTE: {totals['unknown']:,} archives had unknown "
                "sizes, so the real download total is slightly larger."
            )


if __name__ == "__main__":
    main()
