"""
PrototypeInternationalPlayerLayer.py
====================================

Feasibility prototype for a GENERAL men's international-football player layer.

Goal
----
Measure how much of the user's international-results dataset can be enriched
with modern player information without touching the frozen match/coach pipeline.

Primary source
--------------
dcaribou/transfermarkt-datasets DuckDB snapshot.

Important methodological rules
------------------------------
1. Player age is calculated at match date from DOB.
2. "Current" player fields are NEVER used as historical match features.
3. Current international_caps is NEVER used as caps-before-match; that would leak.
4. Club-at-match is reconstructed from transfer history using the latest transfer
   on/before the match date. This is a feasibility approximation and is audited.
5. "Observed prior international appearances" below counts only national-team
   matches present in this Transfermarkt dataset. Because this source is currently
   focused on major national-team competitions, it is a LOWER BOUND, not the
   production caps-before-match variable.
6. No current-match or future information is used in prior-appearance counts.

Outputs
-------
tm_international_game_coverage_by_year.csv
tm_player_lineup_coverage_by_year.csv
tm_results_overlap_by_year.csv
tm_player_fact_prototype.csv
tm_player_layer_feasibility_summary.csv
tm_international_competitions.csv

Default audit window: 2013-2024.
"""

from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(r"D:\PhD\Metropolis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "player_layer_prototype_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/"
    "data/transfermarkt-datasets.duckdb"
)
DB_PATH = CACHE_DIR / "transfermarkt-datasets.duckdb"

RESULTS_LOCAL = OUTPUT_DIR / "results.csv"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

START_YEAR = 2013
END_YEAR = 2024

USER_AGENT = (
    "InternationalPlayerLayerPrototype/1.0 "
    "(academic research feasibility audit)"
)


# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------

try:
    import duckdb
except ImportError:
    raise SystemExit(
        "\nMissing dependency: duckdb\n"
        "Install once with:\n\n"
        "    pip install duckdb\n\n"
        "Then run this script again.\n"
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


TEAM_ALIASES = {
    "united states": "usa",
    "united states of america": "usa",
    "u s a": "usa",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "cote d ivoire": "ivory coast",
    "czechia": "czech republic",
    "iran": "iran",
    "ir iran": "iran",
    "china pr": "china",
    "china": "china",
    "turkiye": "turkey",
    "russia": "russia",
    "germany": "germany",
}


def team_key(value) -> str:
    x = normalize_text(value)
    return TEAM_ALIASES.get(x, x)


def pct(num, den):
    if den in (0, None) or pd.isna(den):
        return pd.NA
    return round(100.0 * float(num) / float(den), 2)


def download_file(url: str, path: Path, retries: int = 6) -> Path:
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached database: {path}")
        return path

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Connection": "close",
    }

    tmp = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading Transfermarkt DuckDB ({attempt}/{retries})...")
            with requests.get(
                url,
                headers=headers,
                timeout=(20, 180),
                stream=True,
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
            print(f"Downloaded: {path}")
            return path
        except Exception as exc:
            print(f"  download failed: {exc}")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Could not download {url}")


def load_results() -> pd.DataFrame:
    if RESULTS_LOCAL.exists():
        print(f"Using local international results: {RESULTS_LOCAL}")
        df = pd.read_csv(RESULTS_LOCAL, low_memory=False)
    else:
        print(f"Local results.csv not found; downloading: {RESULTS_URL}")
        df = pd.read_csv(RESULTS_URL, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        df["date"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    df["_home"] = df["home_team"].map(team_key)
    df["_away"] = df["away_team"].map(team_key)
    df["_match_key"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["_home"]
        + "|"
        + df["_away"]
    )
    return df


def existing_tables(con):
    return set(
        con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).df()["table_name"].tolist()
    )


def first_existing_table(tables, candidates):
    for c in candidates:
        if c in tables:
            return c
    return None


def column_names(con, table):
    return set(
        con.execute(
            f"PRAGMA table_info('{table}')"
        ).df()["name"].tolist()
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    download_file(DB_URL, DB_PATH)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = existing_tables(con)

    games_t = first_existing_table(tables, ["games", "cur_games"])
    lineups_t = first_existing_table(
        tables, ["game_lineups", "cur_game_lineups"]
    )
    players_t = first_existing_table(tables, ["players", "cur_players"])
    transfers_t = first_existing_table(tables, ["transfers", "cur_transfers"])
    comps_t = first_existing_table(
        tables, ["competitions", "cur_competitions"]
    )
    nat_t = first_existing_table(
        tables, ["national_teams", "cur_national_teams"]
    )
    clubs_t = first_existing_table(tables, ["clubs", "cur_clubs"])

    required = {
        "games": games_t,
        "game_lineups": lineups_t,
        "players": players_t,
        "national_teams": nat_t,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            "Required tables not found in DuckDB: " + ", ".join(missing)
        )

    print("\nTables selected:")
    for k, v in {
        **required,
        "transfers": transfers_t,
        "competitions": comps_t,
        "clubs": clubs_t,
    }.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Identify national-team games.
    # National-team IDs use Transfermarkt entity IDs and can therefore be
    # joined to game home/away IDs.
    # ------------------------------------------------------------------

    national_games = con.execute(
        f"""
        SELECT
            g.game_id,
            CAST(g.date AS DATE) AS date,
            g.competition_id,
            g.competition_type,
            g.season,
            g.round,
            g.home_club_id,
            g.away_club_id,
            g.home_club_name,
            g.away_club_name,
            g.home_club_goals,
            g.away_club_goals
        FROM {games_t} g
        INNER JOIN {nat_t} nh
            ON g.home_club_id = nh.national_team_id
        INNER JOIN {nat_t} na
            ON g.away_club_id = na.national_team_id
        WHERE EXTRACT(YEAR FROM CAST(g.date AS DATE))
              BETWEEN {START_YEAR} AND {END_YEAR}
        ORDER BY g.date, g.game_id
        """
    ).df()

    if national_games.empty:
        raise RuntimeError(
            "No national-team games were identified. "
            "The current dataset may use a different national-team ID mapping."
        )

    national_games["date"] = pd.to_datetime(national_games["date"])
    national_games["year"] = national_games["date"].dt.year

    print(
        f"\nTransfermarkt national-team games "
        f"{START_YEAR}-{END_YEAR}: {len(national_games):,}"
    )

    # Competition inventory
    comp_inventory = (
        national_games.groupby(
            ["competition_id", "competition_type"], dropna=False
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            games=("game_id", "nunique"),
        )
        .reset_index()
        .sort_values(["games", "competition_id"], ascending=[False, True])
    )

    # ------------------------------------------------------------------
    # Match lineups + DOB
    # ------------------------------------------------------------------

    lineup_cols = column_names(con, lineups_t)
    player_cols = column_names(con, players_t)

    type_expr = "l.type" if "type" in lineup_cols else "NULL"
    position_expr = "l.position" if "position" in lineup_cols else "NULL"

    dob_expr = (
        "p.date_of_birth"
        if "date_of_birth" in player_cols
        else "NULL"
    )

    player_facts = con.execute(
        f"""
        SELECT
            g.game_id,
            CAST(g.date AS DATE) AS match_date,
            g.competition_id,
            g.home_club_id,
            g.away_club_id,
            g.home_club_name,
            g.away_club_name,
            l.club_id AS national_team_id,
            l.player_id,
            l.player_name,
            {type_expr} AS lineup_type,
            {position_expr} AS lineup_position,
            CAST({dob_expr} AS DATE) AS date_of_birth
        FROM {games_t} g
        INNER JOIN {nat_t} nh
            ON g.home_club_id = nh.national_team_id
        INNER JOIN {nat_t} na
            ON g.away_club_id = na.national_team_id
        INNER JOIN {lineups_t} l
            ON g.game_id = l.game_id
        LEFT JOIN {players_t} p
            ON l.player_id = p.player_id
        WHERE EXTRACT(YEAR FROM CAST(g.date AS DATE))
              BETWEEN {START_YEAR} AND {END_YEAR}
        """
    ).df()

    player_facts["match_date"] = pd.to_datetime(player_facts["match_date"])
    player_facts["date_of_birth"] = pd.to_datetime(
        player_facts["date_of_birth"], errors="coerce"
    )

    player_facts["age_at_match"] = (
        (
            player_facts["match_date"]
            - player_facts["date_of_birth"]
        ).dt.days
        / 365.2425
    )

    # Infer side/team name.
    player_facts["national_team_name"] = player_facts.apply(
        lambda r: (
            r["home_club_name"]
            if r["national_team_id"] == r["home_club_id"]
            else (
                r["away_club_name"]
                if r["national_team_id"] == r["away_club_id"]
                else pd.NA
            )
        ),
        axis=1,
    )

    # Normalize lineup type into starter/bench where possible.
    lt = player_facts["lineup_type"].astype("string").str.lower()
    player_facts["is_starter"] = lt.str.contains(
        "start", na=False
    )
    player_facts["is_bench"] = lt.str.contains(
        "bench|sub", regex=True, na=False
    )

    # ------------------------------------------------------------------
    # Leakage-safe prior appearances WITHIN THIS SOURCE ONLY.
    # This is NOT yet full official caps.
    # ------------------------------------------------------------------

    # One row per player per game regardless of duplicate lineup records.
    unique_apps = (
        player_facts[
            ["player_id", "game_id", "match_date"]
        ]
        .drop_duplicates()
        .sort_values(["player_id", "match_date", "game_id"])
    )

    unique_apps["observed_prior_intl_appearances"] = (
        unique_apps.groupby("player_id").cumcount()
    )

    player_facts = player_facts.merge(
        unique_apps,
        on=["player_id", "game_id", "match_date"],
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # Club-at-match feasibility from transfer history.
    # Latest transfer on/before match_date -> to_club.
    # DO NOT substitute players.current_club_id for historical rows.
    # ------------------------------------------------------------------

    player_facts["club_at_match_id"] = pd.NA
    player_facts["club_at_match_name"] = pd.NA
    player_facts["club_at_match_source"] = "unresolved"

    if transfers_t is not None:
        con.register(
            "prototype_player_matches",
            player_facts[
                ["game_id", "player_id", "match_date"]
            ].drop_duplicates()
        )

        club_at_match = con.execute(
            f"""
            WITH ranked AS (
                SELECT
                    pm.game_id,
                    pm.player_id,
                    pm.match_date,
                    t.to_club_id,
                    t.to_club_name,
                    t.transfer_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY pm.game_id, pm.player_id
                        ORDER BY t.transfer_date DESC
                    ) AS rn
                FROM prototype_player_matches pm
                LEFT JOIN {transfers_t} t
                    ON pm.player_id = t.player_id
                   AND CAST(t.transfer_date AS DATE) <= pm.match_date
            )
            SELECT
                game_id,
                player_id,
                match_date,
                to_club_id AS club_at_match_id,
                to_club_name AS club_at_match_name,
                transfer_date AS club_evidence_date
            FROM ranked
            WHERE rn = 1
            """
        ).df()

        club_at_match["match_date"] = pd.to_datetime(
            club_at_match["match_date"]
        )
        club_at_match["club_evidence_date"] = pd.to_datetime(
            club_at_match["club_evidence_date"], errors="coerce"
        )

        player_facts = player_facts.drop(
            columns=["club_at_match_id", "club_at_match_name"]
        ).merge(
            club_at_match,
            on=["game_id", "player_id", "match_date"],
            how="left",
            validate="many_to_one",
        )

        player_facts["club_at_match_source"] = player_facts[
            "club_at_match_id"
        ].notna().map(
            {True: "latest_transfer_on_or_before_match", False: "unresolved"}
        )

    # ------------------------------------------------------------------
    # Lineup coverage by match/year
    # ------------------------------------------------------------------

    match_team = (
        player_facts.groupby(
            ["game_id", "match_date", "national_team_id"],
            dropna=False,
        )
        .agg(
            lineup_players=("player_id", "nunique"),
            starters=("is_starter", "sum"),
            dob_known=("date_of_birth", lambda s: s.notna().sum()),
            club_at_match_known=(
                "club_at_match_id",
                lambda s: s.notna().sum(),
            ),
        )
        .reset_index()
    )

    match_team["year"] = match_team["match_date"].dt.year
    match_team["has_lineup"] = match_team["lineup_players"].gt(0)
    match_team["has_exact_11_starters"] = match_team["starters"].eq(11)

    lineup_by_year = (
        match_team.groupby("year")
        .agg(
            observed_team_matches=("game_id", "size"),
            team_matches_with_11_starters=(
                "has_exact_11_starters", "sum"
            ),
            mean_lineup_players=("lineup_players", "mean"),
            player_rows=("lineup_players", "sum"),
            dob_known=("dob_known", "sum"),
            club_at_match_known=("club_at_match_known", "sum"),
        )
        .reset_index()
    )

    games_by_year = (
        national_games.groupby("year")
        .agg(
            national_team_games=("game_id", "nunique")
        )
        .reset_index()
    )
    games_by_year["expected_team_matches"] = (
        games_by_year["national_team_games"] * 2
    )

    lineup_by_year = games_by_year.merge(
        lineup_by_year, on="year", how="left"
    )

    for c in [
        "observed_team_matches",
        "team_matches_with_11_starters",
        "player_rows",
        "dob_known",
        "club_at_match_known",
    ]:
        lineup_by_year[c] = lineup_by_year[c].fillna(0)

    lineup_by_year["team_match_lineup_coverage_pct"] = (
        100
        * lineup_by_year["observed_team_matches"]
        / lineup_by_year["expected_team_matches"]
    ).round(2)

    lineup_by_year["valid_xi_coverage_pct"] = (
        100
        * lineup_by_year["team_matches_with_11_starters"]
        / lineup_by_year["expected_team_matches"]
    ).round(2)

    lineup_by_year["dob_coverage_pct"] = [
        pct(n, d)
        for n, d in zip(
            lineup_by_year["dob_known"],
            lineup_by_year["player_rows"],
        )
    ]

    lineup_by_year["club_at_match_coverage_pct"] = [
        pct(n, d)
        for n, d in zip(
            lineup_by_year["club_at_match_known"],
            lineup_by_year["player_rows"],
        )
    ]

    # ------------------------------------------------------------------
    # Coverage against user's full results dataset
    # ------------------------------------------------------------------

    results = load_results()

    national_games["_home"] = national_games["home_club_name"].map(team_key)
    national_games["_away"] = national_games["away_club_name"].map(team_key)
    national_games["_match_key"] = (
        national_games["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + national_games["_home"]
        + "|"
        + national_games["_away"]
    )

    tm_keys = set(national_games["_match_key"])

    results["tm_match_found"] = results["_match_key"].isin(tm_keys)
    results["year"] = results["date"].dt.year

    overlap_by_year = (
        results.groupby("year")
        .agg(
            results_matches=("_match_key", "size"),
            transfermarkt_match_found=("tm_match_found", "sum"),
        )
        .reset_index()
    )
    overlap_by_year["match_overlap_pct"] = (
        100
        * overlap_by_year["transfermarkt_match_found"]
        / overlap_by_year["results_matches"]
    ).round(2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    total_results = int(len(results))
    found_results = int(results["tm_match_found"].sum())

    total_player_rows = int(len(player_facts))
    dob_rows = int(player_facts["date_of_birth"].notna().sum())
    club_rows = int(player_facts["club_at_match_id"].notna().sum())

    summary = pd.DataFrame(
        [
            {
                "metric": "audit_start_year",
                "value": START_YEAR,
                "interpretation": "Prototype window.",
            },
            {
                "metric": "audit_end_year",
                "value": END_YEAR,
                "interpretation": "Prototype window.",
            },
            {
                "metric": "transfermarkt_national_team_games",
                "value": len(national_games),
                "interpretation": (
                    "National-team games identified in the Transfermarkt dataset."
                ),
            },
            {
                "metric": "user_results_matches_in_window",
                "value": total_results,
                "interpretation": "All source international matches in same window.",
            },
            {
                "metric": "user_results_matched_to_transfermarkt",
                "value": found_results,
                "interpretation": "Strict date/home/away normalized join.",
            },
            {
                "metric": "user_results_match_overlap_pct",
                "value": pct(found_results, total_results),
                "interpretation": (
                    "This measures whether Transfermarkt can be the universal "
                    "international-match backbone. Expect partial coverage."
                ),
            },
            {
                "metric": "player_lineup_rows",
                "value": total_player_rows,
                "interpretation": "Rows in modern national-team lineups.",
            },
            {
                "metric": "player_dob_coverage_pct",
                "value": pct(dob_rows, total_player_rows),
                "interpretation": "DOB availability for age-at-match.",
            },
            {
                "metric": "club_at_match_transfer_reconstruction_pct",
                "value": pct(club_rows, total_player_rows),
                "interpretation": (
                    "Latest transfer <= match date. Feasibility approximation; "
                    "must be validated before production."
                ),
            },
            {
                "metric": "production_caps_before_match_available",
                "value": 0,
                "interpretation": (
                    "No. Current international_caps would leak; observed prior "
                    "appearances here cover only matches present in this dataset."
                ),
            },
            {
                "metric": "club_elo_ready",
                "value": 0,
                "interpretation": (
                    "Not yet. Add only after club-at-match identity is validated."
                ),
            },
        ]
    )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    out_games = OUTPUT_DIR / "tm_international_game_coverage_by_year.csv"
    out_lineups = OUTPUT_DIR / "tm_player_lineup_coverage_by_year.csv"
    out_overlap = OUTPUT_DIR / "tm_results_overlap_by_year.csv"
    out_facts = OUTPUT_DIR / "tm_player_fact_prototype.csv"
    out_summary = OUTPUT_DIR / "tm_player_layer_feasibility_summary.csv"
    out_comps = OUTPUT_DIR / "tm_international_competitions.csv"

    games_by_year.to_csv(out_games, index=False)
    lineup_by_year.to_csv(out_lineups, index=False)
    overlap_by_year.to_csv(out_overlap, index=False)
    player_facts.to_csv(out_facts, index=False)
    summary.to_csv(out_summary, index=False)
    comp_inventory.to_csv(out_comps, index=False)

    print("\n=== Feasibility summary ===")
    print(summary.to_string(index=False))

    print("\n=== Match overlap by year ===")
    print(overlap_by_year.to_string(index=False))

    print("\n=== Player coverage by year ===")
    print(
        lineup_by_year[
            [
                "year",
                "national_team_games",
                "team_match_lineup_coverage_pct",
                "valid_xi_coverage_pct",
                "dob_coverage_pct",
                "club_at_match_coverage_pct",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    for p in [
        out_games,
        out_lineups,
        out_overlap,
        out_facts,
        out_summary,
        out_comps,
    ]:
        print(p)

    print(
        "\nNOTE: observed_prior_intl_appearances is NOT production caps. "
        "It is only a leakage-safe lower-bound diagnostic within the "
        "Transfermarkt international matches available in this dataset."
    )


if __name__ == "__main__":
    main()
