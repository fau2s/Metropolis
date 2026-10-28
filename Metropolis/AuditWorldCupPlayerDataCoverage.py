"""
World Cup player-data coverage audit
====================================
Audits the men's FIFA World Cup player backbone in the Fjelstul World Cup Database
for 1930-2022 before adding player features to the prediction pipeline.

Outputs
-------
player_data_coverage_by_tournament.csv
player_data_coverage_overall.csv
player_data_source_capabilities.csv
player_data_match_lineup_diagnostics.csv

Important source facts
----------------------
* squads.csv: one row per player/team/tournament; includes position and shirt number.
* players.csv: player master table; includes birth_date.
* player_appearances.csv: match appearances and starter/substitute indicators, documented
  from 1970 onward.
* Fjelstul does NOT provide current club affiliation or national-team caps in these
  player tables. Those require an external source and should not be treated as
  ordinary missing values here.

The script is intentionally separate from the frozen match/coach enrichment pipeline.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests


BASE = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
URLS = {
    "players": f"{BASE}/players.csv",
    "squads": f"{BASE}/squads.csv",
    "player_appearances": f"{BASE}/player_appearances.csv",
    "matches": f"{BASE}/matches.csv",
}

OUTPUT_DIR = Path(r"D:\PhD\Metropolis")
CACHE_DIR = OUTPUT_DIR / "fjelstul_player_audit_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "WorldCupPlayerCoverageAudit/1.0 "
    "(academic research; contact via local project)"
)


def download_csv(name: str, url: str, retries: int = 6) -> pd.DataFrame:
    """Download with retries and persistent local cache."""
    cache_file = CACHE_DIR / f"{name}.csv"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
        "Connection": "close",
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {name}: {url}")
            r = requests.get(url, headers=headers, timeout=(15, 120))
            r.raise_for_status()
            cache_file.write_bytes(r.content)
            return pd.read_csv(io.BytesIO(r.content), low_memory=False)
        except Exception as exc:
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))

    if cache_file.exists():
        print(f"Using cached copy: {cache_file}")
        return pd.read_csv(cache_file, low_memory=False)

    raise RuntimeError(f"Could not download {name} and no cache exists.")


def tournament_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    ).astype("Int64")


def is_mens_world_cup_name(series: pd.Series) -> pd.Series:
    """
    Current Fjelstul naming is e.g. '1930 FIFA Men's World Cup'.
    Filter explicitly to men's tournaments.
    """
    s = series.astype(str)
    return s.str.contains("Men's World Cup", case=False, na=False)


def pct(num, den):
    if den == 0:
        return pd.NA
    return round(100.0 * num / den, 2)


def nonempty(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return series.notna() & s.ne("") & ~s.str.lower().isin(
        {"nan", "none", "not applicable", "na", "n/a"}
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    players = download_csv("players", URLS["players"])
    squads = download_csv("squads", URLS["squads"])
    apps = download_csv("player_appearances", URLS["player_appearances"])
    matches = download_csv("matches", URLS["matches"])

    print("\nLoaded columns:")
    print("players:", list(players.columns))
    print("squads:", list(squads.columns))
    print("player_appearances:", list(apps.columns))
    print("matches:", list(matches.columns))

    # ------------------------------------------------------------------
    # Filter men's World Cups 1930-2022
    # ------------------------------------------------------------------
    squads["_year"] = tournament_year(squads["tournament_name"])
    apps["_year"] = tournament_year(apps["tournament_name"])
    matches["_year"] = tournament_year(matches["tournament_name"])

    squads_m = squads[
        is_mens_world_cup_name(squads["tournament_name"])
        & squads["_year"].between(1930, 2022)
    ].copy()

    apps_m = apps[
        is_mens_world_cup_name(apps["tournament_name"])
        & apps["_year"].between(1930, 2022)
    ].copy()

    matches_m = matches[
        is_mens_world_cup_name(matches["tournament_name"])
        & matches["_year"].between(1930, 2022)
    ].copy()

    # ------------------------------------------------------------------
    # Birth date / age-capability coverage
    # ------------------------------------------------------------------
    if "birth_date" not in players.columns:
        raise RuntimeError("players.csv does not contain birth_date.")

    player_birth = players[["player_id", "birth_date"]].copy()
    player_birth["birth_date"] = pd.to_datetime(
        player_birth["birth_date"], errors="coerce"
    )

    squad = squads_m.merge(
        player_birth,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    squad["birth_date_known"] = squad["birth_date"].notna()

    if "position_code" in squad.columns:
        squad["position_known"] = nonempty(squad["position_code"])
    elif "position_name" in squad.columns:
        squad["position_known"] = nonempty(squad["position_name"])
    else:
        squad["position_known"] = False

    if "shirt_number" in squad.columns:
        shirt_num = pd.to_numeric(squad["shirt_number"], errors="coerce")
        # Historical 0 values represent unavailable/not assigned, not a usable number.
        squad["shirt_number_known"] = shirt_num.gt(0)
    else:
        squad["shirt_number_known"] = False

    # ------------------------------------------------------------------
    # Match-team starting-XI diagnostics.
    # Fjelstul explicitly documents player_appearances only from 1970.
    # ------------------------------------------------------------------
    if "starter" in apps_m.columns:
        apps_m["starter_num"] = pd.to_numeric(
            apps_m["starter"], errors="coerce"
        ).fillna(0)
    else:
        apps_m["starter_num"] = 0

    if "substitute" in apps_m.columns:
        apps_m["substitute_num"] = pd.to_numeric(
            apps_m["substitute"], errors="coerce"
        ).fillna(0)
    else:
        apps_m["substitute_num"] = 0

    if not apps_m.empty:
        team_match = (
            apps_m.groupby(
                ["_year", "tournament_id", "match_id", "team_id"],
                dropna=False,
            )
            .agg(
                appearance_rows=("player_id", "nunique"),
                starters=("starter_num", "sum"),
                substitutes_used=("substitute_num", "sum"),
            )
            .reset_index()
        )
        team_match["has_exactly_11_starters"] = team_match["starters"].eq(11)
    else:
        team_match = pd.DataFrame(
            columns=[
                "_year", "tournament_id", "match_id", "team_id",
                "appearance_rows", "starters", "substitutes_used",
                "has_exactly_11_starters",
            ]
        )

    # Expected team-match observations are 2 per match.
    expected_by_year = (
        matches_m.groupby("_year")["match_id"]
        .nunique()
        .rename("matches")
        .reset_index()
    )
    expected_by_year["expected_team_matches"] = (
        expected_by_year["matches"] * 2
    )

    lineup_by_year = (
        team_match.groupby("_year", dropna=False)
        .agg(
            observed_team_matches=("match_id", "size"),
            team_matches_with_11_starters=(
                "has_exactly_11_starters", "sum"
            ),
            mean_starters=("starters", "mean"),
            min_starters=("starters", "min"),
            max_starters=("starters", "max"),
        )
        .reset_index()
    )

    lineup_by_year = expected_by_year.merge(
        lineup_by_year, on="_year", how="left"
    )
    for c in [
        "observed_team_matches",
        "team_matches_with_11_starters",
    ]:
        lineup_by_year[c] = lineup_by_year[c].fillna(0).astype(int)

    lineup_by_year["appearance_team_match_coverage_pct"] = (
        100
        * lineup_by_year["observed_team_matches"]
        / lineup_by_year["expected_team_matches"]
    ).round(2)

    lineup_by_year["valid_starting_xi_coverage_pct"] = (
        100
        * lineup_by_year["team_matches_with_11_starters"]
        / lineup_by_year["expected_team_matches"]
    ).round(2)

    # Explicitly label pre-1970 as source-unavailable, not just missing.
    lineup_by_year["lineup_source_scope"] = lineup_by_year["_year"].map(
        lambda y: (
            "not_available_in_Fjelstul_player_appearances"
            if int(y) < 1970
            else "documented_from_1970"
        )
    )

    # ------------------------------------------------------------------
    # Tournament-level squad coverage
    # ------------------------------------------------------------------
    rows = []

    for year, g in squad.groupby("_year", sort=True):
        teams = int(g["team_id"].nunique())
        squad_rows = int(len(g))
        unique_players = int(g["player_id"].nunique())

        dob_known = int(g["birth_date_known"].sum())
        pos_known = int(g["position_known"].sum())
        shirt_known = int(g["shirt_number_known"].sum())

        tm = team_match[team_match["_year"].eq(year)]
        expected_tm_row = lineup_by_year[
            lineup_by_year["_year"].eq(year)
        ]

        if expected_tm_row.empty:
            expected_tm = 0
            observed_tm = 0
            valid_xi_tm = 0
            app_cov = pd.NA
            xi_cov = pd.NA
        else:
            er = expected_tm_row.iloc[0]
            expected_tm = int(er["expected_team_matches"])
            observed_tm = int(er["observed_team_matches"])
            valid_xi_tm = int(er["team_matches_with_11_starters"])
            app_cov = er["appearance_team_match_coverage_pct"]
            xi_cov = er["valid_starting_xi_coverage_pct"]

        rows.append(
            {
                "year": int(year),
                "teams": teams,
                "squad_rows": squad_rows,
                "unique_players": unique_players,
                "birth_date_known": dob_known,
                "birth_date_coverage_pct": pct(dob_known, squad_rows),
                "position_known": pos_known,
                "position_coverage_pct": pct(pos_known, squad_rows),
                "shirt_number_known": shirt_known,
                "shirt_number_coverage_pct": pct(shirt_known, squad_rows),
                "expected_team_matches": expected_tm,
                "appearance_team_matches_observed": observed_tm,
                "appearance_team_match_coverage_pct": app_cov,
                "team_matches_with_exactly_11_starters": valid_xi_tm,
                "valid_starting_xi_coverage_pct": xi_cov,
                "player_appearance_source_expected": int(year) >= 1970,
                # These are not columns in the Fjelstul player/squad tables.
                "club_affiliation_available_in_fjelstul": False,
                "national_caps_available_in_fjelstul": False,
            }
        )

    coverage = pd.DataFrame(rows).sort_values("year")

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    overall = pd.DataFrame(
        [
            {
                "metric": "mens_world_cup_tournaments_1930_2022",
                "value": int(coverage["year"].nunique()),
            },
            {
                "metric": "squad_rows",
                "value": int(len(squad)),
            },
            {
                "metric": "birth_date_nonmissing",
                "value": int(squad["birth_date_known"].sum()),
            },
            {
                "metric": "birth_date_coverage_pct",
                "value": round(
                    100 * squad["birth_date_known"].mean(), 2
                ),
            },
            {
                "metric": "position_coverage_pct",
                "value": round(
                    100 * squad["position_known"].mean(), 2
                ),
            },
            {
                "metric": "shirt_number_coverage_pct",
                "value": round(
                    100 * squad["shirt_number_known"].mean(), 2
                ),
            },
            {
                "metric": "player_appearances_documented_from_year",
                "value": 1970,
            },
            {
                "metric": "club_affiliation_in_fjelstul",
                "value": 0,
            },
            {
                "metric": "national_caps_in_fjelstul",
                "value": 0,
            },
        ]
    )

    capabilities = pd.DataFrame(
        [
            {
                "desired_feature": "squad membership",
                "fjelstul_source": "squads.csv",
                "available": True,
                "usable_period": "1930-2022",
                "comment": "Direct tournament squad membership.",
            },
            {
                "desired_feature": "date of birth / match age",
                "fjelstul_source": "players.csv + match date",
                "available": True,
                "usable_period": "coverage-dependent; audited by year",
                "comment": "Age should be computed at the prediction date/match date.",
            },
            {
                "desired_feature": "position",
                "fjelstul_source": "squads.csv",
                "available": True,
                "usable_period": "1930-2022",
                "comment": "Position name/code included.",
            },
            {
                "desired_feature": "starting XI",
                "fjelstul_source": "player_appearances.csv",
                "available": True,
                "usable_period": "1970-2022",
                "comment": (
                    "Fjelstul documents player appearances from 1970; "
                    "starter and substitute flags are included."
                ),
            },
            {
                "desired_feature": "prior World Cup player appearances",
                "fjelstul_source": "player_appearances.csv",
                "available": True,
                "usable_period": "1970-2022 directly",
                "comment": (
                    "Can be derived chronologically with match_date < current match."
                ),
            },
            {
                "desired_feature": "current club",
                "fjelstul_source": "",
                "available": False,
                "usable_period": "",
                "comment": "Requires external historical squad/club source.",
            },
            {
                "desired_feature": "national-team caps",
                "fjelstul_source": "",
                "available": False,
                "usable_period": "",
                "comment": (
                    "Requires external source or reconstruction from international "
                    "lineup/appearance data."
                ),
            },
            {
                "desired_feature": "club strength / club Elo",
                "fjelstul_source": "",
                "available": False,
                "usable_period": "",
                "comment": (
                    "Requires club affiliation first, then a date-specific external "
                    "club-strength source."
                ),
            },
        ]
    )

    # ------------------------------------------------------------------
    # Suggested cutoff diagnostics
    # ------------------------------------------------------------------
    coverage["rich_backbone_candidate"] = (
        coverage["birth_date_coverage_pct"].ge(95)
        & coverage["valid_starting_xi_coverage_pct"].ge(95)
        & coverage["player_appearance_source_expected"]
    )

    # Find earliest tournament from which every subsequent tournament satisfies
    # the Fjelstul-only backbone criteria.
    candidate_year = None
    years = coverage["year"].tolist()
    for y in years:
        tail = coverage[coverage["year"].ge(y)]
        if (
            not tail.empty
            and tail["rich_backbone_candidate"].fillna(False).all()
        ):
            candidate_year = int(y)
            break

    print("\nCoverage by tournament:")
    print(
        coverage[
            [
                "year",
                "squad_rows",
                "birth_date_coverage_pct",
                "position_coverage_pct",
                "shirt_number_coverage_pct",
                "appearance_team_match_coverage_pct",
                "valid_starting_xi_coverage_pct",
            ]
        ].to_string(index=False)
    )

    print("\nImportant source limitations:")
    print("  - current club: NOT in Fjelstul player/squad tables")
    print("  - national-team caps: NOT in Fjelstul player/squad tables")
    print("  - player appearances / starters: documented from 1970 onward")

    if candidate_year is None:
        print(
            "\nNo single cutoff meets >=95% DOB and >=95% valid starting-XI "
            "coverage for every subsequent tournament."
        )
    else:
        print(
            f"\nEarliest Fjelstul-only rich-backbone candidate cutoff: "
            f"{candidate_year}"
        )
        print(
            "(This does NOT yet consider club or caps coverage; those require "
            "external sources.)"
        )

    out_coverage = OUTPUT_DIR / "player_data_coverage_by_tournament.csv"
    out_overall = OUTPUT_DIR / "player_data_coverage_overall.csv"
    out_caps = OUTPUT_DIR / "player_data_source_capabilities.csv"
    out_lineups = OUTPUT_DIR / "player_data_match_lineup_diagnostics.csv"

    coverage.to_csv(out_coverage, index=False)
    overall.to_csv(out_overall, index=False)
    capabilities.to_csv(out_caps, index=False)
    lineup_by_year.to_csv(out_lineups, index=False)

    print("\nCreated:")
    print(out_coverage)
    print(out_overall)
    print(out_caps)
    print(out_lineups)


if __name__ == "__main__":
    main()
