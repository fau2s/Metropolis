"""
AuditOpenNationalSquadCoverage.py
=================================

Audits the CC0 footballcsv/cache.footballsquads repository for NATIONAL-TEAM
tournament squad coverage.

Why this source is useful
-------------------------
The repository mirrors historical national-team tournament squads into simple
text/CSV-like files and is licensed CC0-1.0. Typical squad files contain:

    Number, Name, Nat, Pos, Height, Weight, Date of Birth, Birth Place, Current Club

This gives us a reproducible open source for:
- player identity
- DOB -> age at tournament/match
- squad membership
- club at tournament time
- position

Important limitation
--------------------
This is NOT a universal source for all internationals. The national directory
is organised around major tournaments (World Cup, EURO, Copa America, etc.).
The purpose of this audit is to measure its exact usable scope and decide
where it can serve as a competition-specific player layer.

Outputs
-------
D:\\PhD\\Metropolis\\open_squad_coverage_by_edition.csv
D:\\PhD\\Metropolis\\open_squad_coverage_by_competition.csv
D:\\PhD\\Metropolis\\open_squad_field_coverage.csv
D:\\PhD\\Metropolis\\open_squad_team_editions.csv
D:\\PhD\\Metropolis\\open_squad_player_rows.csv
D:\\PhD\\Metropolis\\open_squad_audit_summary.csv

Dependencies
------------
pip install pandas requests
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIR = Path(r"D:\PhD\Metropolis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "open_squad_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_URL = "https://github.com/footballcsv/cache.footballsquads/archive/refs/heads/master.zip"
ZIP_PATH = CACHE_DIR / "cache.footballsquads-master.zip"
EXTRACT_DIR = CACHE_DIR / "repo"

USER_AGENT = "OpenNationalSquadCoverageAudit/1.0 (academic research)"

NATIONAL_COMPETITIONS = {
    "worldcup": "World Cup",
    "eurocham": "European Championship",
    "copaamer": "Copa America",
    "africacn": "Africa Cup of Nations",
    "confed": "Confederations Cup",
}


def download(url: str, path: Path, retries: int = 5) -> Path:
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached archive: {path}")
        return path

    headers = {"User-Agent": USER_AGENT}
    tmp = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading CC0 squad repository ({attempt}/{retries})...")
            with requests.get(url, headers=headers, timeout=(20, 180), stream=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
            return path
        except Exception as exc:
            print(f"  failed: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Could not download {url}")


def extract_repo(zip_path: Path) -> Path:
    marker = EXTRACT_DIR / ".complete"
    if marker.exists():
        roots = list(EXTRACT_DIR.glob("cache.footballsquads-*"))
        if roots:
            return roots[0]

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)

    roots = list(EXTRACT_DIR.glob("cache.footballsquads-*"))
    if not roots:
        raise RuntimeError("Could not locate extracted repository root.")

    marker.write_text("ok", encoding="utf-8")
    return roots[0]


def infer_year(*parts: str):
    text = " ".join(parts)
    years = [int(x) for x in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)]
    if years:
        return max(years)

    # folder patterns: wc2022, euro2020, ca2019, acn2017, etc.
    m = re.search(r"(?:wc|euro|ec|ca|acn|conf)[^\d]*(19\d{2}|20\d{2})", text, re.I)
    return int(m.group(1)) if m else None


def clean_header(x: str) -> str:
    x = re.sub(r"\s+", " ", x.strip())
    aliases = {
        "DOB": "Date of Birth",
        "Birth Date": "Date of Birth",
        "Club": "Current Club",
        "Team": "Current Club",
    }
    return aliases.get(x, x)


def parse_squad_file(path: Path, competition_key: str):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [x.strip() for x in raw.splitlines() if x.strip()]

    title = lines[0].lstrip("= ").strip() if lines else ""
    year = infer_year(str(path.parent.name), title, str(path))

    # Locate a CSV-style header.
    header_idx = None
    header = None
    for i, line in enumerate(lines[:12]):
        if "," not in line:
            continue
        cols = [clean_header(x) for x in next(csv.reader([line]))]
        low = {c.lower() for c in cols}
        if "name" in low and (
            "date of birth" in low
            or "current club" in low
            or "club" in low
            or "pos" in low
        ):
            header_idx = i
            header = cols
            break

    if header_idx is None:
        return [], {
            "competition_key": competition_key,
            "competition": NATIONAL_COMPETITIONS.get(competition_key, competition_key),
            "edition_folder": path.parent.name,
            "year": year,
            "team_file": path.stem,
            "title": title,
            "parsed": False,
            "players": 0,
            "has_name": False,
            "has_dob": False,
            "has_club": False,
            "has_position": False,
        }

    data_lines = lines[header_idx + 1:]
    rows = []
    for line in data_lines:
        if "," not in line:
            continue
        vals = next(csv.reader([line]))
        if len(vals) < 2:
            continue
        # Pad / collapse cautiously.
        if len(vals) < len(header):
            vals += [""] * (len(header) - len(vals))
        if len(vals) > len(header):
            vals = vals[:len(header) - 1] + [", ".join(vals[len(header) - 1:])]

        rec = dict(zip(header, vals))
        if not str(rec.get("Name", "")).strip():
            continue

        rec.update(
            {
                "competition_key": competition_key,
                "competition": NATIONAL_COMPETITIONS.get(competition_key, competition_key),
                "edition_folder": path.parent.name,
                "year": year,
                "team_file": path.stem,
                "title": title,
                "source_relpath": str(path),
            }
        )
        rows.append(rec)

    low_header = {h.lower() for h in header}
    meta = {
        "competition_key": competition_key,
        "competition": NATIONAL_COMPETITIONS.get(competition_key, competition_key),
        "edition_folder": path.parent.name,
        "year": year,
        "team_file": path.stem,
        "title": title,
        "parsed": True,
        "players": len(rows),
        "has_name": "name" in low_header,
        "has_dob": "date of birth" in low_header,
        "has_club": ("current club" in low_header or "club" in low_header),
        "has_position": ("pos" in low_header or "position" in low_header),
    }
    return rows, meta


def field_nonblank(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            s = df[c].astype("string").str.strip()
            return s.notna() & s.ne("")
    return pd.Series(False, index=df.index)


def main():
    zip_path = download(ZIP_URL, ZIP_PATH)
    repo = extract_repo(zip_path)

    national = repo / "national"
    if not national.exists():
        raise RuntimeError(f"National-team folder not found: {national}")

    all_player_rows = []
    team_editions = []

    for comp_key, comp_name in NATIONAL_COMPETITIONS.items():
        comp_dir = national / comp_key
        if not comp_dir.exists():
            continue

        for path in sorted(comp_dir.rglob("*.txt")):
            # skip readme-like files
            if path.name.lower().startswith("readme"):
                continue
            rows, meta = parse_squad_file(path, comp_key)
            team_editions.append(meta)
            all_player_rows.extend(rows)

    team_df = pd.DataFrame(team_editions)
    players = pd.DataFrame(all_player_rows)

    if players.empty:
        raise RuntimeError("No squad player rows were parsed.")

    players["_dob_known"] = field_nonblank(players, ["Date of Birth", "DOB", "Birth Date"])
    players["_club_known"] = field_nonblank(players, ["Current Club", "Club", "Team"])
    players["_pos_known"] = field_nonblank(players, ["Pos", "Position"])
    players["_name_known"] = field_nonblank(players, ["Name"])

    # Edition-level coverage.
    edition = (
        team_df.groupby(
            ["competition", "competition_key", "edition_folder", "year"],
            dropna=False,
        )
        .agg(
            teams=("team_file", "nunique"),
            parsed_team_files=("parsed", "sum"),
            squad_player_rows=("players", "sum"),
            teams_with_dob_field=("has_dob", "sum"),
            teams_with_club_field=("has_club", "sum"),
            teams_with_position_field=("has_position", "sum"),
        )
        .reset_index()
        .sort_values(["year", "competition"], na_position="first")
    )

    edition["dob_field_team_pct"] = (
        100 * edition["teams_with_dob_field"] / edition["teams"]
    ).round(2)
    edition["club_field_team_pct"] = (
        100 * edition["teams_with_club_field"] / edition["teams"]
    ).round(2)
    edition["position_field_team_pct"] = (
        100 * edition["teams_with_position_field"] / edition["teams"]
    ).round(2)

    # Player-value coverage.
    field_cov = (
        players.groupby(
            ["competition", "competition_key", "edition_folder", "year"],
            dropna=False,
        )
        .agg(
            player_rows=("Name", "size"),
            name_known=("_name_known", "sum"),
            dob_known=("_dob_known", "sum"),
            club_known=("_club_known", "sum"),
            position_known=("_pos_known", "sum"),
        )
        .reset_index()
        .sort_values(["year", "competition"], na_position="first")
    )

    for col in ["name_known", "dob_known", "club_known", "position_known"]:
        field_cov[col.replace("_known", "_coverage_pct")] = (
            100 * field_cov[col] / field_cov["player_rows"]
        ).round(2)

    comp_summary = (
        field_cov.groupby(["competition", "competition_key"], dropna=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            editions=("edition_folder", "nunique"),
            player_rows=("player_rows", "sum"),
            dob_known=("dob_known", "sum"),
            club_known=("club_known", "sum"),
            position_known=("position_known", "sum"),
        )
        .reset_index()
    )
    comp_summary["dob_coverage_pct"] = (
        100 * comp_summary["dob_known"] / comp_summary["player_rows"]
    ).round(2)
    comp_summary["club_coverage_pct"] = (
        100 * comp_summary["club_known"] / comp_summary["player_rows"]
    ).round(2)
    comp_summary["position_coverage_pct"] = (
        100 * comp_summary["position_known"] / comp_summary["player_rows"]
    ).round(2)

    usable = field_cov[
        (field_cov["dob_coverage_pct"] >= 95)
        & (field_cov["club_coverage_pct"] >= 95)
    ].copy()

    first_usable = (
        usable.groupby("competition", dropna=False)["year"]
        .min()
        .reset_index(name="first_edition_with_95pct_dob_and_club")
    )

    audit_summary = pd.DataFrame(
        [
            {
                "metric": "repository_license",
                "value": "CC0-1.0",
                "interpretation": "Open/reproducible for research reuse.",
            },
            {
                "metric": "national_competitions_audited",
                "value": team_df["competition"].nunique(),
                "interpretation": "Major tournament directories present in repository.",
            },
            {
                "metric": "team_edition_files",
                "value": len(team_df),
                "interpretation": "One squad file per team/tournament edition.",
            },
            {
                "metric": "parsed_team_edition_files",
                "value": int(team_df["parsed"].sum()),
                "interpretation": "Squad files parsed successfully.",
            },
            {
                "metric": "player_rows",
                "value": len(players),
                "interpretation": "Tournament squad player records.",
            },
            {
                "metric": "overall_dob_coverage_pct",
                "value": round(100 * players["_dob_known"].mean(), 2),
                "interpretation": "DOB availability across parsed player rows.",
            },
            {
                "metric": "overall_club_coverage_pct",
                "value": round(100 * players["_club_known"].mean(), 2),
                "interpretation": "Club-at-tournament availability across parsed player rows.",
            },
            {
                "metric": "universal_international_match_source",
                "value": "NO",
                "interpretation": (
                    "Tournament squads only; does not cover every friendly/qualifier."
                ),
            },
        ]
    )

    # Exports
    edition_path = OUTPUT_DIR / "open_squad_coverage_by_edition.csv"
    comp_path = OUTPUT_DIR / "open_squad_coverage_by_competition.csv"
    fields_path = OUTPUT_DIR / "open_squad_field_coverage.csv"
    teams_path = OUTPUT_DIR / "open_squad_team_editions.csv"
    players_path = OUTPUT_DIR / "open_squad_player_rows.csv"
    summary_path = OUTPUT_DIR / "open_squad_audit_summary.csv"
    usable_path = OUTPUT_DIR / "open_squad_first_usable_edition.csv"

    edition.to_csv(edition_path, index=False)
    comp_summary.to_csv(comp_path, index=False)
    field_cov.to_csv(fields_path, index=False)
    team_df.to_csv(teams_path, index=False)
    players.drop(columns=[c for c in players.columns if c.startswith("_")]).to_csv(
        players_path, index=False
    )
    audit_summary.to_csv(summary_path, index=False)
    first_usable.to_csv(usable_path, index=False)

    print("\n=== OPEN NATIONAL-SQUAD SOURCE SUMMARY ===")
    print(audit_summary.to_string(index=False))

    print("\n=== COVERAGE BY COMPETITION ===")
    print(
        comp_summary[
            [
                "competition",
                "first_year",
                "last_year",
                "editions",
                "player_rows",
                "dob_coverage_pct",
                "club_coverage_pct",
                "position_coverage_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== FIRST EDITION WITH >=95% DOB + CLUB COVERAGE ===")
    if first_usable.empty:
        print("None.")
    else:
        print(first_usable.to_string(index=False))

    print("\nCreated:")
    for p in [
        edition_path,
        comp_path,
        fields_path,
        teams_path,
        players_path,
        summary_path,
        usable_path,
    ]:
        print(p)

    print(
        "\nInterpretation: use this source as a tournament-specific open player "
        "layer. Do not infer that the first usable tournament year is a valid "
        "cutoff for ALL international matches."
    )


if __name__ == "__main__":
    main()