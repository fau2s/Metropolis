"""
BuildOpenTournamentPlayerFeatures_v2.py

Corrected tournament-squad integration:
1. Extracts country from the text BEFORE " - " in the squad title.
2. Uses explicit conservative aliases only after that extraction.
3. Handles tournaments whose nominal edition year differs from match calendar year
   (e.g. AFCON 2023 played in Jan/Feb 2024) by mapping each match to the nearest
   plausible squad edition for that competition.
4. Reports both raw match counts and eligible matches where squad editions exist.
"""

from __future__ import annotations
from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd

BASE = Path(r"D:\PhD\Metropolis")
RESULTS = BASE / "results.csv"
SQUADS = BASE / "open_squad_player_rows.csv"

OUT = BASE / "open_tournament_player_features_v2.csv"
DIAG = BASE / "open_tournament_player_join_diagnostics_v2.csv"
UNMATCHED = BASE / "open_tournament_player_unmatched_teams_v2.csv"

def norm(x):
    if pd.isna(x): return ""
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "iran ir": "iran",
    "ir iran": "iran",
    "dr congo": "democratic republic of the congo",
    "congo dr": "democratic republic of the congo",
    "guinea bissau": "guinea-bissau",
}

def team_key(x):
    y = norm(x)
    return norm(ALIASES.get(y, y))

def map_tournament(t):
    t = norm(t)
    if "fifa world cup" in t and "qualification" not in t: return "World Cup"
    if ("uefa euro" in t or "european championship" in t) and "qualification" not in t:
        return "European Championship"
    if "copa america" in t: return "Copa America"
    if "africa cup of nations" in t and "qualification" not in t:
        return "Africa Cup of Nations"
    if "confederations cup" in t: return "Confederations Cup"
    return None

def squad_country(title, team_file):
    title = "" if pd.isna(title) else str(title).strip()
    if " - " in title:
        return title.split(" - ", 1)[0].strip()
    # fallback only
    return str(team_file).strip()

def find_col(df, names):
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lookup:
            return lookup[norm(n)]
    return None

r = pd.read_csv(RESULTS, low_memory=False)
s = pd.read_csv(SQUADS, low_memory=False)

r["date"] = pd.to_datetime(r["date"], errors="coerce")
r["match_year"] = r["date"].dt.year.astype("Int64")
r["squad_competition"] = r["tournament"].map(map_tournament)

name_col = find_col(s, ["Name"])
dob_col = find_col(s, ["Date of Birth", "DOB", "Birth Date"])
club_col = find_col(s, ["Current Club", "Club"])
if name_col is None or dob_col is None:
    raise RuntimeError(f"Name/DOB columns not found. Columns: {list(s.columns)}")

try:
    s["dob"] = pd.to_datetime(s[dob_col], errors="coerce", format="mixed", dayfirst=True)
except (TypeError, ValueError):
    s["dob"] = pd.to_datetime(s[dob_col], errors="coerce", dayfirst=True)

s["edition_year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
s["squad_country"] = [
    squad_country(t, f) for t, f in zip(s["title"], s["team_file"])
]
s["team_key"] = s["squad_country"].map(team_key)

# Available edition years per competition.
edition_years = {
    comp: sorted(pd.Series(g["edition_year"].dropna().unique()).astype(int).tolist())
    for comp, g in s.groupby("competition")
}

def choose_edition(comp, match_date):
    """Nearest plausible edition: same calendar year first, then previous year.
    This handles delayed tournaments such as AFCON 2023 played in early 2024.
    Never maps to an edition more than one year away.
    """
    if pd.isna(match_date) or comp not in edition_years:
        return pd.NA
    y = int(match_date.year)
    yrs = edition_years[comp]
    if y in yrs:
        return y
    if y - 1 in yrs and match_date.month <= 3:
        return y - 1
    return pd.NA

r["edition_year"] = [
    choose_edition(c, d) if c else pd.NA
    for c, d in zip(r["squad_competition"], r["date"])
]
r["edition_year"] = pd.array(r["edition_year"], dtype="Int64")

# Build sides without duplicate columns.
side_frames = []
for side in ("home", "away"):
    x = r[[
        "date", "match_year", "home_team", "away_team", "tournament",
        "squad_competition", "edition_year"
    ]].copy()
    x["side"] = side
    x["team"] = x["home_team"] if side == "home" else x["away_team"]
    x["team_key"] = x["team"].map(team_key)
    side_frames.append(x)
m = pd.concat(side_frames, ignore_index=True)
m = m[m["squad_competition"].notna()].copy()

joined = m.merge(
    s,
    left_on=["squad_competition", "edition_year", "team_key"],
    right_on=["competition", "edition_year", "team_key"],
    how="left",
    suffixes=("_match", "_squad")
)

joined["age_at_match"] = (joined["date"] - joined["dob"]).dt.days / 365.2425
joined.loc[~joined["age_at_match"].between(14, 50), "age_at_match"] = np.nan

if club_col:
    ck = joined[club_col].astype("string").str.strip()
    joined["_club_known"] = ck.notna() & ck.ne("")
else:
    joined["_club_known"] = False

keys = [
    "date", "match_year", "home_team", "away_team", "tournament",
    "squad_competition", "edition_year", "side", "team"
]

def nunique_nonblank(x):
    x = x.dropna().astype(str).str.strip()
    return x[x.ne("")].nunique()

agg_spec = dict(
    squad_players=(name_col, lambda x: x.notna().sum()),
    squad_avg_age=("age_at_match", "mean"),
    squad_age_sd=("age_at_match", "std"),
    squad_min_age=("age_at_match", "min"),
    squad_max_age=("age_at_match", "max"),
    dob_known=("dob", lambda x: x.notna().sum()),
    club_known=("_club_known", "sum"),
)
if club_col:
    agg_spec["squad_unique_clubs"] = (club_col, nunique_nonblank)
else:
    agg_spec["squad_unique_clubs"] = (name_col, lambda x: np.nan)

agg = joined.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
agg["squad_data_available"] = agg["squad_players"] > 0
agg["squad_dob_known_pct"] = np.where(
    agg["squad_players"] > 0, 100*agg["dob_known"]/agg["squad_players"], np.nan)
agg["squad_club_known_pct"] = np.where(
    agg["squad_players"] > 0, 100*agg["club_known"]/agg["squad_players"], np.nan)

vals = [
    "squad_data_available","squad_players","squad_avg_age","squad_age_sd",
    "squad_min_age","squad_max_age","squad_dob_known_pct",
    "squad_club_known_pct","squad_unique_clubs"
]
basekeys = [
    "date","match_year","home_team","away_team","tournament",
    "squad_competition","edition_year"
]
home = agg[agg.side.eq("home")][basekeys+vals].rename(columns={c:f"home_{c}" for c in vals})
away = agg[agg.side.eq("away")][basekeys+vals].rename(columns={c:f"away_{c}" for c in vals})
out = home.merge(away, on=basekeys, how="outer", validate="one_to_one")
out["age_diff_home_minus_away"] = out["home_squad_avg_age"] - out["away_squad_avg_age"]
out["both_squads_available"] = (
    out["home_squad_data_available"].fillna(False) &
    out["away_squad_data_available"].fillna(False)
)
out.to_csv(OUT, index=False)

# Diagnostics distinguish editions that actually exist in the squad source.
diag = (
    out.groupby(["squad_competition","match_year","edition_year"], dropna=False)
       .agg(
           matches=("date","size"),
           home_squad_found=("home_squad_data_available","sum"),
           away_squad_found=("away_squad_data_available","sum"),
           both_squads_found=("both_squads_available","sum")
       ).reset_index()
)
diag["both_squads_coverage_pct"] = (
    100*diag["both_squads_found"]/diag["matches"]
).round(2)
diag.to_csv(DIAG, index=False)

# Unmatched teams only where an edition was actually selected.
unmatched = agg[
    agg["edition_year"].notna() & ~agg["squad_data_available"]
][["squad_competition","match_year","edition_year","team"]].drop_duplicates()
unmatched.to_csv(UNMATCHED, index=False)

print("\n=== CORRECTED TOURNAMENT PLAYER JOIN DIAGNOSTICS ===")
print(diag[diag["edition_year"].notna()].to_string(index=False))
print("\n=== UNMATCHED TEAMS IN ELIGIBLE EDITIONS ===")
print(unmatched.to_string(index=False) if len(unmatched) else "None")
print("\nCreated:")
print(OUT)
print(DIAG)
print(UNMATCHED)