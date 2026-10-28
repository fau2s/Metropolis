"""
BuildOpenTournamentPlayerFeatures_v3.py

Production-style tournament squad feature builder with conservative,
date/edition-aware aliases derived from the exact source-label audit.

Important:
- Tournament squad layer only; not universal internationals.
- Age is computed at match date.
- No historical caps are invented.
- Copa America 2024 remains incomplete and is flagged rather than imputed.
"""

from __future__ import annotations
from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd

BASE = Path(r"D:\PhD\Metropolis")
RESULTS = BASE / "results.csv"
SQUADS = BASE / "open_squad_player_rows.csv"

OUT = BASE / "open_tournament_player_features_v4.csv"
DIAG = BASE / "open_tournament_player_join_diagnostics_v4.csv"
UNMATCHED = BASE / "open_tournament_player_unmatched_teams_v4.csv"

def norm(x):
    if pd.isna(x): return ""
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

# Safe aliases independent of edition.
GLOBAL = {
    "united states": "usa",
    "united states of america": "usa",
    "netherlands": "holland",
    "republic of ireland": "ireland",
    "ivory coast": "cote d ivoire",
    "cote divoire": "cote d ivoire",
    "czechia": "czech republic",
    "turkiye": "turkey",
}

# Historical aliases must be edition-aware.
def canonical_team(team, competition, edition_year):
    k = norm(team)
    y = None if pd.isna(edition_year) else int(edition_year)

    if k in GLOBAL:
        k = norm(GLOBAL[k])

    # results.csv modernizes some historical country names.
    if k == "germany" and y is not None and 1954 <= y <= 1990:
        return "west germany"
    if k == "russia" and y is not None and 1958 <= y <= 1990:
        return "ussr"
    if k == "russia" and y == 1992 and competition == "European Championship":
        return "cis"
    if k == "german dr" and y == 1974:
        return "east germany"

    # Yugoslavia-era naming in results.csv.
    if k == "serbia":
        if y == 1998:
            return "yugoslavia"
        if y == 2000 and competition == "European Championship":
            return "yugoslavia"
        if y == 2006:
            return "serbia and montenegro"

    # 1974 Zaire is represented as DR Congo in results.csv.
    if k in {"dr congo", "democratic republic of the congo"} and y == 1974:
        return "zaire"

    return k

def map_tournament(t):
    t = norm(t)
    if "fifa world cup" in t and "qualification" not in t: return "World Cup"
    if ("uefa euro" in t or "european championship" in t) and "qualification" not in t:
        return "European Championship"
    if "copa america" in t and "qualification" not in t: return "Copa America"
    if ("africa cup of nations" in t or "african cup of nations" in t) and "qualification" not in t:
        return "Africa Cup of Nations"
    if "confederations cup" in t: return "Confederations Cup"
    return None

def country_from_title(title, fallback):
    title = "" if pd.isna(title) else str(title).strip()
    return title.split(" - ", 1)[0].strip() if " - " in title else str(fallback).strip()

def find_col(df, names):
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lookup: return lookup[norm(n)]
    return None

r = pd.read_csv(RESULTS, low_memory=False)
s = pd.read_csv(SQUADS, low_memory=False)
r["date"] = pd.to_datetime(r["date"], errors="coerce")
r["match_year"] = r["date"].dt.year.astype("Int64")
r["squad_competition"] = r["tournament"].map(map_tournament)

name_col = find_col(s, ["Name"])
dob_col = find_col(s, ["Date of Birth","DOB","Birth Date"])
club_col = find_col(s, ["Current Club","Club"])
if name_col is None or dob_col is None:
    raise RuntimeError(f"Required Name/DOB columns missing: {list(s.columns)}")

try:
    s["dob"] = pd.to_datetime(s[dob_col], errors="coerce", format="mixed", dayfirst=True)
except (TypeError, ValueError):
    s["dob"] = pd.to_datetime(s[dob_col], errors="coerce", dayfirst=True)

s["edition_year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
s["squad_country"] = [country_from_title(t,f) for t,f in zip(s["title"],s["team_file"])]
s["team_key"] = [
    canonical_team(t,c,y) for t,c,y in zip(s["squad_country"],s["competition"],s["edition_year"])
]

edition_years = {
    c: sorted(pd.Series(g["edition_year"].dropna().unique()).astype(int).tolist())
    for c,g in s.groupby("competition")
}
def choose_edition(comp, date):
    if pd.isna(date) or comp not in edition_years: return pd.NA
    y=int(date.year); yrs=edition_years[comp]
    if y in yrs: return y
    # UEFA EURO 2020 was played in June/July 2021.
    if comp == "European Championship" and y == 2021 and 2020 in yrs:
        return 2020
    # Delayed winter tournaments such as AFCON 2023 played in early 2024.
    if date.month <= 3 and y-1 in yrs: return y-1
    return pd.NA

r["edition_year"] = pd.array([
    choose_edition(c,d) if c else pd.NA
    for c,d in zip(r["squad_competition"],r["date"])
], dtype="Int64")

frames=[]
for side in ("home","away"):
    x=r[["date","match_year","home_team","away_team","tournament",
         "squad_competition","edition_year"]].copy()
    x["side"]=side
    x["team"]=x["home_team"] if side=="home" else x["away_team"]
    x["team_key"]=[
        canonical_team(t,c,y)
        for t,c,y in zip(x["team"],x["squad_competition"],x["edition_year"])
    ]
    frames.append(x)
m=pd.concat(frames,ignore_index=True)
m=m[m["squad_competition"].notna()].copy()

j=m.merge(
    s,
    left_on=["squad_competition","edition_year","team_key"],
    right_on=["competition","edition_year","team_key"],
    how="left",
    suffixes=("_match","_squad")
)
j["age_at_match"]=(j["date"]-j["dob"]).dt.days/365.2425
j.loc[~j["age_at_match"].between(14,50),"age_at_match"]=np.nan

if club_col:
    z=j[club_col].astype("string").str.strip()
    j["_club_known"]=z.notna() & z.ne("")
else:
    j["_club_known"]=False

keys=["date","match_year","home_team","away_team","tournament",
      "squad_competition","edition_year","side","team"]

def nunique_nonblank(x):
    x=x.dropna().astype(str).str.strip()
    return x[x.ne("")].nunique()

spec=dict(
    squad_players=(name_col,lambda x:x.notna().sum()),
    squad_avg_age=("age_at_match","mean"),
    squad_age_sd=("age_at_match","std"),
    squad_min_age=("age_at_match","min"),
    squad_max_age=("age_at_match","max"),
    dob_known=("dob",lambda x:x.notna().sum()),
    club_known=("_club_known","sum"),
)
spec["squad_unique_clubs"]=(club_col,nunique_nonblank) if club_col else (name_col,lambda x:np.nan)
a=j.groupby(keys,dropna=False).agg(**spec).reset_index()
a["squad_data_available"]=a["squad_players"]>0
a["squad_dob_known_pct"]=np.where(a["squad_players"]>0,100*a["dob_known"]/a["squad_players"],np.nan)
a["squad_club_known_pct"]=np.where(a["squad_players"]>0,100*a["club_known"]/a["squad_players"],np.nan)

vals=["squad_data_available","squad_players","squad_avg_age","squad_age_sd",
      "squad_min_age","squad_max_age","squad_dob_known_pct",
      "squad_club_known_pct","squad_unique_clubs"]
base=["date","match_year","home_team","away_team","tournament","squad_competition","edition_year"]
h=a[a.side.eq("home")][base+vals].rename(columns={c:"home_"+c for c in vals})
w=a[a.side.eq("away")][base+vals].rename(columns={c:"away_"+c for c in vals})
out=h.merge(w,on=base,how="outer",validate="one_to_one")
out["both_squads_available"]=out["home_squad_data_available"].fillna(False)&out["away_squad_data_available"].fillna(False)
out["age_diff_home_minus_away"]=out["home_squad_avg_age"]-out["away_squad_avg_age"]

# Explicit source completeness flag: coverage is empirical, not imputed.
out["squad_source_complete_for_match"]=out["both_squads_available"]
out.to_csv(OUT,index=False)

diag=(out.groupby(["squad_competition","match_year","edition_year"],dropna=False)
      .agg(matches=("date","size"),
           home_squad_found=("home_squad_data_available","sum"),
           away_squad_found=("away_squad_data_available","sum"),
           both_squads_found=("both_squads_available","sum"))
      .reset_index())
diag["both_squads_coverage_pct"]=(100*diag["both_squads_found"]/diag["matches"]).round(2)
diag.to_csv(DIAG,index=False)

unmatched=a[a["edition_year"].notna() & ~a["squad_data_available"]][
    ["squad_competition","match_year","edition_year","team"]
].drop_duplicates()
unmatched.to_csv(UNMATCHED,index=False)

print("\n=== V4 FINAL TOURNAMENT-SQUAD COVERAGE ===")
print(diag[diag["edition_year"].notna()].to_string(index=False))
print("\n=== REMAINING UNMATCHED TEAMS ===")
print(unmatched.to_string(index=False) if len(unmatched) else "None")
print("\nCreated:")
print(OUT); print(DIAG); print(UNMATCHED)