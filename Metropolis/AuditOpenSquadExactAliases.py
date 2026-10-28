"""
AuditOpenSquadExactAliases.py
=============================
Reads the audited open squad player rows and results.csv and prints exact
unmatched result-team -> squad-title-country candidates for the SAME
competition/edition. This is for building safe historical aliases from
observed source labels, not guesses.

Outputs:
D:\PhD\Metropolis\open_squad_exact_alias_audit.csv
"""

from pathlib import Path
import re, unicodedata, difflib
import pandas as pd

BASE = Path(r"D:\PhD\Metropolis")
RESULTS = BASE / "results.csv"
SQUADS = BASE / "open_squad_player_rows.csv"
OUT = BASE / "open_squad_exact_alias_audit.csv"

def norm(x):
    if pd.isna(x): return ""
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

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

def country_from_title(title, fallback):
    title = "" if pd.isna(title) else str(title).strip()
    return title.split(" - ", 1)[0].strip() if " - " in title else str(fallback).strip()

r = pd.read_csv(RESULTS, low_memory=False)
s = pd.read_csv(SQUADS, low_memory=False)
r["date"] = pd.to_datetime(r["date"], errors="coerce")
r["year"] = r["date"].dt.year
r["competition"] = r["tournament"].map(map_tournament)
s["edition_year"] = pd.to_numeric(s["year"], errors="coerce")
s["squad_country"] = [country_from_title(t,f) for t,f in zip(s["title"],s["team_file"])]

# one row per squad country/edition
ed = s[["competition","edition_year","squad_country"]].drop_duplicates()

rows=[]
for (comp, year), g in ed.groupby(["competition","edition_year"]):
    rr = r[(r["competition"]==comp) & (r["year"]==year)]
    result_teams = sorted(set(rr.home_team.dropna()) | set(rr.away_team.dropna()))
    squad_teams = sorted(g.squad_country.dropna().astype(str).unique())
    squad_norm = {norm(x):x for x in squad_teams}
    for rt in result_teams:
        if norm(rt) in squad_norm:
            continue
        ranked = sorted(
            [(difflib.SequenceMatcher(None,norm(rt),norm(st)).ratio(), st)
             for st in squad_teams],
            reverse=True
        )[:3]
        rows.append({
            "competition":comp, "edition_year":int(year),
            "results_team":rt,
            "candidate_1": ranked[0][1] if len(ranked)>0 else "",
            "score_1": round(ranked[0][0],3) if len(ranked)>0 else None,
            "candidate_2": ranked[1][1] if len(ranked)>1 else "",
            "score_2": round(ranked[1][0],3) if len(ranked)>1 else None,
            "candidate_3": ranked[2][1] if len(ranked)>2 else "",
            "score_3": round(ranked[2][0],3) if len(ranked)>2 else None,
        })

out=pd.DataFrame(rows)
out.to_csv(OUT,index=False)
print(out.to_string(index=False))
print(f"\nCreated: {OUT}")