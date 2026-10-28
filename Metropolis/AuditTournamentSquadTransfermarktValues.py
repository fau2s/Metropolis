"""
AuditTournamentSquadTransfermarktValues.py

Audits historical player market-value coverage for open national-team tournament
squads using the locally cached transfermarkt-datasets DuckDB.

Inputs
------
D:\PhD\Metropolis\open_squad_player_rows.csv
D:\PhD\Metropolis\player_layer_prototype_cache\transfermarkt-datasets.duckdb

Outputs
-------
tournament_tm_value_player_audit.csv
tournament_tm_value_coverage_by_edition.csv
tournament_tm_value_team_coverage.csv
tournament_tm_value_unmatched_players.csv
tournament_tm_value_audit_summary.csv

Matching policy
---------------
1. Exact normalized player name + exact date of birth.
2. Exact date of birth + strong name similarity (>=0.90) only when unique.
No DOB-free fuzzy matching.
Historical value must be dated on/before the tournament start (no leakage).
"""

from pathlib import Path
import re, unicodedata, difflib
import numpy as np
import pandas as pd
import duckdb

BASE=Path(r"D:\PhD\Metropolis")
SQUADS=BASE/"open_squad_player_rows.csv"
DB=BASE/"player_layer_prototype_cache"/"transfermarkt-datasets.duckdb"

OUTP=BASE/"tournament_tm_value_player_audit.csv"
OUTE=BASE/"tournament_tm_value_coverage_by_edition.csv"
OUTT=BASE/"tournament_tm_value_team_coverage.csv"
OUTU=BASE/"tournament_tm_value_unmatched_players.csv"
OUTS=BASE/"tournament_tm_value_audit_summary.csv"

def norm(x):
    if pd.isna(x): return ""
    x=unicodedata.normalize("NFKD",str(x))
    x="".join(c for c in x if not unicodedata.combining(c)).lower()
    x=re.sub(r"[^a-z0-9]+"," ",x)
    return re.sub(r"\s+"," ",x).strip()

def findcol(df,names):
    z={norm(c):c for c in df.columns}
    return next((z[norm(n)] for n in names if norm(n) in z),None)

def start(comp,y):
    special={
      ("European Championship",2020):"2021-06-11",
      ("World Cup",2022):"2022-11-20",
      ("Africa Cup of Nations",2021):"2022-01-09",
      ("Africa Cup of Nations",2023):"2024-01-13",
    }
    if (comp,y) in special:return pd.Timestamp(special[(comp,y)])
    if comp=="Africa Cup of Nations":return pd.Timestamp(f"{y}-01-15")
    if comp=="Copa America":return pd.Timestamp(f"{y}-06-15")
    if comp=="Confederations Cup":return pd.Timestamp(f"{y}-06-15")
    if comp=="European Championship":return pd.Timestamp(f"{y}-06-10")
    if comp=="World Cup":return pd.Timestamp(f"{y}-06-10")
    return pd.NaT

if not DB.exists():
    raise FileNotFoundError(f"DuckDB not found: {DB}")

con=duckdb.connect(str(DB),read_only=True)
tables={r[0] for r in con.execute("SHOW TABLES").fetchall()}
if "players" not in tables:
    raise RuntimeError("players table not found")
if "player_valuations" not in tables:
    raise RuntimeError("player_valuations table not found; tables="+str(sorted(tables)))

players=con.execute("SELECT * FROM players").df()
vals=con.execute("SELECT * FROM player_valuations").df()
con.close()

s=pd.read_csv(SQUADS,low_memory=False)

# Squad columns
namec=findcol(s,["Name"])
dobc=findcol(s,["Date of Birth","birth_date","dob"])
if namec is None or dobc is None:
    raise RuntimeError("Squad Name/DOB columns not found: "+str(list(s.columns)))

# TM player columns
pid=findcol(players,["player_id"])
pname=findcol(players,["name","player_name"])
pdob=findcol(players,["date_of_birth","birth_date"])
if not all([pid,pname,pdob]):
    raise RuntimeError("Unexpected players columns: "+str(list(players.columns)))

# Valuation columns
vpid=findcol(vals,["player_id"])
vdate=findcol(vals,["date","valuation_date"])
vvalue=findcol(vals,["market_value_in_eur","market_value"])
if not all([vpid,vdate,vvalue]):
    raise RuntimeError("Unexpected player_valuations columns: "+str(list(vals.columns)))

players["dob2"]=pd.to_datetime(players[pdob],errors="coerce").dt.date
players["name2"]=players[pname].map(norm)
s["dob2"]=pd.to_datetime(s[dobc],errors="coerce").dt.date
s["name2"]=s[namec].map(norm)
s["edition_year"]=pd.to_numeric(s["year"],errors="coerce").astype("Int64")

vals["vdate2"]=pd.to_datetime(vals[vdate],errors="coerce")
vals["value2"]=pd.to_numeric(vals[vvalue],errors="coerce")
vals=vals.dropna(subset=[vpid,"vdate2","value2"]).sort_values("vdate2")

# Index TM players by DOB.
bydob={}
for _,r in players.dropna(subset=["dob2"]).iterrows():
    bydob.setdefault(r["dob2"],[]).append(r)

rows=[]
for _,r in s.dropna(subset=["edition_year"]).iterrows():
    y=int(r["edition_year"]); comp=r["competition"]; st=start(comp,y)
    candidates=bydob.get(r["dob2"],[]) if pd.notna(r["dob2"]) else []
    match=None; method="unmatched"; score=np.nan

    exact=[x for x in candidates if x["name2"]==r["name2"]]
    if len(exact)==1:
        match=exact[0]; method="exact_name_dob"; score=1.0
    elif candidates:
        scored=[(difflib.SequenceMatcher(None,r["name2"],x["name2"]).ratio(),x) for x in candidates]
        scored.sort(key=lambda z:z[0],reverse=True)
        if scored and scored[0][0]>=.90 and (len(scored)==1 or scored[0][0]>scored[1][0]):
            score,match=scored[0]; method="unique_dob_name90"

    tm_id=np.nan; tm_name=pd.NA; val=np.nan; valdate=pd.NaT
    if match is not None:
        tm_id=match[pid]; tm_name=match[pname]
        hist=vals[(vals[vpid]==tm_id)&(vals["vdate2"]<=st)]
        if len(hist):
            z=hist.iloc[-1]; val=z["value2"]; valdate=z["vdate2"]

    rows.append({
      "competition":comp,"edition_year":y,"team_file":r.get("team_file",pd.NA),
      "player_name":r[namec],"date_of_birth":r[dobc],"tournament_start":st.date(),
      "tm_player_id":tm_id,"tm_player_name":tm_name,"player_match_method":method,
      "name_similarity":round(score,3) if pd.notna(score) else np.nan,
      "tm_identity_matched":int(match is not None),
      "market_value_available":int(pd.notna(val)),
      "market_value_date":valdate.date() if pd.notna(valdate) else pd.NaT,
      "market_value_eur":val
    })

p=pd.DataFrame(rows)
p.to_csv(OUTP,index=False)

ed=(p.groupby(["competition","edition_year"])
      .agg(player_rows=("player_name","size"),
           identity_matched=("tm_identity_matched","sum"),
           value_available=("market_value_available","sum"))
      .reset_index())
ed["identity_match_pct"]=(100*ed.identity_matched/ed.player_rows).round(2)
ed["value_coverage_pct"]=(100*ed.value_available/ed.player_rows).round(2)
ed.to_csv(OUTE,index=False)

team=(p.groupby(["competition","edition_year","team_file"])
        .agg(squad_players=("player_name","size"),
             identity_matched=("tm_identity_matched","sum"),
             value_players=("market_value_available","sum"),
             squad_mean_market_value_eur=("market_value_eur","mean"),
             squad_median_market_value_eur=("market_value_eur","median"),
             squad_total_market_value_eur=("market_value_eur","sum"))
        .reset_index())
team["value_coverage_pct"]=(100*team.value_players/team.squad_players).round(2)
team["coverage_ge_50"]=(team.value_coverage_pct>=50).astype(int)
team["coverage_ge_70"]=(team.value_coverage_pct>=70).astype(int)
team["coverage_ge_80"]=(team.value_coverage_pct>=80).astype(int)
team.to_csv(OUTT,index=False)

p[p.tm_identity_matched==0].to_csv(OUTU,index=False)

summary=pd.DataFrame([{
 "player_rows":len(p),
 "identity_matched":int(p.tm_identity_matched.sum()),
 "identity_match_pct":round(100*p.tm_identity_matched.mean(),2),
 "historical_value_available":int(p.market_value_available.sum()),
 "historical_value_coverage_pct":round(100*p.market_value_available.mean(),2),
 "team_snapshots":len(team),
 "teams_value_coverage_ge_50":int(team.coverage_ge_50.sum()),
 "teams_value_coverage_ge_70":int(team.coverage_ge_70.sum()),
 "teams_value_coverage_ge_80":int(team.coverage_ge_80.sum()),
 "policy":"DOB-based identity matching; historical valuation date <= tournament start only."
}])
summary.to_csv(OUTS,index=False)

print("\n=== TRANSFERMARKT HISTORICAL VALUE COVERAGE BY EDITION ===")
print(ed.to_string(index=False))
print("\n=== OVERALL ===")
print(summary.to_string(index=False))
print("\nCreated:")
for x in [OUTP,OUTE,OUTT,OUTU,OUTS]: print(x)