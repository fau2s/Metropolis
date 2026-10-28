"""
BuildTournamentClubEloMatcher_v2.py

Re-runs the static ClubElo audit with:
1) conservative normalization,
2) a curated deterministic alias dictionary for known European club naming conventions,
3) no automatic fuzzy matching,
4) coverage diagnostics by edition.

Requires:
  D:\PhD\Metropolis\open_squad_player_rows.csv
  D:\PhD\Metropolis\clubelo_static_cache\EloRatings.csv

Outputs:
  tournament_clubelo_v2_player_matches.csv
  tournament_clubelo_v2_coverage_by_edition.csv
  tournament_clubelo_v2_unmatched_clubs.csv
  tournament_clubelo_v2_summary.csv
"""

from pathlib import Path
import re, unicodedata, difflib
import numpy as np
import pandas as pd

BASE=Path(r"D:\PhD\Metropolis")
SQUADS=BASE/"open_squad_player_rows.csv"
ELO_FILE=BASE/"clubelo_static_cache"/"EloRatings.csv"

OUTP=BASE/"tournament_clubelo_v2_player_matches.csv"
OUTE=BASE/"tournament_clubelo_v2_coverage_by_edition.csv"
OUTU=BASE/"tournament_clubelo_v2_unmatched_clubs.csv"
OUTS=BASE/"tournament_clubelo_v2_summary.csv"

def norm(x):
    if pd.isna(x): return ""
    x=unicodedata.normalize("NFKD",str(x))
    x="".join(c for c in x if not unicodedata.combining(c)).lower()
    x=x.replace("&"," and ")
    x=re.sub(r"\[[^\]]+\]"," ",x)       # source country tags, e.g. [KSA]
    x=re.sub(r"\b(fc|cf|afc|sc|ac|football club|futbol club)\b"," ",x)
    x=re.sub(r"[^a-z0-9]+"," ",x)
    return re.sub(r"\s+"," ",x).strip()

def findcol(df,names):
    z={norm(c):c for c in df.columns}
    return next((z[norm(n)] for n in names if norm(n) in z),None)

# Only aliases where identity is unambiguous. Values are ClubElo labels.
ALIASES_RAW = {
    "Manchester U":"Man United",
    "Manchester United":"Man United",
    "Manchester C":"Man City",
    "Manchester City":"Man City",
    "Tottenham H":"Tottenham",
    "Tottenham Hotspur":"Tottenham",
    "Paris St-Germain":"Paris SG",
    "Paris Saint-Germain":"Paris SG",
    "Bayer Leverkusen":"Leverkusen",
    "Borussia Dortmund":"Dortmund",
    "Atlético Madrid":"Ath Madrid",
    "Atletico Madrid":"Ath Madrid",
    "Internazionale":"Inter",
    "Inter Milan":"Inter",
    "Internazionale Milano":"Inter",
    "Schalke":"Schalke 04",
    "Schalke 04":"Schalke 04",
    "Eintracht Frankfurt":"Ein Frankfurt",
    "Newcastle U":"Newcastle",
    "Newcastle United":"Newcastle",
    "West Ham U":"West Ham",
    "West Ham United":"West Ham",
    "Wolverhampton W":"Wolves",
    "Wolverhampton Wanderers":"Wolves",
    "Borussia M'gladbach":"Gladbach",
    "Borussia Mönchengladbach":"Gladbach",
    "Bayern München":"Bayern",
    "Bayern Munich":"Bayern",
    "AS Roma":"Roma",
    "SS Lazio":"Lazio",
    "SSC Napoli":"Napoli",
    "AC Milan":"Milan",
    "Athletic Bilbao":"Ath Bilbao",
    "Athletic Club":"Ath Bilbao",
    "Real Betis":"Betis",
    "Real Sociedad":"Sociedad",
    "Sporting Lisbon":"Sporting",
    "Sporting CP":"Sporting",
    "Olympique Lyonnais":"Lyon",
    "Olympique Marseille":"Marseille",
    "Olympique de Marseille":"Marseille",
}
ALIASES={norm(k):norm(v) for k,v in ALIASES_RAW.items()}

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

s=pd.read_csv(SQUADS,low_memory=False)
elo=pd.read_csv(ELO_FILE,low_memory=False)
dc=findcol(elo,["date"]); cc=findcol(elo,["club"]); rc=findcol(elo,["elo"])
sc=findcol(s,["Current Club","Club"]); nc=findcol(s,["Name"])
if not all([dc,cc,rc,sc]): raise RuntimeError("Unexpected input columns")

elo["date2"]=pd.to_datetime(elo[dc],errors="coerce")
elo["club2"]=elo[cc].astype("string").str.strip()
elo["elo2"]=pd.to_numeric(elo[rc],errors="coerce")
elo["key"]=elo["club2"].map(norm)

s["edition_year"]=pd.to_numeric(s["year"],errors="coerce").astype("Int64")
s["club_raw"]=s[sc].astype("string").str.strip()
s["club_known"]=s["club_raw"].notna()&s["club_raw"].ne("")
s["raw_key"]=s["club_raw"].map(norm)
s["match_key"]=s["raw_key"].map(lambda k:ALIASES.get(k,k))

rows=[]; ed=[]
for (comp,y),g in s.dropna(subset=["edition_year"]).groupby(["competition","edition_year"]):
    y=int(y)
    if y<2000: continue
    st=start(comp,y)
    dates=elo.loc[elo.date2<=st,"date2"].dropna()
    if dates.empty: continue
    sd=dates.max(); snap=elo[elo.date2==sd]
    lookup={}
    for _,r in snap.iterrows():
        if r["key"]: lookup.setdefault(r["key"],r)
    names=list(lookup); known=matched=aliasn=0
    for _,r in g.iterrows():
        mt="missing_club"; ec=pd.NA; ev=np.nan;c1=pd.NA;score=np.nan
        if r.club_known:
            known+=1;k=r.match_key
            if k in lookup:
                rr=lookup[k];ec=rr.club2;ev=rr.elo2;matched+=1
                if r.raw_key in ALIASES and ALIASES[r.raw_key]!=r.raw_key:
                    mt="curated_alias";aliasn+=1
                else: mt="exact_normalized"
            else:
                cand=difflib.get_close_matches(k,names,n=1,cutoff=.35)
                if cand:
                    c1=lookup[cand[0]].club2
                    score=round(difflib.SequenceMatcher(None,k,cand[0]).ratio(),3)
                mt="unmatched"
        rows.append(dict(competition=comp,edition_year=y,snapshot_date=sd.date(),
          team_file=r.get("team_file",pd.NA),player_name=r.get(nc,pd.NA) if nc else pd.NA,
          club_raw=r.club_raw,match_type=mt,clubelo_club=ec,clubelo_elo=ev,
          candidate_1=c1,candidate_1_score=score))
    ed.append(dict(competition=comp,edition_year=y,player_rows=len(g),club_known_rows=known,
      matched_rows=matched,curated_alias_rows=aliasn,
      match_pct_known=round(100*matched/known,2) if known else np.nan,
      snapshot_date=sd.date()))
    print(f"{comp} {y}: {matched}/{known} = {100*matched/known:.2f}%")

p=pd.DataFrame(rows);e=pd.DataFrame(ed)
p.to_csv(OUTP,index=False);e.to_csv(OUTE,index=False)
u=p[p.match_type=="unmatched"].groupby(
 ["competition","edition_year","club_raw","candidate_1","candidate_1_score"],dropna=False
).size().reset_index(name="player_rows").sort_values("player_rows",ascending=False)
u.to_csv(OUTU,index=False)

known=sum(x["club_known_rows"] for x in ed);matched=sum(x["matched_rows"] for x in ed)
summary=pd.DataFrame([{
 "known_club_rows":known,"matched_rows_after_curated_aliases":matched,
 "match_pct_known":round(100*matched/known,2) if known else np.nan,
 "curated_alias_rows":sum(x["curated_alias_rows"] for x in ed),
 "policy":"Exact normalized + explicit aliases only; fuzzy candidates never accepted."
}])
summary.to_csv(OUTS,index=False)
print("\n=== V2 COVERAGE ===");print(e.to_string(index=False))
print("\n=== OVERALL ===");print(summary.to_string(index=False))
print("\nCreated:");[print(x) for x in (OUTP,OUTE,OUTU,OUTS)]