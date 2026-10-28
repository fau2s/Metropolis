from pathlib import Path
import re, unicodedata, urllib.request, difflib
import numpy as np
import pandas as pd

BASE = Path(r"D:\PhD\Metropolis")
SQUADS = BASE / "open_squad_player_rows.csv"
CACHE = BASE / "clubelo_static_cache"
CACHE.mkdir(parents=True, exist_ok=True)
ELO_FILE = CACHE / "EloRatings.csv"

OUT_PLAYER = BASE / "tournament_clubelo_static_player_audit.csv"
OUT_EDITION = BASE / "tournament_clubelo_static_coverage_by_edition.csv"
OUT_UNMATCHED = BASE / "tournament_clubelo_static_unmatched_clubs.csv"
OUT_SUMMARY = BASE / "tournament_clubelo_static_audit_summary.csv"

URLS = [
    "https://huggingface.co/datasets/xgabora/club-football-match-data/resolve/main/EloRatings.csv?download=true",
    "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/main/data/EloRatings.csv",
]

def norm(x):
    if pd.isna(x): return ""
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower().replace("&", " and ")
    x = re.sub(r"\b(fc|cf|afc|sc|ac|football club|futbol club)\b", " ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def findcol(df, names):
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lookup:
            return lookup[norm(n)]
    return None

def download_elo():
    if ELO_FILE.exists() and ELO_FILE.stat().st_size > 1000:
        print("Using cached Elo file:", ELO_FILE)
        return
    last = None
    for url in URLS:
        try:
            print("Downloading:", url)
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            if len(raw) < 1000:
                raise ValueError("download too small")
            ELO_FILE.write_bytes(raw)
            print("Saved:", ELO_FILE)
            return
        except Exception as e:
            last = e
            print("Failed:", e)
    raise RuntimeError(f"Could not download EloRatings.csv: {last}")

def tournament_start(comp, y):
    special = {
        ("European Championship", 2020): "2021-06-11",
        ("World Cup", 2022): "2022-11-20",
        ("Africa Cup of Nations", 2021): "2022-01-09",
        ("Africa Cup of Nations", 2023): "2024-01-13",
    }
    if (comp, y) in special:
        return pd.Timestamp(special[(comp, y)])
    if comp == "Africa Cup of Nations": return pd.Timestamp(f"{y}-01-15")
    if comp == "Copa America": return pd.Timestamp(f"{y}-06-15")
    if comp == "Confederations Cup": return pd.Timestamp(f"{y}-06-15")
    if comp == "European Championship": return pd.Timestamp(f"{y}-06-10")
    if comp == "World Cup": return pd.Timestamp(f"{y}-06-10")
    return pd.NaT

download_elo()
s = pd.read_csv(SQUADS, low_memory=False)
elo = pd.read_csv(ELO_FILE, low_memory=False)

date_col = findcol(elo, ["date"])
club_elo_col = findcol(elo, ["club"])
rating_col = findcol(elo, ["elo"])
if not all([date_col, club_elo_col, rating_col]):
    raise RuntimeError(f"Unexpected Elo columns: {list(elo.columns)}")

elo["elo_date"] = pd.to_datetime(elo[date_col], errors="coerce")
elo["elo_club"] = elo[club_elo_col].astype("string").str.strip()
elo["elo_value"] = pd.to_numeric(elo[rating_col], errors="coerce")
elo["elo_key"] = elo["elo_club"].map(norm)

club_col = findcol(s, ["Current Club", "Club"])
name_col = findcol(s, ["Name"])
if club_col is None:
    raise RuntimeError(f"Club column not found: {list(s.columns)}")

s["edition_year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
s["club_raw"] = s[club_col].astype("string").str.strip()
s["club_known"] = s["club_raw"].notna() & s["club_raw"].ne("")
s["club_key"] = s["club_raw"].map(norm)

records, edition_rows = [], []

for (comp, y), g in s.dropna(subset=["edition_year"]).groupby(["competition","edition_year"]):
    y = int(y)
    if y < 2000:
        continue
    start = tournament_start(comp, y)
    if pd.isna(start):
        continue
    dates = elo.loc[elo["elo_date"] <= start, "elo_date"].dropna()
    if dates.empty:
        continue
    snap_date = dates.max()
    snap = elo[elo["elo_date"] == snap_date].copy()
    lookup = {}
    for _, rr in snap.iterrows():
        if rr["elo_key"]:
            lookup.setdefault(rr["elo_key"], rr)
    names = list(lookup)
    known = matched = 0
    for _, r in g.iterrows():
        raw, ck = r["club_raw"], r["club_key"]
        mtype, ec, ev = "missing_club", pd.NA, np.nan
        c1 = c2 = pd.NA
        s1 = s2 = np.nan
        if r["club_known"]:
            known += 1
            if ck in lookup:
                rr = lookup[ck]
                mtype = "exact_normalized"
                ec = rr["elo_club"]
                ev = rr["elo_value"]
                matched += 1
            else:
                cand = difflib.get_close_matches(ck, names, n=2, cutoff=.35)
                if cand:
                    c1 = lookup[cand[0]]["elo_club"]
                    s1 = round(difflib.SequenceMatcher(None, ck, cand[0]).ratio(), 3)
                if len(cand) > 1:
                    c2 = lookup[cand[1]]["elo_club"]
                    s2 = round(difflib.SequenceMatcher(None, ck, cand[1]).ratio(), 3)
                mtype = "unmatched"
        records.append({
            "competition": comp, "edition_year": y,
            "tournament_start": start.date(), "snapshot_date": snap_date.date(),
            "team_file": r.get("team_file", pd.NA),
            "player_name": r.get(name_col, pd.NA) if name_col else pd.NA,
            "club_raw": raw, "club_known": r["club_known"],
            "match_type": mtype, "clubelo_club": ec, "clubelo_elo": ev,
            "candidate_1": c1, "candidate_1_score": s1,
            "candidate_2": c2, "candidate_2_score": s2,
        })
    edition_rows.append({
        "competition": comp, "edition_year": y,
        "tournament_start": start.date(), "snapshot_date": snap_date.date(),
        "player_rows": len(g), "club_known_rows": known,
        "exact_clubelo_matches": matched,
        "club_known_pct": round(100*known/len(g), 2) if len(g) else np.nan,
        "clubelo_match_pct_of_known": round(100*matched/known, 2) if known else np.nan,
        "clubelo_match_pct_all_players": round(100*matched/len(g), 2) if len(g) else np.nan,
        "snapshot_clubs": len(snap),
    })
    print(f"{comp} {y}: {matched}/{known} exact matches using {snap_date.date()}")

pdat = pd.DataFrame(records)
ddat = pd.DataFrame(edition_rows)
pdat.to_csv(OUT_PLAYER, index=False)
ddat.to_csv(OUT_EDITION, index=False)

u = pdat[(pdat["match_type"]=="unmatched") & pdat["club_known"]]
if len(u):
    u = (u.groupby(["competition","edition_year","club_raw","candidate_1","candidate_1_score",
                    "candidate_2","candidate_2_score"], dropna=False)
           .size().reset_index(name="player_rows")
           .sort_values(["competition","edition_year","player_rows"],
                        ascending=[True,True,False]))
u.to_csv(OUT_UNMATCHED, index=False)

known = int(pdat["club_known"].sum()) if len(pdat) else 0
matched = int((pdat["match_type"]=="exact_normalized").sum()) if len(pdat) else 0
summary = pd.DataFrame([{
    "elo_source_start": str(elo["elo_date"].min().date()),
    "elo_source_end": str(elo["elo_date"].max().date()),
    "elo_rows": len(elo), "editions_audited": len(ddat),
    "player_rows_audited": len(pdat), "known_club_rows": known,
    "exact_clubelo_matches": matched,
    "exact_match_pct_of_known": round(100*matched/known,2) if known else np.nan,
    "method": "latest static Elo snapshot on/before tournament start",
    "fuzzy_policy": "diagnostic only; never auto-accepted",
}])
summary.to_csv(OUT_SUMMARY, index=False)

print("\n=== STATIC CLUB ELO COVERAGE ===")
print(ddat.to_string(index=False))
print("\n=== OVERALL ===")
print(summary.to_string(index=False))
print("\nCreated:")
for x in [OUT_PLAYER, OUT_EDITION, OUT_UNMATCHED, OUT_SUMMARY]:
    print(x)