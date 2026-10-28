"""
AuditTournamentClubEloTeamCoverage.py

Purpose
-------
Evaluate ClubElo coverage at the NATIONAL-TEAM SQUAD level after the V2 matcher.
This determines whether squad-level Elo aggregates are statistically usable.

Input
-----
D:\PhD\Metropolis\tournament_clubelo_v2_player_matches.csv

Outputs
-------
D:\PhD\Metropolis\tournament_clubelo_team_coverage.csv
D:\PhD\Metropolis\tournament_clubelo_team_coverage_summary.csv
D:\PhD\Metropolis\tournament_clubelo_candidate_squad_features.csv

Important
---------
- Missing Elo is NEVER filled with 0 or an average.
- Aggregate Elo features are computed only from matched players.
- Coverage counts/pct are retained explicitly.
- Reliability flags are supplied at 50%, 70%, and 80% coverage.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"D:\PhD\Metropolis")
INFILE = BASE / "tournament_clubelo_v2_player_matches.csv"

OUT_TEAM = BASE / "tournament_clubelo_team_coverage.csv"
OUT_SUMMARY = BASE / "tournament_clubelo_team_coverage_summary.csv"
OUT_FEATURES = BASE / "tournament_clubelo_candidate_squad_features.csv"

df = pd.read_csv(INFILE, low_memory=False)

required = [
    "competition","edition_year","team_file","player_name",
    "club_raw","match_type","clubelo_club","clubelo_elo"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing expected columns: {missing}")

df["clubelo_elo"] = pd.to_numeric(df["clubelo_elo"], errors="coerce")
df["elo_available"] = df["clubelo_elo"].notna()

def q75(x):
    x = pd.Series(x).dropna()
    return x.quantile(.75) if len(x) else np.nan

def q90(x):
    x = pd.Series(x).dropna()
    return x.quantile(.90) if len(x) else np.nan

rows = []

for (comp, year, team), g in df.groupby(
    ["competition","edition_year","team_file"], dropna=False
):
    n = len(g)
    known_club = int(g["club_raw"].notna().sum())
    elo_n = int(g["elo_available"].sum())
    vals = g.loc[g["elo_available"], "clubelo_elo"].sort_values(ascending=False)

    rows.append({
        "competition": comp,
        "edition_year": year,
        "team_file": team,
        "squad_players": n,
        "known_club_players": known_club,
        "clubelo_players": elo_n,
        "clubelo_coverage_pct": round(100 * elo_n / n, 2) if n else np.nan,
        "clubelo_coverage_known_club_pct":
            round(100 * elo_n / known_club, 2) if known_club else np.nan,

        "squad_mean_club_elo": vals.mean() if len(vals) else np.nan,
        "squad_median_club_elo": vals.median() if len(vals) else np.nan,
        "squad_max_club_elo": vals.max() if len(vals) else np.nan,
        "squad_club_elo_sd": vals.std(ddof=0) if len(vals) else np.nan,
        "squad_q75_club_elo": q75(vals),
        "squad_q90_club_elo": q90(vals),

        # These top-k values are only meaningful if enough players are covered.
        "squad_top5_mean_club_elo":
            vals.head(5).mean() if len(vals) >= 5 else np.nan,
        "squad_top11_mean_club_elo":
            vals.head(11).mean() if len(vals) >= 11 else np.nan,

        "clubelo_at_least_5_players": int(elo_n >= 5),
        "clubelo_at_least_11_players": int(elo_n >= 11),
        "clubelo_coverage_ge_50": int(n > 0 and elo_n/n >= .50),
        "clubelo_coverage_ge_70": int(n > 0 and elo_n/n >= .70),
        "clubelo_coverage_ge_80": int(n > 0 and elo_n/n >= .80),
    })

team = pd.DataFrame(rows).sort_values(
    ["competition","edition_year","team_file"]
)

# Candidate "safe" versions: only expose aggregates when coverage is reasonably high.
# 70% is an audit threshold, not assumed to be the final modeling cutoff.
for c in [
    "squad_mean_club_elo",
    "squad_median_club_elo",
    "squad_top5_mean_club_elo",
    "squad_top11_mean_club_elo",
]:
    team[c + "_cov70"] = team[c].where(team["clubelo_coverage_ge_70"] == 1)

team.to_csv(OUT_TEAM, index=False)

summary_rows = []
for (comp, year), g in team.groupby(["competition","edition_year"]):
    summary_rows.append({
        "competition": comp,
        "edition_year": year,
        "teams": len(g),
        "median_team_coverage_pct": round(g["clubelo_coverage_pct"].median(),2),
        "mean_team_coverage_pct": round(g["clubelo_coverage_pct"].mean(),2),
        "min_team_coverage_pct": round(g["clubelo_coverage_pct"].min(),2),
        "max_team_coverage_pct": round(g["clubelo_coverage_pct"].max(),2),
        "teams_ge_50_pct": int(g["clubelo_coverage_ge_50"].sum()),
        "teams_ge_70_pct": int(g["clubelo_coverage_ge_70"].sum()),
        "teams_ge_80_pct": int(g["clubelo_coverage_ge_80"].sum()),
        "teams_with_11plus_elo_players":
            int(g["clubelo_at_least_11_players"].sum()),
    })

summary = pd.DataFrame(summary_rows).sort_values(
    ["competition","edition_year"]
)
summary.to_csv(OUT_SUMMARY, index=False)

feature_cols = [
    "competition","edition_year","team_file",
    "squad_players","clubelo_players","clubelo_coverage_pct",
    "clubelo_coverage_ge_50","clubelo_coverage_ge_70","clubelo_coverage_ge_80",
    "squad_mean_club_elo","squad_median_club_elo",
    "squad_top5_mean_club_elo","squad_top11_mean_club_elo",
    "squad_mean_club_elo_cov70","squad_median_club_elo_cov70",
    "squad_top5_mean_club_elo_cov70","squad_top11_mean_club_elo_cov70",
]
team[feature_cols].to_csv(OUT_FEATURES, index=False)

print("\n=== TEAM-LEVEL CLUB ELO COVERAGE BY EDITION ===")
print(summary.to_string(index=False))

print("\n=== OVERALL TEAM COVERAGE ===")
print("National-team squad snapshots:", len(team))
print("Median squad coverage %:", round(team["clubelo_coverage_pct"].median(),2))
print("Squads >=50% coverage:", int(team["clubelo_coverage_ge_50"].sum()), "/", len(team))
print("Squads >=70% coverage:", int(team["clubelo_coverage_ge_70"].sum()), "/", len(team))
print("Squads >=80% coverage:", int(team["clubelo_coverage_ge_80"].sum()), "/", len(team))
print("Squads with >=11 Elo-rated players:",
      int(team["clubelo_at_least_11_players"].sum()), "/", len(team))

print("\nCreated:")
for p in [OUT_TEAM, OUT_SUMMARY, OUT_FEATURES]:
    print(p)