"""
Audit11v11Sample.py

SMALL, NON-CRAWLING feasibility parser for 11v11 match-report pages.

Why this is deliberately not a crawler
---------------------------------------
11v11's published terms assert copyright/database rights and restrict copying
and exploitation of downloaded material. This script therefore DOES NOT crawl
11v11, enumerate seasons, discover URLs, or download pages.

Use it only on a small set of HTML pages that you have lawfully saved for a
feasibility check / where your use is permitted. For any systematic research
extract, obtain permission/licensing from 11v11/AFS first.

Input
-----
Put saved 11v11 match-report HTML files in:
    D:\PhD\Metropolis\11v11_sample_html

Output
------
    D:\PhD\Metropolis\11v11_sample_match_audit.csv

The audit asks only whether each saved report contains enough structure for:
- match date
- two teams
- starting-lineup sections
- 11 player links per starting XI
- substitutions
- player IDs/URLs suitable for stable identity linking

This is a feasibility parser, not a production data extractor.
"""

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

INPUT_DIR = Path(r"D:\PhD\Metropolis\11v11_sample_html")
OUTPUT = Path(r"D:\PhD\Metropolis\11v11_sample_match_audit.csv")

MATCH_URL_RE = re.compile(r"/matches/([^\"'#?]+)")
PLAYER_URL_RE = re.compile(r"/players/([^\"'#?]+)")
DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})\b",
    re.I,
)


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def player_links_in_section(node):
    out = []
    if node is None:
        return out
    for a in node.find_all("a", href=True):
        href = a.get("href", "")
        if "/players/" in href:
            out.append((clean(a.get_text(" ", strip=True)), href))
    # stable de-duplication
    seen = set()
    unique = []
    for name, href in out:
        key = href.rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append((name, href))
    return unique


def parse_file(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    text = clean(soup.get_text(" ", strip=True))

    title = clean(soup.title.get_text(" ", strip=True)) if soup.title else ""
    date_m = DATE_RE.search(title + " " + text[:1500])
    match_date = date_m.group(1) if date_m else None

    # Count explicit lineup headings.
    headings = soup.find_all(
        lambda tag: tag.name in {"h3", "h4", "strong"}
        and "starting lineup" in clean(tag.get_text(" ", strip=True)).lower()
    )

    # Generic player-link inventory. Production parser would assign links to
    # sides using DOM structure after permission is obtained.
    all_player_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/players/" in href:
            all_player_links.append((clean(a.get_text(" ", strip=True)), href))

    seen = set()
    players = []
    for name, href in all_player_links:
        key = href.rstrip("/")
        if key not in seen:
            seen.add(key)
            players.append((name, href))

    # Identify likely team links.
    team_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/teams/" in href and "/tab/" not in href:
            item = (clean(a.get_text(" ", strip=True)), href)
            if item[0] and item not in team_links:
                team_links.append(item)

    has_subs = "substitutions" in text.lower()
    has_bench = "on the bench" in text.lower()
    has_comp = "competition" in text.lower()

    return {
        "file": path.name,
        "title": title,
        "match_date": match_date,
        "starting_lineup_heading_count": len(headings),
        "unique_player_links": len(players),
        "likely_two_starting_xi": len(players) >= 22,
        "has_substitutions_section": has_subs,
        "has_bench_section": has_bench,
        "has_competition_field": has_comp,
        "candidate_team_links": len(team_links),
        "stable_player_urls_present": len(players) > 0,
        "sample_player_urls": " | ".join(h for _, h in players[:5]),
        "parser_feasible": (
            match_date is not None
            and len(players) >= 22
            and has_comp
        ),
    }


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        list(INPUT_DIR.glob("*.html")) + list(INPUT_DIR.glob("*.htm"))
    )

    if not files:
        print(f"No HTML files found in: {INPUT_DIR}")
        print(
            "Save a SMALL, permission-compliant sample of 11v11 match-report "
            "pages there, then rerun."
        )
        return

    rows = []
    for path in files:
        try:
            rows.append(parse_file(path))
        except Exception as exc:
            rows.append({
                "file": path.name,
                "parser_feasible": False,
                "error": repr(exc),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)

    print("\n=== 11v11 SMALL-SAMPLE STRUCTURE AUDIT ===")
    print(df.to_string(index=False))
    print(f"\nCreated: {OUTPUT}")

    if "parser_feasible" in df:
        n = len(df)
        ok = int(df["parser_feasible"].fillna(False).sum())
        print(f"Parser-feasible pages: {ok}/{n} ({100*ok/n:.1f}%)")


if __name__ == "__main__":
    main()