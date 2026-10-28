from pathlib import Path
from io import BytesIO
import json
import re
import time
import unicodedata

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "results.csv"

OUTPUT_FILE = BASE_DIR / "world_cup_enriched.csv"
UNMATCHED_FILE = BASE_DIR / "world_cup_unmatched.csv"
SUMMARY_FILE = BASE_DIR / "world_cup_enrichment_summary.csv"
COACH_DIAGNOSTICS_FILE = BASE_DIR / "world_cup_coach_coverage_diagnostics.csv"


# Fjelstul World Cup Database.
# It contains World Cup matches, stages, groups, and manager appearances.
FJELSTUL_MATCHES_URL = (
    "https://raw.githubusercontent.com/"
    "jfjelstul/worldcup/master/data-csv/matches.csv"
)

FJELSTUL_MANAGER_APPEARANCES_URL = (
    "https://raw.githubusercontent.com/"
    "jfjelstul/worldcup/master/data-csv/manager_appearances.csv"
)


# Optional 2026 source.
# If this URL changes, the script will continue with historical data
# and report that 2026 data was unavailable.
WORLD_CUP_2026_MATCHES_URL = (
    "https://raw.githubusercontent.com/"
    "mominullptr/FIFA-World-Cup-2026-Dataset/main/matches_detailed.csv"
)


# Wikidata is used for national-team head-coach tenure dates.
# Historical World Cup coach identity remains anchored to Fjelstul.
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_CACHE_FILE = BASE_DIR / "wikidata_coach_tenures_cache.json"

HTTP_HEADERS = {
    "User-Agent": (
        "WorldCupCoachEnrichment/1.0 "
        "(research script; pandas/requests)"
    )
}

# Be deliberately conservative with Wikimedia's unauthenticated APIs.
# The script also respects Retry-After on 429/503 responses.
WIKIDATA_MIN_REQUEST_INTERVAL_SECONDS = 1.1
WIKIDATA_MAX_RETRIES = 8
WIKIDATA_DEFAULT_RETRY_SECONDS = 10
_last_wikidata_request_time = 0.0


# ---------------------------------------------------------------------
# Team-name normalization
# ---------------------------------------------------------------------

TEAM_ALIASES = {
    "west germany": "germany",
    "soviet union": "russia",
    "ussr": "russia",
    "czechoslovakia": "czech republic",
    "yugoslavia": "serbia",
    "kingdom of yugoslavia": "serbia",
    "netherlands east indies": "indonesia",
    "zaire": "dr congo",
    "rhodesia": "zimbabwe",
    "burma": "myanmar",
    "dahomey": "benin",
    "upper volta": "burkina faso",
    "cape verde islands": "cape verde",
    "swaziland": "eswatini",
    "korea republic": "south korea",
    "korea": "south korea",
    "usa": "united states",
    "united states of america": "united states",
    "german dr": "east germany",
    "german democratic republic": "east germany",
}


def normalize_text(value):
    """
    Normalize text for matching:
    - remove accents
    - lowercase
    - replace punctuation with spaces
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_team(value):
    normalized = normalize_text(value)
    return TEAM_ALIASES.get(normalized, normalized)


# ---------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------

def download_csv(url):
    print(f"Downloading: {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return pd.read_csv(BytesIO(response.content))


# ---------------------------------------------------------------------
# Stage and importance mapping
# ---------------------------------------------------------------------

def standardize_stage(stage_name):
    """
    Convert source-specific stage labels into consistent categories.
    """
    if pd.isna(stage_name):
        return pd.NA

    value = normalize_text(stage_name)

    if value == "final" or value.endswith(" final"):
        return "Final"

    if (
        "third" in value
        or "3rd" in value
        or "bronze" in value
    ):
        return "Third-place / Bronze match"

    if "semi" in value:
        return "Semi-final"

    if "quarter" in value:
        return "Quarter-final"

    if "round of 32" in value:
        return "Round of 32"

    if "round of 16" in value or "last 16" in value:
        return "Round of 16"

    if "group" in value:
        return "Group stage"

    return str(stage_name).strip()


def importance_score(stage):
    """
    Importance scale:

    1 = competitive match outside the World Cup final tournament
        or an unrecognized World Cup stage

    2 = World Cup group stage

    3 = Round of 32, Round of 16, or quarter-final

    4 = semi-final or third-place match

    5 = final
    """
    scores = {
        "Group stage": 2,
        "Round of 32": 3,
        "Round of 16": 3,
        "Quarter-final": 3,
        "Semi-final": 4,
        "Third-place / Bronze match": 4,
        "Final": 5,
    }

    return scores.get(stage, 1)


# ---------------------------------------------------------------------
# Source preparation
# ---------------------------------------------------------------------

def load_user_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Place results.csv beside this script."
        )

    results = pd.read_csv(INPUT_FILE)

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {sorted(missing)}"
        )

    results["date"] = pd.to_datetime(
        results["date"],
        errors="coerce",
    )

    return results


def load_fjelstul_data():
    matches = download_csv(FJELSTUL_MATCHES_URL)
    managers = download_csv(FJELSTUL_MANAGER_APPEARANCES_URL)

    matches["match_date"] = pd.to_datetime(
        matches["match_date"],
        errors="coerce",
    )

    managers["match_date"] = pd.to_datetime(
        managers["match_date"],
        errors="coerce",
    )

    # Keep men's FIFA World Cup matches only.
    # Fjelstul labels these as e.g. "1930 FIFA Men's World Cup",
    # so an equality test against "FIFA World Cup" would remove all rows.
    matches = matches[
        matches["tournament_name"]
        .astype(str)
        .str.contains(r"FIFA Men's World Cup$", regex=True, na=False)
    ].copy()

    managers = managers[
        managers["tournament_name"]
        .astype(str)
        .str.contains(r"FIFA Men's World Cup$", regex=True, na=False)
    ].copy()

    return matches, managers


def load_2026_data():
    """
    Load the optional 2026 dataset and identify likely column names.
    """
    matches = download_csv(WORLD_CUP_2026_MATCHES_URL)

    def find_column(*candidate_names):
        columns = {
            column.lower().strip(): column
            for column in matches.columns
        }

        for candidate in candidate_names:
            if candidate in columns:
                return columns[candidate]

        return None

    date_column = find_column(
        "date",
        "match_date",
    )

    home_column = find_column(
        "home_team",
        "home_team_name",
        "home team",
        "home",
    )

    away_column = find_column(
        "away_team",
        "away_team_name",
        "away team",
        "away",
    )

    stage_column = find_column(
        "stage",
        "stage_name",
        "round",
    )

    group_column = find_column(
        "group",
        "group_name",
    )

    required = {
        "date": date_column,
        "home_team": home_column,
        "away_team": away_column,
        "stage": stage_column,
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            "Could not identify 2026 columns. "
            f"Missing logical fields: {missing}. "
            f"Available columns: {list(matches.columns)}"
        )

    result = pd.DataFrame({
        "_date": pd.to_datetime(
            matches[date_column],
            errors="coerce",
        ),
        "_home": matches[home_column].map(normalize_team),
        "_away": matches[away_column].map(normalize_team),
        "_stage_2026_raw": matches[stage_column],
        "_group_2026": (
            matches[group_column]
            if group_column is not None
            else ""
        ),
    }).drop_duplicates(
        subset=["_date", "_home", "_away"]
    ).reset_index(drop=True)

    result["_stage_2026"] = result["_stage_2026_raw"].map(
        standardize_stage
    )

    return result


# ---------------------------------------------------------------------
# Match matching
# ---------------------------------------------------------------------

def match_against_fjelstul(left, reference):
    """
    Match World Cup rows against Fjelstul.

    Matching priority:
        1. date + home_team + away_team
        2. date + away_team + home_team

    Uses positional indexing throughout the fallback to avoid
    pandas "Unalignable boolean Series" errors.
    """

    left = left.copy().reset_index(drop=True)
    reference = reference.copy().reset_index(drop=True)

    # --------------------------------------------------------------
    # Normalize dates and team names
    # --------------------------------------------------------------

    left["_match_date"] = pd.to_datetime(
        left["date"],
        errors="coerce",
    )

    left["_home_key"] = left["home_team"].map(
        normalize_team
    )

    left["_away_key"] = left["away_team"].map(
        normalize_team
    )

    # --------------------------------------------------------------
    # Historical team identity corrections
    # --------------------------------------------------------------
    # These must be date-aware. A global alias would incorrectly
    # rewrite modern Indonesia or Serbia in later World Cups.

    # 1938: the team represented the Dutch East Indies.
    indonesia_1938_home_mask = (
        left["_match_date"].dt.year.eq(1938)
        & left["_home_key"].eq("indonesia")
    )
    left.loc[
        indonesia_1938_home_mask,
        "_home_key",
    ] = "dutch east indies"

    indonesia_1938_away_mask = (
        left["_match_date"].dt.year.eq(1938)
        & left["_away_key"].eq("indonesia")
    )
    left.loc[
        indonesia_1938_away_mask,
        "_away_key",
    ] = "dutch east indies"

    # 2006: Serbia and Montenegro is recorded as Serbia in the
    # international_results source.
    serbia_2006_home_mask = (
        left["_match_date"].dt.year.eq(2006)
        & left["_home_key"].eq("serbia")
    )
    left.loc[
        serbia_2006_home_mask,
        "_home_key",
    ] = "serbia and montenegro"

    serbia_2006_away_mask = (
        left["_match_date"].dt.year.eq(2006)
        & left["_away_key"].eq("serbia")
    )
    left.loc[
        serbia_2006_away_mask,
        "_away_key",
    ] = "serbia and montenegro"

    reference["_match_date"] = pd.to_datetime(
        reference["match_date"],
        errors="coerce",
    )

    reference["_home_key"] = reference["home_team_name"].map(
        normalize_team
    )

    reference["_away_key"] = reference["away_team_name"].map(
        normalize_team
    )

    reference_columns = [
        "_match_date",
        "_home_key",
        "_away_key",
        "key_id",
        "match_id",
        "stage_name",
        "group_name",
    ]

    reference_lookup = (
        reference[reference_columns]
        .drop_duplicates(
            subset=[
                "_match_date",
                "_home_key",
                "_away_key",
            ]
        )
        .copy()
    )

    # --------------------------------------------------------------
    # First pass: exact home/away order
    # --------------------------------------------------------------

    enriched = left.merge(
        reference_lookup,
        on=[
            "_match_date",
            "_home_key",
            "_away_key",
        ],
        how="left",
        sort=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # Second pass: reversed home/away order
    # --------------------------------------------------------------

    unmatched_mask = enriched["match_id"].isna()

    if unmatched_mask.any():

        unmatched_positions = (
            enriched.index[unmatched_mask].to_numpy()
        )

        reversed_rows = left.iloc[
            unmatched_positions
        ].copy()

        # Preserve the original keys before swapping.
        original_home_keys = (
            reversed_rows["_home_key"].copy()
        )

        reversed_rows["_home_key"] = (
            reversed_rows["_away_key"].copy()
        )

        reversed_rows["_away_key"] = (
            original_home_keys
        )

        reversed_matches = reversed_rows.merge(
            reference_lookup,
            on=[
                "_match_date",
                "_home_key",
                "_away_key",
            ],
            how="left",
            sort=False,
        ).reset_index(drop=True)

        # Put the reversed matches back into the corresponding
        # positions in the enriched dataframe.
        for column in [
            "key_id",
            "match_id",
            "stage_name",
            "group_name",
        ]:

            enriched.loc[
                unmatched_positions,
                column,
            ] = reversed_matches[column].to_numpy()

    return enriched

def attach_2026_stages(enriched, world_cup_2026=None):
    """
    Assign 2026 World Cup stages from official FIFA match-number structure.

    FIFA's 104-match format is:
        M1-M72   Group stage
        M73-M88  Round of 32
        M89-M96  Round of 16
        M97-M100 Quarter-finals
        M101-M102 Semi-finals
        M103     Third-place / Bronze match
        M104     Final

    We derive match number by chronological order within the 2026 World Cup
    rows. This avoids forcing joins between fixture datasets that disagree
    on some team names/fixtures.

    The optional world_cup_2026 argument is retained for compatibility, but
    is not required for stage assignment.
    """
    result = enriched.copy()

    mask_2026 = result["date"].dt.year.eq(2026)
    rows_2026 = result.loc[mask_2026].copy()

    if len(rows_2026) != 104:
        raise ValueError(
            "Expected exactly 104 FIFA World Cup rows for 2026, "
            f"but found {len(rows_2026)}. "
            "Refusing to infer stages from match order."
        )

    # Preserve original row order for matches on the same date. The source
    # dataset is expected to be in chronological match order.
    rows_2026["_original_position"] = range(len(rows_2026))
    rows_2026 = rows_2026.sort_values(
        ["date", "_original_position"],
        kind="stable",
    )
    rows_2026["_wc2026_match_number"] = range(1, 105)

    def stage_from_match_number(number):
        if number <= 72:
            return "Group stage"
        if number <= 88:
            return "Round of 32"
        if number <= 96:
            return "Round of 16"
        if number <= 100:
            return "Quarter-final"
        if number <= 102:
            return "Semi-final"
        if number == 103:
            return "Third-place / Bronze match"
        if number == 104:
            return "Final"
        return pd.NA

    rows_2026["_derived_stage"] = rows_2026[
        "_wc2026_match_number"
    ].map(stage_from_match_number)

    for idx, row in rows_2026.iterrows():
        stage = row["_derived_stage"]
        result.loc[idx, "world_cup_stage_raw"] = stage
        result.loc[idx, "world_cup_stage"] = stage
        result.loc[idx, "world_cup_group"] = pd.NA
        result.loc[idx, "stage_source"] = (
            "FIFA 2026 official match-number structure"
        )

    print(
        "2026 stages assigned from FIFA match-number structure: "
        f"{len(rows_2026):,}/104"
    )

    return result


# ---------------------------------------------------------------------
# Coach enrichment
# ---------------------------------------------------------------------

def create_manager_names(manager_appearances):
    managers = manager_appearances.copy()

    managers["manager_name"] = (
        managers["given_name"].fillna("").astype(str).str.strip()
        + " "
        + managers["family_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    managers["_team"] = managers["team_name"].map(normalize_team)

    return managers


def attach_coaches(enriched, managers):
    """
    Attach managers to each World Cup match.

    Fjelstul records one row per manager/team/match, so co-managers
    are preserved as semicolon-separated names.
    """
    def manager_names_for_match(match_id, is_home):
        if pd.isna(match_id):
            return pd.NA

        rows = managers[
            managers["match_id"].eq(match_id)
            & managers["home_team"].eq(is_home)
        ]

        names = [
            name
            for name in rows["manager_name"]
            if name and name != "nan"
        ]

        names = list(dict.fromkeys(names))

        if not names:
            return pd.NA

        return "; ".join(names)

    result = enriched.copy()

    result["home_coach"] = result["match_id"].map(
        lambda match_id: manager_names_for_match(
            match_id,
            True,
        )
    )

    result["away_coach"] = result["match_id"].map(
        lambda match_id: manager_names_for_match(
            match_id,
            False,
        )
    )

    return result


def calculate_prior_world_cup_coach_matches(enriched, managers):
    """
    Count a coach's PRIOR MEN'S WORLD CUP MATCHES before each match.

    This is a career World Cup experience measure, not a same-national-team
    measure. That distinction matters for coaches who managed more than one
    country at World Cups.

    Rules
    -----
    * Only manager appearances with match_date < current match date count.
    * A known coach with no earlier World Cup appearance receives 0.
    * Missing coach identity remains NA.
    * Co-managers are represented by the sum of the listed managers' prior
      appearances, preserving the existing semicolon-separated convention.
    * Matching uses normalized coach names so harmless punctuation/accent/
      spacing differences do not turn legitimate first-tournament zeros into
      missing values.

    The companion *_coach_world_cup_experience_quality column records whether
    the coach name exists anywhere in the 1930-2022 Fjelstul manager history.
    For a 2026 coach absent from Fjelstul, a value of 0 is still the logically
    correct prior-World-Cup count provided the coach identity is correct:
    Fjelstul covers every men's World Cup through 2022.
    """
    manager_history = managers.copy()

    manager_history["match_date"] = pd.to_datetime(
        manager_history["match_date"],
        errors="coerce",
    )

    manager_history["_coach_key"] = manager_history[
        "manager_name"
    ].map(normalize_text)

    # One manager/team/match row is expected. Use match_id uniqueness so an
    # accidental duplicate row cannot inflate experience.
    manager_history = manager_history.drop_duplicates(
        subset=["match_id", "_coach_key"]
    )

    known_coach_keys = set(
        manager_history["_coach_key"].dropna()
    )

    def prior_count_and_quality(row, side):
        coach_field = row[f"{side}_coach"]
        cutoff = row["date"]

        if pd.isna(coach_field) or pd.isna(cutoff):
            return pd.NA, "missing_coach_identity"

        coaches = [
            coach.strip()
            for coach in str(coach_field).split(";")
            if coach.strip()
        ]

        if not coaches:
            return pd.NA, "missing_coach_identity"

        coach_prior_counts = []
        qualities = []

        for coach in coaches:
            coach_key = normalize_text(coach)

            prior = manager_history[
                manager_history["_coach_key"].eq(coach_key)
                & manager_history["match_date"].lt(cutoff)
            ]

            coach_prior_counts.append(
                int(prior["match_id"].nunique())
            )

            if coach_key in known_coach_keys:
                qualities.append("fjelstul_identity_matched")
            else:
                # Because Fjelstul covers men's World Cups through 2022,
                # absence means there is no recorded prior WC appearance
                # under this normalized identity. Keep a flag so 2026 name
                # variants can be audited separately.
                qualities.append("no_fjelstul_prior_identity")

        if all(q == "fjelstul_identity_matched" for q in qualities):
            quality = "fjelstul_identity_matched"
        elif all(q == "no_fjelstul_prior_identity" for q in qualities):
            quality = "no_fjelstul_prior_identity"
        else:
            quality = "mixed_co_manager_identity"

        # This is a team-side predictor. When co-managers are listed, summing
        # their counts double-counts jointly managed matches. Use the maximum
        # individual prior-WC experience instead.
        total = max(coach_prior_counts) if coach_prior_counts else 0

        return total, quality

    result = enriched.copy()

    home_values = result.apply(
        lambda row: prior_count_and_quality(row, "home"),
        axis=1,
    )
    away_values = result.apply(
        lambda row: prior_count_and_quality(row, "away"),
        axis=1,
    )

    result["home_coach_world_cup_matches_before"] = pd.array(
        [value[0] for value in home_values],
        dtype="Int64",
    )
    result["away_coach_world_cup_matches_before"] = pd.array(
        [value[0] for value in away_values],
        dtype="Int64",
    )

    result["home_coach_world_cup_experience_quality"] = [
        value[1] for value in home_values
    ]
    result["away_coach_world_cup_experience_quality"] = [
        value[1] for value in away_values
    ]

    return result



# ---------------------------------------------------------------------
# All-international coach tenure enrichment (Wikidata)
# ---------------------------------------------------------------------

def _wait_for_wikidata_slot():
    """
    Keep requests below a conservative unauthenticated request rate.
    """
    global _last_wikidata_request_time

    elapsed = time.monotonic() - _last_wikidata_request_time
    remaining = WIKIDATA_MIN_REQUEST_INTERVAL_SECONDS - elapsed

    if remaining > 0:
        time.sleep(remaining)


def _retry_after_seconds(response, attempt):
    """
    Respect Wikimedia Retry-After. If absent/unparseable, use
    exponential backoff capped at two minutes.
    """
    raw = response.headers.get("Retry-After")

    if raw:
        try:
            return max(float(raw), 1.0)
        except ValueError:
            pass

    return min(
        WIKIDATA_DEFAULT_RETRY_SECONDS * (2 ** attempt),
        120,
    )


def _wikidata_request(url, params, timeout):
    """
    Rate-limit aware GET helper for both Action API and SPARQL.

    Retries HTTP 429 and 503, respecting Retry-After where supplied.
    """
    global _last_wikidata_request_time

    last_error = None

    for attempt in range(WIKIDATA_MAX_RETRIES):
        _wait_for_wikidata_slot()

        try:
            response = requests.get(
                url,
                params=params,
                headers=HTTP_HEADERS,
                timeout=timeout,
            )
            _last_wikidata_request_time = time.monotonic()

            if response.status_code in (429, 503):
                wait_seconds = _retry_after_seconds(
                    response,
                    attempt,
                )
                print(
                    f"Wikidata HTTP {response.status_code}; "
                    f"retrying after {wait_seconds:.0f}s "
                    f"(attempt {attempt + 1}/"
                    f"{WIKIDATA_MAX_RETRIES})"
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            last_error = error

            if attempt >= WIKIDATA_MAX_RETRIES - 1:
                raise

            wait_seconds = min(
                WIKIDATA_DEFAULT_RETRY_SECONDS * (2 ** attempt),
                120,
            )
            print(
                "Temporary Wikidata request error; "
                f"retrying after {wait_seconds:.0f}s: {error}"
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Wikidata request failed after retries.")


def _wikidata_get(params, timeout=60):
    # maxlag is recommended for non-interactive Action API clients.
    params = dict(params)
    params.setdefault("maxlag", 5)

    return _wikidata_request(
        WIKIDATA_API_URL,
        params=params,
        timeout=timeout,
    )


def _wikidata_sparql(query, timeout=90):
    return _wikidata_request(
        WIKIDATA_SPARQL_URL,
        params={"query": query, "format": "json"},
        timeout=timeout,
    )


def _load_wikidata_cache():
    if not WIKIDATA_CACHE_FILE.exists():
        return {
            "team_qids": {},
            "entities": {},
            "coach_reverse_tenures": {},
        }

    try:
        with WIKIDATA_CACHE_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {
            "team_qids": {},
            "entities": {},
            "coach_reverse_tenures": {},
        }

    data.setdefault("team_qids", {})
    data.setdefault("entities", {})
    data.setdefault("coach_reverse_tenures", {})
    return data


def _save_wikidata_cache(cache):
    with WIKIDATA_CACHE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def _wikidata_search_team_qid(team_name, cache):
    """
    Resolve a national-team name to a Wikidata item.

    Search is intentionally conservative: prefer results whose English label
    contains "national football team". Results are cached locally.
    """
    key = normalize_text(team_name)

    # Stable direct mappings for historical names that are difficult for
    # Wikidata text search. These take precedence over cached null results.
    direct_qids = {
        "german dr": "Q152424",
        "east germany": "Q152424",
    }

    if key in direct_qids:
        qid = direct_qids[key]
        cache["team_qids"][key] = qid
        _save_wikidata_cache(cache)
        return qid

    if key in cache["team_qids"]:
        return cache["team_qids"][key]

    # Historical/current naming variants that improve Wikidata search.
    search_overrides = {
        "south korea": "South Korea national football team",
        "north korea": "North Korea national football team",
        "united states": "United States men's national soccer team",
        "dr congo": "DR Congo national football team",
        "east germany": "East Germany national football team",
        "german dr": "East Germany national football team",
        "dutch east indies": "Dutch East Indies national football team",
        "serbia and montenegro": (
            "Serbia and Montenegro national football team"
        ),
        "czech republic": "Czech Republic national football team",
        "ivory coast": "Ivory Coast national football team",
    }

    query = search_overrides.get(
        key,
        f"{team_name} national football team",
    )

    data = _wikidata_get({
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": 10,
        "type": "item",
    })

    candidates = data.get("search", [])

    chosen = None
    for candidate in candidates:
        label = normalize_text(candidate.get("label", ""))
        description = normalize_text(candidate.get("description", ""))

        if (
            "national football team" in label
            or "national soccer team" in label
            or "national association football team" in description
            or "national football team" in description
            or "national soccer team" in description
        ):
            chosen = candidate.get("id")
            break

    if chosen is None and candidates:
        # Fallback only when the first result is clearly football-related.
        first = candidates[0]
        description = normalize_text(first.get("description", ""))
        if "football" in description or "soccer" in description:
            chosen = first.get("id")

    cache["team_qids"][key] = chosen
    _save_wikidata_cache(cache)
    return chosen


def _wikidata_get_entities(qids, cache):
    """
    Fetch Wikidata entities in batches and cache the raw JSON.
    """
    qids = [qid for qid in qids if qid]
    missing = [
        qid
        for qid in qids
        if qid not in cache["entities"]
    ]

    for start in range(0, len(missing), 40):
        batch = missing[start:start + 40]

        data = _wikidata_get({
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|claims",
            "languages": "en",
            "format": "json",
        })

        for qid, entity in data.get("entities", {}).items():
            cache["entities"][qid] = entity

        _save_wikidata_cache(cache)

    return {
        qid: cache["entities"].get(qid, {})
        for qid in qids
    }


def _wikidata_time_from_qualifier(statement, property_id):
    qualifiers = statement.get("qualifiers", {})
    values = qualifiers.get(property_id, [])

    if not values:
        return pd.NaT

    raw = (
        values[0]
        .get("datavalue", {})
        .get("value", {})
        .get("time")
    )

    if not raw:
        return pd.NaT

    # Wikidata dates are typically +YYYY-MM-DDT00:00:00Z.
    cleaned = raw.lstrip("+")
    return pd.to_datetime(cleaned, errors="coerce", utc=True).tz_localize(None)


def _wikidata_entity_label(entity):
    return (
        entity.get("labels", {})
        .get("en", {})
        .get("value", "")
    )


def load_wikidata_team_tenures(world_cup_rows):
    """
    Download head-coach statements (P286) for every national team appearing
    in the World Cup data. Start/end dates come from P580/P582 qualifiers.

    Returns one row per team/coach tenure statement.
    """
    cache = _load_wikidata_cache()

    teams = sorted(
        set(world_cup_rows["home_team"].dropna())
        | set(world_cup_rows["away_team"].dropna())
    )

    team_qids = {}
    print(
        f"Resolving Wikidata national-team entities: {len(teams):,} teams"
    )

    for number, team in enumerate(teams, start=1):
        qid = _wikidata_search_team_qid(team, cache)
        team_qids[team] = qid

        if number == 1 or number % 10 == 0 or number == len(teams):
            print(
                "  Wikidata team resolution: "
                f"{number:,}/{len(teams):,}"
            )

    unresolved = [
        team
        for team, qid in team_qids.items()
        if not qid
    ]

    if unresolved:
        print(
            "WARNING: Wikidata team entity not resolved for: "
            + ", ".join(unresolved)
        )

    team_entities = _wikidata_get_entities(
        list(team_qids.values()),
        cache,
    )

    coach_qids = set()
    raw_records = []

    for source_team, team_qid in team_qids.items():
        if not team_qid:
            continue

        entity = team_entities.get(team_qid, {})
        statements = entity.get("claims", {}).get("P286", [])

        for statement in statements:
            mainsnak = statement.get("mainsnak", {})
            coach_qid = (
                mainsnak
                .get("datavalue", {})
                .get("value", {})
                .get("id")
            )

            if not coach_qid:
                continue

            coach_qids.add(coach_qid)

            raw_records.append({
                "team_source_name": source_team,
                "team_qid": team_qid,
                "coach_qid": coach_qid,
                "tenure_start": _wikidata_time_from_qualifier(
                    statement,
                    "P580",
                ),
                "tenure_end": _wikidata_time_from_qualifier(
                    statement,
                    "P582",
                ),
            })

    coach_entities = _wikidata_get_entities(
        list(coach_qids),
        cache,
    )

    for record in raw_records:
        entity = coach_entities.get(record["coach_qid"], {})
        record["coach_name_wikidata"] = _wikidata_entity_label(entity)
        record["team_key"] = normalize_team(record["team_source_name"])
        record["coach_key"] = normalize_text(
            record["coach_name_wikidata"]
        )

    tenures = pd.DataFrame(raw_records)

    if tenures.empty:
        return tenures, cache

    tenures = tenures.drop_duplicates(
        subset=[
            "team_qid",
            "coach_qid",
            "tenure_start",
            "tenure_end",
        ]
    ).reset_index(drop=True)

    return tenures, cache


def load_other_national_team_tenures_for_coaches(coach_qids, cache):
    """
    Find other national teams managed by the same coaches.

    This reverse lookup is what allows us to estimate total senior
    international managerial experience across multiple national teams.
    """
    coach_qids = sorted({qid for qid in coach_qids if qid})
    records = []

    for start in range(0, len(coach_qids), 25):
        batch = coach_qids[start:start + 25]
        cache_key = "|".join(batch)

        cached = cache["coach_reverse_tenures"].get(cache_key)

        if cached is None:
            values = " ".join(f"wd:{qid}" for qid in batch)

            query = f"""
SELECT ?team ?teamLabel ?coach ?start ?end WHERE {{
  VALUES ?coach {{ {values} }}
  ?team p:P286 ?statement .
  ?statement ps:P286 ?coach .
  OPTIONAL {{ ?statement pq:P580 ?start . }}
  OPTIONAL {{ ?statement pq:P582 ?end . }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
  }}
}}
"""

            data = _wikidata_sparql(query)

            cached = []
            for binding in data.get("results", {}).get("bindings", []):
                cached.append({
                    "team_qid": binding["team"]["value"].rsplit("/", 1)[-1],
                    "team_source_name": binding.get(
                        "teamLabel",
                        {},
                    ).get("value", ""),
                    "coach_qid": binding["coach"]["value"].rsplit("/", 1)[-1],
                    "tenure_start": binding.get(
                        "start",
                        {},
                    ).get("value"),
                    "tenure_end": binding.get(
                        "end",
                        {},
                    ).get("value"),
                })

            cache["coach_reverse_tenures"][cache_key] = cached
            _save_wikidata_cache(cache)

        records.extend(cached)

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result["tenure_start"] = pd.to_datetime(
        result["tenure_start"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    result["tenure_end"] = pd.to_datetime(
        result["tenure_end"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    result["team_key"] = result["team_source_name"].map(
        normalize_team
    )

    return result.drop_duplicates().reset_index(drop=True)


def _active_tenure_for_match(
    team_name,
    match_date,
    coach_name,
    tenures,
):
    """
    Return the best Wikidata tenure row for a World Cup team/date.

    When a Fjelstul coach name is available, require a name match.
    For 2026 rows without Fjelstul data, use the active dated tenure.
    """
    if tenures.empty or pd.isna(match_date):
        return None

    team_key = normalize_team(team_name)
    candidates = tenures[tenures["team_key"].eq(team_key)].copy()

    if candidates.empty:
        return None

    if pd.notna(coach_name):
        coach_key = normalize_text(coach_name)
        by_name = candidates[
            candidates["coach_key"].eq(coach_key)
        ]

        if not by_name.empty:
            candidates = by_name
        else:
            # Do not silently substitute a different historical coach.
            return None

    dated = candidates[
        candidates["tenure_start"].notna()
        & candidates["tenure_start"].le(match_date)
        & (
            candidates["tenure_end"].isna()
            | candidates["tenure_end"].ge(match_date)
        )
    ]

    if dated.empty:
        return None

    # If multiple statements overlap, prefer the latest start date.
    dated = dated.sort_values(
        "tenure_start",
        ascending=False,
        na_position="last",
    )

    return dated.iloc[0]


def attach_wikidata_coach_tenures(enriched, team_tenures):
    """
    Attach tenure metadata and fill missing 2026 coach names.

    Historical Fjelstul coach names are preserved and never overwritten.
    """
    result = enriched.copy()

    for side in ["home", "away"]:
        result[f"{side}_coach_qid"] = pd.NA
        result[f"{side}_coach_tenure_start"] = pd.NaT
        result[f"{side}_coach_tenure_end"] = pd.NaT
        result[f"{side}_coach_tenure_source"] = pd.NA

    for idx, row in result.iterrows():
        for side in ["home", "away"]:
            existing_coach = row.get(f"{side}_coach", pd.NA)

            tenure = _active_tenure_for_match(
                team_name=row[f"{side}_team"],
                match_date=row["date"],
                coach_name=existing_coach,
                tenures=team_tenures,
            )

            if tenure is None:
                continue

            # Fill coach identity only when Fjelstul did not already provide it.
            if pd.isna(existing_coach):
                result.loc[idx, f"{side}_coach"] = (
                    tenure["coach_name_wikidata"]
                )

            result.loc[idx, f"{side}_coach_qid"] = tenure["coach_qid"]
            result.loc[idx, f"{side}_coach_tenure_start"] = (
                tenure["tenure_start"]
            )
            result.loc[idx, f"{side}_coach_tenure_end"] = (
                tenure["tenure_end"]
            )
            result.loc[idx, f"{side}_coach_tenure_source"] = "Wikidata"

    return result


def _result_match_events(results):
    """
    Create a two-sided match table:
    one row for the home team and one for the away team.
    """
    base = results.copy().reset_index(drop=False).rename(
        columns={"index": "_result_row_id"}
    )

    home = pd.DataFrame({
        "_result_row_id": base["_result_row_id"],
        "date": base["date"],
        "team": base["home_team"],
    })

    away = pd.DataFrame({
        "_result_row_id": base["_result_row_id"],
        "date": base["date"],
        "team": base["away_team"],
    })

    events = pd.concat([home, away], ignore_index=True)
    events["team_key"] = events["team"].map(normalize_team)

    return events


def add_all_international_coach_experience(
    enriched,
    all_results,
    team_tenures,
    cache,
):
    """
    Add two leakage-safe experience measures for each coach:

    1. *_coach_current_team_matches_before
       Senior international matches managed in the current national-team
       spell before the World Cup match.

    2. *_coach_career_international_matches_before
       Senior international matches managed across all national-team spells
       represented in Wikidata before the World Cup match.

    Counts use only rows with date < current World Cup match date.
    Missing tenure dates produce NA rather than fabricated counts.
    """
    result = enriched.copy()
    events = _result_match_events(all_results)

    coach_qids = set(
        result["home_coach_qid"].dropna().astype(str)
    ) | set(
        result["away_coach_qid"].dropna().astype(str)
    )

    reverse_tenures = load_other_national_team_tenures_for_coaches(
        coach_qids,
        cache,
    )

    # Include direct team-tenure records too, in case reverse lookup is sparse.
    direct = team_tenures[
        [
            "team_source_name",
            "team_qid",
            "coach_qid",
            "tenure_start",
            "tenure_end",
            "team_key",
        ]
    ].copy()

    if reverse_tenures.empty:
        all_tenures = direct
    else:
        reverse_subset = reverse_tenures[
            [
                "team_source_name",
                "team_qid",
                "coach_qid",
                "tenure_start",
                "tenure_end",
                "team_key",
            ]
        ]
        all_tenures = pd.concat(
            [direct, reverse_subset],
            ignore_index=True,
        ).drop_duplicates()

    for side in ["home", "away"]:
        result[
            f"{side}_coach_current_team_matches_before"
        ] = pd.array([pd.NA] * len(result), dtype="Int64")

        result[
            f"{side}_coach_career_international_matches_before"
        ] = pd.array([pd.NA] * len(result), dtype="Int64")

        result[
            f"{side}_coach_experience_quality"
        ] = pd.NA

    def count_events_for_tenure(
        team_key,
        start_date,
        end_date,
        cutoff_date,
    ):
        if pd.isna(start_date) or pd.isna(cutoff_date):
            return None

        effective_end = cutoff_date
        if pd.notna(end_date):
            effective_end = min(effective_end, end_date + pd.Timedelta(days=1))

        mask = (
            events["team_key"].eq(team_key)
            & events["date"].ge(start_date)
            & events["date"].lt(effective_end)
            & events["date"].lt(cutoff_date)
        )

        return int(events.loc[mask, "_result_row_id"].nunique())

    for idx, row in result.iterrows():
        cutoff = row["date"]

        for side in ["home", "away"]:
            qid = row.get(f"{side}_coach_qid")
            start = row.get(f"{side}_coach_tenure_start")
            end = row.get(f"{side}_coach_tenure_end")
            team_key = normalize_team(row[f"{side}_team"])

            if pd.isna(qid) or pd.isna(start):
                result.loc[
                    idx,
                    f"{side}_coach_experience_quality",
                ] = "missing_tenure"
                continue

            current_count = count_events_for_tenure(
                team_key=team_key,
                start_date=start,
                end_date=end,
                cutoff_date=cutoff,
            )

            if current_count is not None:
                result.loc[
                    idx,
                    f"{side}_coach_current_team_matches_before",
                ] = current_count

            coach_spells = all_tenures[
                all_tenures["coach_qid"].astype(str).eq(str(qid))
            ].copy()

            if coach_spells.empty:
                result.loc[
                    idx,
                    f"{side}_coach_experience_quality",
                ] = "current_spell_only"
                continue

            career_match_ids = set()
            dated_spell_count = 0

            for _, spell in coach_spells.iterrows():
                spell_start = spell["tenure_start"]

                if pd.isna(spell_start):
                    continue

                dated_spell_count += 1
                spell_end = spell["tenure_end"]

                effective_end = cutoff
                if pd.notna(spell_end):
                    effective_end = min(
                        effective_end,
                        spell_end + pd.Timedelta(days=1),
                    )

                mask = (
                    events["team_key"].eq(spell["team_key"])
                    & events["date"].ge(spell_start)
                    & events["date"].lt(effective_end)
                    & events["date"].lt(cutoff)
                )

                career_match_ids.update(
                    events.loc[mask, "_result_row_id"].astype(int).tolist()
                )

            if dated_spell_count > 0:
                result.loc[
                    idx,
                    f"{side}_coach_career_international_matches_before",
                ] = len(career_match_ids)

                result.loc[
                    idx,
                    f"{side}_coach_experience_quality",
                ] = "wikidata_dated_tenures"
            else:
                result.loc[
                    idx,
                    f"{side}_coach_experience_quality",
                ] = "missing_dated_spells"

    return result


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    results = load_user_results()

    world_cup_rows = results[
        results["tournament"].eq("FIFA World Cup")
    ].copy()

    print(f"Total input rows: {len(results):,}")
    print(f"World Cup rows: {len(world_cup_rows):,}")

    fjelstul_matches, manager_appearances = load_fjelstul_data()

    enriched = match_against_fjelstul(
        world_cup_rows,
        fjelstul_matches,
    )

    matched_count = enriched["match_id"].notna().sum()
    unmatched_count = enriched["match_id"].isna().sum()

    print(
        f"Fjelstul World Cup matches loaded: "
        f"{len(fjelstul_matches):,}"
    )
    print(
        f"Fjelstul matches matched: "
        f"{matched_count:,}"
    )
    print(
        f"Fjelstul matches unmatched: "
        f"{unmatched_count:,}"
    )

    enriched["world_cup_stage_raw"] = enriched["stage_name"]
    enriched["world_cup_stage"] = enriched["stage_name"].map(
        standardize_stage
    )
    enriched["world_cup_group"] = enriched["group_name"]

    enriched["stage_source"] = enriched[
        "world_cup_stage"
    ].notna().map({
        True: "Fjelstul",
        False: pd.NA,
    })

    # Assign 2026 stages from FIFA's official 104-match structure.
    # The external 2026 fixture CSV is no longer required for stage matching.
    enriched = attach_2026_stages(enriched)

    manager_appearances = create_manager_names(
        manager_appearances,
    )

    enriched = attach_coaches(
        enriched,
        manager_appearances,
    )

    enriched = calculate_prior_world_cup_coach_matches(
        enriched,
        manager_appearances,
    )

    # Broader senior-international coach experience.
    # Historical World Cup coach identity remains Fjelstul-based; Wikidata
    # supplies tenure dates and fills missing 2026 coach identities.
    team_tenures, wikidata_cache = load_wikidata_team_tenures(
        world_cup_rows
    )

    enriched = attach_wikidata_coach_tenures(
        enriched,
        team_tenures,
    )

    # Recalculate after Wikidata has filled missing 2026 coach identities.
    # This is essential: the earlier pass occurs before 2026 coach names exist.
    enriched = calculate_prior_world_cup_coach_matches(
        enriched,
        manager_appearances,
    )

    enriched = add_all_international_coach_experience(
        enriched,
        results,
        team_tenures,
        wikidata_cache,
    )

    enriched["importance_score"] = enriched[
        "world_cup_stage"
    ].map(importance_score)

    stage_flags = {
        "is_group_stage": "Group stage",
        "is_round_of_32": "Round of 32",
        "is_round_of_16": "Round of 16",
        "is_quarter_final": "Quarter-final",
        "is_semi_final": "Semi-final",
        "is_third_place": "Third-place / Bronze match",
        "is_final": "Final",
    }

    for column, stage in stage_flags.items():
        enriched[column] = (
            enriched["world_cup_stage"]
            .eq(stage)
            .astype("Int64")
        )

    enriched["stage_match_status"] = (
        enriched["world_cup_stage"]
        .notna()
        .map({
            True: "matched",
            False: "unmatched",
        })
    )

    enriched["coach_match_status"] = (
        enriched["home_coach"].notna()
        | enriched["away_coach"].notna()
    ).map({
        True: "matched",
        False: "unmatched",
    })

    # Detect duplicate source matches after joining.
    key_columns = [
        "date",
        "home_team",
        "away_team",
    ]

    join_counts = (
        enriched.groupby(key_columns)
        .size()
        .rename("join_row_count")
        .reset_index()
    )

    enriched = enriched.merge(
        join_counts,
        on=key_columns,
        how="left",
    )

    enriched["match_join_status"] = enriched[
        "join_row_count"
    ].map(
        lambda count: (
            "ambiguous"
            if count > 1
            else "unique"
        )
    )

    # Preserve original columns first.
    extra_columns = [
        "world_cup_stage_raw",
        "world_cup_stage",
        "world_cup_group",
        "importance_score",
        "is_group_stage",
        "is_round_of_32",
        "is_round_of_16",
        "is_quarter_final",
        "is_semi_final",
        "is_third_place",
        "is_final",
        "home_coach",
        "away_coach",
        "home_coach_world_cup_matches_before",
        "away_coach_world_cup_matches_before",
        "home_coach_world_cup_experience_quality",
        "away_coach_world_cup_experience_quality",
        "home_coach_qid",
        "away_coach_qid",
        "home_coach_tenure_start",
        "away_coach_tenure_start",
        "home_coach_tenure_end",
        "away_coach_tenure_end",
        "home_coach_current_team_matches_before",
        "away_coach_current_team_matches_before",
        "home_coach_career_international_matches_before",
        "away_coach_career_international_matches_before",
        "home_coach_experience_quality",
        "away_coach_experience_quality",
        "home_coach_tenure_source",
        "away_coach_tenure_source",
        "stage_source",
        "stage_match_status",
        "coach_match_status",
        "match_join_status",
    ]

    output_columns = (
        list(results.columns)
        + extra_columns
    )

    enriched[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    unmatched = enriched[
        (enriched["stage_match_status"] == "unmatched")
        | (enriched["coach_match_status"] == "unmatched")
        | (enriched["match_join_status"] == "ambiguous")
    ].copy()

    unmatched.to_csv(
        UNMATCHED_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Coverage diagnostics
    # -------------------------------------------------------------
    wc2026_mask = enriched["date"].dt.year.eq(2026)

    home_name = enriched["home_coach"].notna()
    away_name = enriched["away_coach"].notna()

    home_tenure = enriched["home_coach_tenure_start"].notna()
    away_tenure = enriched["away_coach_tenure_start"].notna()

    home_current_exp = enriched[
        "home_coach_current_team_matches_before"
    ].notna()
    away_current_exp = enriched[
        "away_coach_current_team_matches_before"
    ].notna()

    home_career_exp = enriched[
        "home_coach_career_international_matches_before"
    ].notna()
    away_career_exp = enriched[
        "away_coach_career_international_matches_before"
    ].notna()

    summary_rows = [
        ("input_rows", len(results)),
        ("world_cup_rows", len(world_cup_rows)),
        (
            "stage_matched",
            int((enriched["stage_match_status"] == "matched").sum()),
        ),
        (
            "stage_unmatched",
            int((enriched["stage_match_status"] == "unmatched").sum()),
        ),
        (
            "coach_rows_with_data",
            int((enriched["coach_match_status"] == "matched").sum()),
        ),
        (
            "ambiguous_joins",
            int((enriched["match_join_status"] == "ambiguous").sum()),
        ),
        ("home_coach_name_nonmissing", int(home_name.sum())),
        ("away_coach_name_nonmissing", int(away_name.sum())),
        ("both_coach_names_nonmissing", int((home_name & away_name).sum())),
        ("home_tenure_start_nonmissing", int(home_tenure.sum())),
        ("away_tenure_start_nonmissing", int(away_tenure.sum())),
        (
            "home_current_team_experience_nonmissing",
            int(home_current_exp.sum()),
        ),
        (
            "away_current_team_experience_nonmissing",
            int(away_current_exp.sum()),
        ),
        (
            "home_career_international_experience_nonmissing",
            int(home_career_exp.sum()),
        ),
        (
            "away_career_international_experience_nonmissing",
            int(away_career_exp.sum()),
        ),
        (
            "home_world_cup_experience_nonmissing",
            int(
                enriched[
                    "home_coach_world_cup_matches_before"
                ].notna().sum()
            ),
        ),
        (
            "away_world_cup_experience_nonmissing",
            int(
                enriched[
                    "away_coach_world_cup_matches_before"
                ].notna().sum()
            ),
        ),
        (
            "home_world_cup_experience_zero",
            int(
                enriched[
                    "home_coach_world_cup_matches_before"
                ].eq(0).sum()
            ),
        ),
        (
            "away_world_cup_experience_zero",
            int(
                enriched[
                    "away_coach_world_cup_matches_before"
                ].eq(0).sum()
            ),
        ),
        (
            "home_world_cup_identity_not_in_fjelstul",
            int(
                enriched[
                    "home_coach_world_cup_experience_quality"
                ].eq("no_fjelstul_prior_identity").sum()
            ),
        ),
        (
            "away_world_cup_identity_not_in_fjelstul",
            int(
                enriched[
                    "away_coach_world_cup_experience_quality"
                ].eq("no_fjelstul_prior_identity").sum()
            ),
        ),
        ("world_cup_2026_rows", int(wc2026_mask.sum())),
        (
            "world_cup_2026_home_coach_name_nonmissing",
            int((wc2026_mask & home_name).sum()),
        ),
        (
            "world_cup_2026_away_coach_name_nonmissing",
            int((wc2026_mask & away_name).sum()),
        ),
        (
            "world_cup_2026_both_coach_names_nonmissing",
            int((wc2026_mask & home_name & away_name).sum()),
        ),
        (
            "world_cup_2026_home_current_team_experience_nonmissing",
            int((wc2026_mask & home_current_exp).sum()),
        ),
        (
            "world_cup_2026_away_current_team_experience_nonmissing",
            int((wc2026_mask & away_current_exp).sum()),
        ),
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )

    # One row per team/side/quality bucket. This makes missingness patterns
    # visible without having to inspect the full enriched file manually.
    diagnostics_parts = []

    for side in ["home", "away"]:
        part = pd.DataFrame({
            "year": enriched["date"].dt.year,
            "team": enriched[f"{side}_team"],
            "side": side,
            "coach_name_missing": enriched[f"{side}_coach"].isna(),
            "tenure_start_missing": enriched[
                f"{side}_coach_tenure_start"
            ].isna(),
            "current_team_experience_missing": enriched[
                f"{side}_coach_current_team_matches_before"
            ].isna(),
            "career_experience_missing": enriched[
                f"{side}_coach_career_international_matches_before"
            ].isna(),
            "experience_quality": enriched[
                f"{side}_coach_experience_quality"
            ],
            "world_cup_experience_quality": enriched[
                f"{side}_coach_world_cup_experience_quality"
            ],
        })
        diagnostics_parts.append(part)

    coach_diagnostics = pd.concat(
        diagnostics_parts,
        ignore_index=True,
    )

    coach_diagnostics = (
        coach_diagnostics
        .groupby(
            [
                "year",
                "team",
                "side",
                "coach_name_missing",
                "tenure_start_missing",
                "current_team_experience_missing",
                "career_experience_missing",
                "experience_quality",
                "world_cup_experience_quality",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="world_cup_match_rows")
        .sort_values(
            ["year", "team", "side"],
            kind="stable",
        )
    )

    coach_diagnostics.to_csv(
        COACH_DIAGNOSTICS_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("Finished.")
    print(f"Created: {OUTPUT_FILE}")
    print(f"Created: {UNMATCHED_FILE}")
    print(f"Created: {SUMMARY_FILE}")
    print(f"Created: {COACH_DIAGNOSTICS_FILE}")


if __name__ == "__main__":
    main()