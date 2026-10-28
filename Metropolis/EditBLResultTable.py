import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import defaultdict, deque

# Function to calculate entropy rate given a sequence of outcomes
def calculate_entropy_rate(outcomes):
    # Encode outcomes (Win = 1, Draw = 0, Loss = -1)
    outcome_map = {'W': 1, 'D': 0, 'L': -1}
    outcomes_encoded = [outcome_map[outcome] for outcome in outcomes]

    # Create transition matrix (3x3 for Win, Draw, Loss)
    transition_matrix = np.zeros((3, 3))

    # Map outcomes to indices: Win -> 0, Draw -> 1, Loss -> 2
    mapped_outcomes = np.array(outcomes_encoded) + 1

    # Calculate transition frequencies
    for (i, j) in zip(mapped_outcomes[:-1], mapped_outcomes[1:]):
        transition_matrix[i, j] += 1

    # Convert to probabilities (normalize each row)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    # If transition matrix has any invalid rows (e.g., all zeros), skip entropy rate calculation
    if np.any(np.isnan(transition_matrix)) or np.any(np.isinf(transition_matrix)):
        return None

    # Calculate the stationary distribution
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])
        stationary = stationary / stationary.sum()
        stationary = stationary.flatten()

        # If stationary distribution has wrong shape, return None
        if stationary.shape[0] != 3:
            return None

        # Calculate the entropy rate
        entropy_rate = -np.sum(
            stationary * np.sum(transition_matrix * np.log2(transition_matrix, where=transition_matrix > 0), axis=1))
    except (np.linalg.LinAlgError, ValueError) as e:
        # Handle cases where the transition matrix is singular, ill-conditioned, or any other error
        entropy_rate = None

    return entropy_rate

# Function to calculate entropy rate from a sequence of results
def calculate_entropy_rate_2(result_sequence):
    if len(result_sequence) < 2:
        return 0  # Not enough data to calculate entropy

    # Count transitions between states
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(result_sequence) - 1):
        transitions[result_sequence[i]][result_sequence[i + 1]] += 1

    # Calculate transition probabilities
    entropy_rate = 0
    for current_state, next_states in transitions.items():
        total_transitions = sum(next_states.values())
        for next_state, count in next_states.items():
            prob = count / total_transitions
            entropy_rate -= prob * np.log2(prob)

    return entropy_rate


def main():
    # Load the CSV file
    file_path = "./bulidata.csv"  # Replace with your file path
    df = pd.read_csv(file_path, sep=';')

    # Add columns for points
    df['HOME_POINTS'] = np.where(df['GOALS_HOME'] > df['GOALS_AWAY'], 3,
                                 np.where(df['GOALS_HOME'] < df['GOALS_AWAY'], 0, 1))
    df['AWAY_POINTS'] = np.where(df['GOALS_AWAY'] > df['GOALS_HOME'], 3,
                                 np.where(df['GOALS_AWAY'] < df['GOALS_HOME'], 0, 1))


    # Initialize columns for home and away team positions and entropy rates
    df['HOME_POSITION'] = np.nan
    df['AWAY_POSITION'] = np.nan
    df['HOME_POINTS_SUM'] = np.nan
    df['AWAY_POINTS_SUM'] = np.nan
    df['HOME_ENTROPY_RATE'] = np.nan
    df['AWAY_ENTROPY_RATE'] = np.nan
    df['HOME_POINTS'] = 0
    df['AWAY_POINTS'] = 0
    df['HOME_POSITION'] = np.nan
    df['AWAY_POSITION'] = np.nan
    df['HOME_TOTAL_POINTS'] = 0
    df['AWAY_TOTAL_POINTS'] = 0
    df['KL_HOME_SCORED_AWAY_CONCEDED'] = np.nan
    df['KL_HOME_CONCEDED_AWAY_SCORED'] = np.nan
    df['ENTROPY_HOME_SCORED'] = np.nan
    df['ENTROPY_HOME_CONCEDED'] = np.nan
    df['ENTROPY_AWAY_SCORED'] = np.nan
    df['ENTROPY_AWAY_CONCEDED'] = np.nan
    df['ENTROPY_RATE_HOME'] = np.nan
    df['ENTROPY_RATE_AWAY'] = np.nan

    # Iterate through each season
    for season in df['SEASON'].unique():
        # Initialize a DataFrame to keep track of standings and outcomes
        standings = pd.DataFrame(0, index=df['HOME_TEAM_ID'].unique(), columns=['POINTS', 'POSITION'])
        outcomes = {team: [] for team in df['HOME_TEAM_ID'].unique()}  # Track outcomes (W, D, L) for each team

        # Iterate through each match week
        for matchday in sorted(df[df['SEASON'] == season]['MATCHDAY_NR'].unique()):
            # Get the matches for the current week
            matchday_matches = df[(df['SEASON'] == season) & (df['MATCHDAY_NR'] == matchday)]
            standings['POSITION'] = standings['POINTS'].rank(method='min', ascending=False)
            for index, match in matchday_matches.iterrows():
                home_team = match['HOME_TEAM_ID']
                away_team = match['AWAY_TEAM_ID']

                df.at[index, 'HOME_POSITION'] = standings.loc[home_team]['POSITION']
                df.at[index, 'AWAY_POSITION'] = standings.loc[away_team]['POSITION']
                df.at[index, 'HOME_POINTS_SUM'] = standings.loc[home_team]['POINTS']
                df.at[index, 'AWAY_POINTS_SUM'] = standings.loc[away_team]['POINTS']

                # Calculate entropy rates
                if outcomes[home_team]:
                    df.at[index, 'HOME_ENTROPY_RATE'] = calculate_entropy_rate(outcomes[home_team])
                if outcomes[away_team]:
                    df.at[index, 'AWAY_ENTROPY_RATE'] = calculate_entropy_rate(outcomes[away_team])

            # Update points after the match and record the outcomes
            for index, match in matchday_matches.iterrows():
                home_team = match['HOME_TEAM_ID']
                away_team = match['AWAY_TEAM_ID']

                standings.at[home_team, 'POINTS'] += match['HOME_POINTS']
                standings.at[away_team, 'POINTS'] += match['AWAY_POINTS']

                # Record match outcome
                if match['HOME_POINTS'] == 3:
                    outcomes[home_team].append('W')
                    outcomes[away_team].append('L')
                elif match['AWAY_POINTS'] == 3:
                    outcomes[home_team].append('L')
                    outcomes[away_team].append('W')
                else:
                    outcomes[home_team].append('D')
                    outcomes[away_team].append('D')

            # Sort standings by points
            standings.sort_values('POINTS', ascending=False, inplace=True)

            # Assign positions based on current standings
            standings['POSITION'] = range(1, len(standings) + 1)

    # Replace NaN positions with a large number (e.g., the maximum position + 1)
    df['HOME_POSITION'].fillna(len(standings) + 1, inplace=True)
    df['AWAY_POSITION'].fillna(len(standings) + 1, inplace=True)

    # Save the DataFrame with the new columns to a CSV file
    df.to_csv("bundesliga_results_with_points_positions_entropy.csv", index=False)

    print("Processing complete! Updated file saved as 'bundesliga_results_with_points_positions_entropy.csv'")


def main2():
    # Load the CSV file
    file_path = "bundesliga_results.csv"  # Replace with your file path
    df = pd.read_csv(file_path)

    # Add columns for points
    df['home_points'] = np.where(df['home_score'] > df['away_score'], 3,
                                 np.where(df['home_score'] < df['away_score'], 0, 1))
    df['away_points'] = np.where(df['away_score'] > df['home_score'], 3,
                                 np.where(df['away_score'] < df['home_score'], 0, 1))

    # Add match week
    df['match_week'] = df.groupby(['season']).cumcount() + 1

    # Initialize columns for home and away team positions and entropy rates
    df['home_position'] = np.nan
    df['away_position'] = np.nan
    df['home_entropy_rate'] = np.nan
    df['away_entropy_rate'] = np.nan

    # Iterate through each season
    for season in df['season'].unique():
        # Initialize a DataFrame to keep track of standings
        standings = pd.DataFrame(0, index=df['home_team'].unique(), columns=['points', 'position'])

        # Initialize result sequences for each team
        team_results = defaultdict(list)  # Stores the sequence of W/D/L results

        # Iterate through each match week
        for week in sorted(df[df['season'] == season]['match_week'].unique()):
            # Get the matches for the current week
            week_matches = df[(df['season'] == season) & (df['match_week'] == week)]

            # Update positions and entropy rate before the match
            for index, match in week_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']

                # Update positions before the match
                df.at[index, 'home_position'] = standings.loc[home_team]['position']
                df.at[index, 'away_position'] = standings.loc[away_team]['position']

                # Update entropy rate before the match
                home_entropy = calculate_entropy_rate_2(team_results[home_team])
                away_entropy = calculate_entropy_rate_2(team_results[away_team])
                df.at[index, 'home_entropy_rate'] = home_entropy
                df.at[index, 'away_entropy_rate'] = away_entropy

            # Update points and result sequences after the match
            for index, match in week_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']

                # Update the standings
                standings.at[home_team, 'points'] += match['home_points']
                standings.at[away_team, 'points'] += match['away_points']

                # Update result sequences
                if match['home_points'] == 3:
                    team_results[home_team].append('W')
                    team_results[away_team].append('L')
                elif match['away_points'] == 3:
                    team_results[home_team].append('L')
                    team_results[away_team].append('W')
                else:
                    team_results[home_team].append('D')
                    team_results[away_team].append('D')

            # Sort standings by points
            standings.sort_values('points', ascending=False, inplace=True)

            # Assign positions based on current standings
            standings['position'] = range(1, len(standings) + 1)

    # Replace NaN positions with a large number (e.g., the maximum position + 1)
    df['home_position'].fillna(len(standings) + 1, inplace=True)
    df['away_position'].fillna(len(standings) + 1, inplace=True)

    # Save the DataFrame with the new columns to a CSV file
    df.to_csv("bundesliga_results_with_points_positions_entropy.csv", index=False)

    print("Processing complete! Updated file saved as 'bundesliga_results_with_points_positions_entropy.csv'")


def calculate_kl_divergence(p, q):
    # Add a small value to avoid log(0)
    p = np.asarray(p) + 1e-9
    q = np.asarray(q) + 1e-9
    return np.sum(p * np.log(p / q))


# Function to calculate entropy
def calculate_entropy(distribution):
    return entropy(distribution)


# Function to calculate entropy rate
def calculate_entropy_rate(distribution):
    # Assuming a simple approximation for the entropy rate based on the distribution's entropy
    return entropy(distribution)


def main3():
    # Load your dataset
    file_path = "./bulidata.csv"  # Update this path to your file location
    df = pd.read_csv(file_path)

    # Initialize columns for points, positions, KL divergences, entropies, and entropy rates
    df['HOME_POINTS'] = 0
    df['AWAY_POINTS'] = 0
    df['HOME_POSITION'] = np.nan
    df['AWAY_POSITION'] = np.nan
    df['HOME_TOTAL_POINTS'] = 0
    df['AWAY_TOTAL_POINTS'] = 0
    df['KL_HOME_SCORED_AWAY_CONCEDED'] = np.nan
    df['KL_HOME_CONCEDED_AWAY_SCORED'] = np.nan
    df['ENTROPY_HOME_SCORED'] = np.nan
    df['ENTROPY_HOME_CONCEDED'] = np.nan
    df['ENTROPY_AWAY_SCORED'] = np.nan
    df['ENTROPY_AWAY_CONCEDED'] = np.nan
    df['ENTROPY_RATE_HOME'] = np.nan
    df['ENTROPY_RATE_AWAY'] = np.nan

    # Calculate total points and positions for each season
    for season in df['SEASON'].unique():
        season_df = df[df['SEASON'] == season].copy()
        team_points = defaultdict(int)  # Points for each team
        team_goals_scored = defaultdict(lambda: deque(maxlen=15))
        team_goals_conceded = defaultdict(lambda: deque(maxlen=15))

        for idx, row in season_df.iterrows():
            home_team = row['HOME_TEAM_ID']
            away_team = row['AWAY_TEAM_ID']
            home_goals = row['GOALS_HOME']
            away_goals = row['GOALS_AWAY']

            # Calculate points for this match
            if home_goals > away_goals:
                df.at[idx, 'HOME_POINTS'] = 3
                team_points[home_team] += 3
            elif home_goals < away_goals:
                df.at[idx, 'AWAY_POINTS'] = 3
                team_points[away_team] += 3
            else:
                df.at[idx, 'HOME_POINTS'] = 1
                df.at[idx, 'AWAY_POINTS'] = 1
                team_points[home_team] += 1
                team_points[away_team] += 1

            # Calculate total points before this match
            df.at[idx, 'HOME_TOTAL_POINTS'] = team_points[home_team] - df.at[idx, 'HOME_POINTS']
            df.at[idx, 'AWAY_TOTAL_POINTS'] = team_points[away_team] - df.at[idx, 'AWAY_POINTS']

            # Calculate positions before this match
            sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
            home_position = [i + 1 for i, (team, _) in enumerate(sorted_teams) if team == home_team][0]
            away_position = [i + 1 for i, (team, _) in enumerate(sorted_teams) if team == away_team][0]
            df.at[idx, 'HOME_POSITION'] = home_position
            df.at[idx, 'AWAY_POSITION'] = away_position

            # Update goals scored and conceded lists with rolling window of the last 15 matches
            team_goals_scored[home_team].append(home_goals)
            team_goals_conceded[home_team].append(away_goals)
            team_goals_scored[away_team].append(away_goals)
            team_goals_conceded[away_team].append(home_goals)

            # Calculate goal distributions from the last 15 matches before this match
            home_scored_distribution = np.histogram(list(team_goals_scored[home_team]), bins=range(11), density=True)[0]
            away_conceded_distribution = \
            np.histogram(list(team_goals_conceded[away_team]), bins=range(11), density=True)[0]
            home_conceded_distribution = \
            np.histogram(list(team_goals_conceded[home_team]), bins=range(11), density=True)[0]
            away_scored_distribution = np.histogram(list(team_goals_scored[away_team]), bins=range(11), density=True)[0]

            # Calculate KL divergences
            df.at[idx, 'KL_HOME_SCORED_AWAY_CONCEDED'] = calculate_kl_divergence(home_scored_distribution,
                                                                                 away_conceded_distribution)
            df.at[idx, 'KL_HOME_CONCEDED_AWAY_SCORED'] = calculate_kl_divergence(home_conceded_distribution,
                                                                                 away_scored_distribution)

            # Calculate entropies
            df.at[idx, 'ENTROPY_HOME_SCORED'] = calculate_entropy(home_scored_distribution)
            df.at[idx, 'ENTROPY_HOME_CONCEDED'] = calculate_entropy(home_conceded_distribution)
            df.at[idx, 'ENTROPY_AWAY_SCORED'] = calculate_entropy(away_scored_distribution)
            df.at[idx, 'ENTROPY_AWAY_CONCEDED'] = calculate_entropy(away_conceded_distribution)

            # Calculate entropy rates
            df.at[idx, 'ENTROPY_RATE_HOME'] = calculate_entropy_rate(home_scored_distribution)
            df.at[idx, 'ENTROPY_RATE_AWAY'] = calculate_entropy_rate(away_scored_distribution)

    # Save updated DataFrame to a new CSV
    df.to_csv("bundesliga_results_with_kl_entropy_rolling.csv", index=False)


if __name__ == "__main__":
    main3()