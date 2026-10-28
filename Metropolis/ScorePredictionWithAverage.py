import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "./bulidata.csv"  # Replace with your file path
df = pd.read_csv(file_path)


# Function to calculate average scored and conceded goals over the last m matches
def calculate_avg_goals(df, team_name, last_m_matches=15, home=True, scored=True):
    if home:
        if scored:
            matches = df[(df['HOME_TEAM_ID'] == team_name)].tail(last_m_matches)
            return matches['GOALS_HOME'].mean()
        else:
            matches = df[(df['HOME_TEAM_ID'] == team_name)].tail(last_m_matches)
            return matches['GOALS_AWAY'].mean()
    else:
        if scored:
            matches = df[(df['AWAY_TEAM_ID'] == team_name)].tail(last_m_matches)
            return matches['GOALS_AWAY'].mean()
        else:
            matches = df[(df['AWAY_TEAM_ID'] == team_name)].tail(last_m_matches)
            return matches['GOALS_HOME'].mean()


def plot_goal_distribution(df, team_type='HOME'):
    real_goal_column = f'GOALS_{team_type}'
    estimated_goal_column = f'estimated_{team_type}_goals'

    unique_real_goals = sorted(df[real_goal_column].unique())

    # Plot the distributions of estimated goals for each real goal value
    for real_goals in unique_real_goals:
        subset = df[df[real_goal_column] == real_goals]

        plt.figure(figsize=(10, 6))
        sns.histplot(subset[estimated_goal_column], kde=True, color='blue', bins=20, alpha=0.6)
        plt.title(
            f'Distribution of Estimated {team_type.capitalize()} Goals for Real {team_type.capitalize()} Goals = {real_goals}')
        plt.xlabel(f'Estimated {team_type.capitalize()} Goals')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def main():
    # Iterate through each row (match) in the DataFrame
    for index, row in df.iterrows():
        season = row['SEASON']
        matchday = row['MATCHDAY_NR']
        home_team = row['HOME_TEAM_ID']
        away_team = row['AWAY_TEAM_ID']

        # Filter the DataFrame to include only matches before the current match
        past_matches_df = df[(df['SEASON'] == season) & (df['MATCHDAY_NR'] < matchday)]

        # Calculate average goals scored and conceded for both teams over the last 15 matches
        avg_home_scored = calculate_avg_goals(past_matches_df, home_team, last_m_matches=15, home=True, scored=True)
        avg_away_conceded = calculate_avg_goals(past_matches_df, away_team, last_m_matches=15, home=False, scored=False)

        avg_away_scored = calculate_avg_goals(past_matches_df, away_team, last_m_matches=15, home=False, scored=True)
        avg_home_conceded = calculate_avg_goals(past_matches_df, home_team, last_m_matches=15, home=True, scored=False)

        # Estimate home and away goals
        estimated_home_goals = (avg_home_scored + avg_away_conceded) / 2
        estimated_away_goals = (avg_away_scored + avg_home_conceded) / 2

        # Add the estimates to the DataFrame
        df.at[index, 'estimated_HOME_goals'] = estimated_home_goals
        df.at[index, 'estimated_AWAY_goals'] = estimated_away_goals

    # Save the DataFrame with the new estimated goals columns to a CSV file
    df.to_csv("bundesliga_results_with_estimated_goals.csv", index=False)

    # Plot the estimated goals against the real goals
    plot_goal_distribution(df, team_type='HOME')
    plot_goal_distribution(df, team_type='AWAY')
    # Home team plot
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df['GOALS_HOME'], df['estimated_home_goals'], color='blue', alpha=0.5)
    # plt.plot([df['GOALS_HOME'].min(), df['GOALS_HOME'].max()],
    #          [df['GOALS_HOME'].min(), df['GOALS_HOME'].max()],
    #          color='red', linestyle='--')
    # plt.title('Estimated vs Real Home Goals')
    # plt.xlabel('Real Home Goals')
    # plt.ylabel('Estimated Home Goals')
    # plt.grid(True)
    # plt.show()
    #
    # # Away team plot
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df['GOALS_AWAY'], df['estimated_away_goals'], color='green', alpha=0.5)
    # plt.plot([df['GOALS_AWAY'].min(), df['GOALS_AWAY'].max()],
    #          [df['GOALS_AWAY'].min(), df['GOALS_AWAY'].max()],
    #          color='red', linestyle='--')
    # plt.title('Estimated vs Real Away Goals')
    # plt.xlabel('Real Away Goals')
    # plt.ylabel('Estimated Away Goals')
    # plt.grid(True)
    # plt.show()

    print("Processing and plotting complete!")


if __name__ == "__main__":
    main()