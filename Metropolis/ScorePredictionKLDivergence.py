import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import rel_entr
import matplotlib.pyplot as plt
def load_data():
    df = pd.read_csv('./bulidata.csv')
    return df


def calculate_goal_distribution(df, team, location, matches):
    if location == 'home':
        scored_goals = df[df['HOME_TEAM_ID'] == team].iloc[:matches]['GOALS_HOME']
        conceded_goals = df[df['HOME_TEAM_ID'] == team].iloc[:matches]['GOALS_AWAY']
    elif location == 'away':
        scored_goals = df[df['AWAY_TEAM_ID'] == team].iloc[:matches]['GOALS_AWAY']
        conceded_goals = df[df['AWAY_TEAM_ID'] == team].iloc[:matches]['GOALS_HOME']
    else:
        raise ValueError("Location must be either 'home' or 'away'")

    scored_goal_counts = scored_goals.value_counts().sort_index()
    conceded_goal_counts = conceded_goals.value_counts().sort_index()

    total_matches = scored_goals.count()

    scored_distribution = scored_goal_counts / total_matches
    conceded_distribution = conceded_goal_counts / total_matches

    return scored_distribution, conceded_distribution


def calculate_distributions(df, m):
    for team, dist in distributions.items():
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

        # Home scored goals distribution
        ax[0, 0].bar(dist['home_scored'].index, dist['home_scored'].values, color='blue', alpha=0.7)
        ax[0, 0].set_title(f'{team} - Home Scored Goals Distribution')
        ax[0, 0].set_xlabel('Goals')
        ax[0, 0].set_ylabel('Proportion')

        # Home conceded goals distribution
        ax[0, 1].bar(dist['home_conceded'].index, dist['home_conceded'].values, color='orange', alpha=0.7)
        ax[0, 1].set_title(f'{team} - Home Conceded Goals Distribution')
        ax[0, 1].set_xlabel('Goals')

        # Away scored goals distribution
        ax[1, 0].bar(dist['away_scored'].index, dist['away_scored'].values, color='green', alpha=0.7)
        ax[1, 0].set_title(f'{team} - Away Scored Goals Distribution')
        ax[1, 0].set_xlabel('Goals')
        ax[1, 0].set_ylabel('Proportion')

        # Away conceded goals distribution
        ax[1, 1].bar(dist['away_conceded'].index, dist['away_conceded'].values, color='red', alpha=0.7)
        ax[1, 1].set_title(f'{team} - Away Conceded Goals Distribution')
        ax[1, 1].set_xlabel('Goals')

        plt.suptitle(f'{team} Goals Distribution')
        plt.show()



def kl_divergence(p, q):
    return entropy(p, q)


def compute_kl_divergence(p, q):
    # Ensure both distributions have the same support
    all_indices = np.union1d(p.index, q.index)
    p = p.reindex(all_indices, fill_value=0)
    q = q.reindex(all_indices, fill_value=0)

    # Compute KL divergence
    kl_divergence = np.sum(rel_entr(p, q))
    return kl_divergence


def calculate_kl_divergences(df, m):
    teams = df['HOME_TEAM_ID'].unique()
    kl_divergences = []

    for i in range(len(df)):
        match = df.iloc[i]
        home_team = match['HOME_TEAM_ID']
        away_team = match['AWAY_TEAM_ID']

        home_distribution = calculate_goal_distribution(df, home_team, 'home', min(i, m))
        away_distribution = calculate_goal_distribution(df, away_team, 'away', min(i, m))

        kl_div = compute_kl_divergence(home_distribution, away_distribution)
        kl_divergences.append(
            (match['MATCH_DATE'], home_team, away_team, match['GOALS_HOME'], match['GOALS_AWAY'], kl_div))

    return pd.DataFrame(kl_divergences, columns=['date', 'home_team', 'away_team', 'home_team_score', 'away_team_score',
                                                 'kl_divergence'])


def plot_goal_distributions(distributions):
    for team, dist in distributions.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Home goals distribution
        ax[0].bar(dist['home'].index, dist['home'].values, color='blue', alpha=0.7)
        ax[0].set_title(f'{team} - Home Goals Distribution')
        ax[0].set_xlabel('Goals')
        ax[0].set_ylabel('Proportion')

        # Away goals distribution
        ax[1].bar(dist['away'].index, dist['away'].values, color='red', alpha=0.7)
        ax[1].set_title(f'{team} - Away Goals Distribution')
        ax[1].set_xlabel('Goals')

        plt.suptitle(f'{team} Goals Distribution')
        plt.show()


def plot_goals_vs_kl_divergence(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot home goals vs KL divergence
    ax[0].scatter(df['home_team_score'], df['kl_divergence'], color='blue', alpha=0.5)
    ax[0].set_title('Home Goals vs KL Divergence')
    ax[0].set_xlabel('Home Goals')
    ax[0].set_ylabel('KL Divergence')

    # Plot away goals vs KL divergence
    ax[1].scatter(df['away_team_score'], df['kl_divergence'], color='red', alpha=0.5)
    ax[1].set_title('Away Goals vs KL Divergence')
    ax[1].set_xlabel('Away Goals')
    ax[1].set_ylabel('KL Divergence')

    plt.tight_layout()
    plt.show()


def main2():
    matches = 15
    df = load_data()
    #['SEASON', 'MATCHDAY_NR', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    # Step 1: Calculate goal distributions
    teams = df['HOME_TEAM_ID'].unique()

    # Step 2: Compute distributions
    distributions = {}
    for team in teams:
        home_scored_dist, home_conceded_dist = calculate_goal_distribution(df, team, 'home', matches)
        away_scored_dist, away_conceded_dist = calculate_goal_distribution(df, team, 'away', matches)
        distributions[team] = {
            'home_scored': home_scored_dist,
            'home_conceded': home_conceded_dist,
            'away_scored': away_scored_dist,
            'away_conceded': away_conceded_dist
        }
    # plot_goal_distributions(distributions)

    # Step 3: Calculate KL divergence
    kl_divergences = {}
    for team, dist in distributions.items():
        kl_div = compute_kl_divergence(dist['home'], dist['away'])
        kl_divergences[team] = kl_div

    # Output the KL divergences
    for team, kl_div in kl_divergences.items():
        print(f"KL Divergence for {team}: {kl_div:.4f}")

    kl_divergences_df = calculate_kl_divergences(df, matches)

    plot_goals_vs_kl_divergence(kl_divergences_df)


def entropy_rate(outcomes):
    # Create transition matrix (3x3 for Win, Draw, Loss)
    transition_matrix = np.zeros((3, 3))

    # Map outcomes to indices: 1 -> 0, 0 -> 1, -1 -> 2
    mapped_outcomes = outcomes + 1

    # Calculate transition frequencies
    for (i, j) in zip(mapped_outcomes[:-1], mapped_outcomes[1:]):
        transition_matrix[i, j] += 1

    # Convert to probabilities (normalize each row)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # Calculate the stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])
    stationary = stationary / stationary.sum()

    # Calculate the entropy rate
    entropy_rate = -np.sum(
        stationary * np.sum(transition_matrix * np.log2(transition_matrix, where=transition_matrix > 0), axis=1))

    print(f"Entropy Rate: {entropy_rate:.4f} bits per match")

    return entropy_rate


if __name__ == "__main__":
    main2()