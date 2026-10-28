import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
def downloadHistoricalMatchData():

    url = (
        "https://raw.githubusercontent.com/"
        "martj42/international_results/master/results.csv"
    )

    df = pd.read_csv(url)
    df.date = pd.to_datetime(df.date)
    df = df.dropna()
    df = df[(df.home_score >= 0) & (df.away_score >= 0)]

    return df

# ----------------------------------------------------
# ELO
# ----------------------------------------------------
def elo_expected(r1, r2):
    return 1 / (1 + 10 ** ((r2-r1) / 400))


def update_elo(r1, r2, result, K = 30):
    e = elo_expected(r1, r2)
    return r1 + K * (result - e), r2 + K * ((1 - result) - (1 - e))


def calculate_elo(df):
    teams = set(df.home_team) | set(df.away_team)
    elo = {t: 1500 for t in teams}
    for _, m in df.sort_values("date").iterrows():
        if m.home_score > m.away_score:
            result = 1
        elif m.home_score<m.away_score:
            result = 0
        else:
            result = 0.5

        elo[m.home_team], elo[m.away_team] = update_elo(elo[m.home_team], elo[m.away_team], result)

    return elo

# ----------------------------------------------------
# TIME DECAY
# ----------------------------------------------------

def time_weight(match_date, decay_years = 10):

    age = (datetime.now() - match_date).days / 365.25
    # lambda chosen so that
    # 10 year old match = 10%

    lam = np.log(10) / decay_years

    return np.exp(-lam*age)

def calculate_strength(df,elo):

    avg_rating = np.mean(list(elo.values()))
    attack = {}
    defense = {}

    for team in elo:
        attack_values = []
        defense_values = []
        weights = []
        matches = df[(df.home_team == team) | (df.away_team == team)]

        for _, m in matches.iterrows():
            if m.home_team == team:
                gf = m.home_score
                ga = m.away_score
                opponent = m.away_team
            else:
                gf = m.away_score
                ga = m.home_score
                opponent = m.home_team
            # time decay
            tw = time_weight(m.date, decay_years = 10)
            # opponent quality
            rw = (elo[opponent] / avg_rating)
            w = tw * rw

            attack_values.append(
                gf*w
            )


            defense_values.append(
                ga*w
            )


            weights.append(w)



        # weighted averages

        avg_attack=(
            np.sum(attack_values)
            /
            np.sum(weights)
        )


        avg_defense=(
            np.sum(defense_values)
            /
            np.sum(weights)
        )


        attack[team]=avg_attack


        defense[team]=avg_defense



    return attack,defense


# ----------------------------------------------------
# EXPECTED GOALS
# ----------------------------------------------------

def expected_goals(
        team1,
        team2,
        attack,
        defense,
        elo,
        calibration,
        df):


    strength1 = (
        attack[team1]
        /
        defense[team2]
    )


    strength2 = (
        attack[team2]
        /
        defense[team1]
    )


    elo_diff = (
        elo[team1]
        -
        elo[team2]
    )


    elo_factor1=np.exp(
        0.0008*elo_diff
    )


    elo_factor2 = np.exp(-0.0008 * elo_diff)

    lambda1 = (0.55 * strength1 + 0.45 * calibration) * elo_factor1
    lambda2 = (0.55 * strength2 + 0.45 * calibration) * elo_factor2

    return lambda1, lambda2

def expected_result(M):
    results=[]
    for i in range(M.shape[0]):

        for j in range(M.shape[1]):

            results.append(
                (
                    f"{i}-{j}",
                    M[i,j]
                )
            )

    return sorted(
        results,
        key=lambda x:x[1],
        reverse=True
    )[:5]

def dixon_coles_adjust(i,j,rho=0.12):

    if i==0 and j==0:
        return 1-rho

    if i==1 and j==0:
        return 1+rho

    if i==0 and j==1:
        return 1+rho

    if i==1 and j==1:
        return 1-rho

    return 1

# ----------------------------------------------------
# SCORE MODEL
# ----------------------------------------------------

def score_matrix(l1,l2,max_goals=7):

    M=np.zeros(
        (max_goals+1,max_goals+1)
    )


    for i in range(max_goals+1):

        for j in range(max_goals+1):

            M[i,j]=(
                poisson.pmf(i,l1)
                *
                poisson.pmf(j,l2)
                *
                dixon_coles_adjust(i,j)
            )

    return M

def copula_score_matrix(
        lambda_home,
        lambda_away,
        rho=0.15,
        max_goals=7):


    M=np.zeros(
        (max_goals+1,
         max_goals+1)
    )


    for i in range(max_goals+1):

        for j in range(max_goals+1):


            # Poisson CDFs

            u=poisson.cdf(
                i,
                lambda_home
            )

            v=poisson.cdf(
                j,
                lambda_away
            )


            # transform to Gaussian

            z1=norm.ppf(u)

            z2=norm.ppf(v)



            # Gaussian copula density

            cov=[
                [1,rho],
                [rho,1]
            ]


            density=multivariate_normal.pdf(
                [z1,z2],
                mean=[0,0],
                cov=cov
            )


            correction=density


            M[i,j]=(
                poisson.pmf(i,lambda_home)
                *
                poisson.pmf(j,lambda_away)
                *
                correction
            )


    return M/M.sum()

def prediction(M):

    idx=np.unravel_index(
        np.argmax(M),
        M.shape
    )

    return idx,M[idx]



def outcomes(M):

    w=d=l=0


    for i in range(M.shape[0]):

        for j in range(M.shape[1]):

            if i>j:
                w+=M[i,j]

            elif i==j:
                d+=M[i,j]

            else:
                l+=M[i,j]


    return {
        "win":w,
        "draw":d,
        "loss":l
    }

def calculate_goal_calibration(df):

    historical_avg = (
        df.home_score.mean()
        +
        df.away_score.mean()
    ) / 2


    return historical_avg

def goal_distribution(goals):

    counts=np.bincount(
        goals,
        minlength=8
    )

    probs=counts/counts.sum()

    return probs

def combine_goal_distributions(
        attack_dist,
        defense_dist):


    P=(
        attack_dist
        *
        defense_dist
    )


    return P/P.sum()


def weighted_goal_distribution(
        team,
        df,
        elo):


    avg_rating=np.mean(
        list(elo.values())
    )


    distribution=np.zeros(8)


    matches=df[
        (df.home_team==team)
        |
        (df.away_team==team)
    ]


    for _,m in matches.iterrows():


        if m.home_team==team:

            gf=m.home_score
            opponent=m.away_team

        else:

            gf=m.away_score
            opponent=m.home_team


        w=(
            time_weight(m.date)
            *
            elo[opponent]
            /
            avg_rating
        )


        distribution[
            min(gf,7)
        ] += w


    return (
        distribution
        /
        distribution.sum()
    )


if __name__ == "__main__":

    df = downloadHistoricalMatchData()
    elo = calculate_elo(df)
    goal_calibration = calculate_goal_calibration(df)
    attack, defense = calculate_strength(df, elo)

    team1 = "England"
    team2 = "Argentina"
    l1, l2 = expected_goals(team1, team2, attack, defense, elo, goal_calibration, df)

    # M = score_matrix(l1, l2)
    # rho = np.corrcoef(
    #     home_residuals,
    #     away_residuals
    # )[0, 1]
    M = copula_score_matrix(
        l1,
        l2,
        rho=0.15
    )
    print(expected_result(M))
    print(outcomes(M))
