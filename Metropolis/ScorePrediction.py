import pandas as pd
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def roundScore(score):
    return math.floor(score) + 1 if score - math.floor(score) >= 0.75 else math.floor(score)


# Function to predict match result
def predict_match_result(encoder, model_home, model_away, season, matchday, home_team, away_team):
    match_encoded = encoder.transform([[season, matchday, home_team, away_team]]).toarray()
    home_score_pred = model_home.predict(match_encoded)
    away_score_pred = model_away.predict(match_encoded)
    return home_score_pred[0][0], away_score_pred[0][0]


def predict_match_result_gan(encoder, generator, discriminator, latent_dim, season, matchday, home_team, away_team):
    match_encoded = encoder.transform([[season, matchday, home_team, away_team]]).toarray()
    noise = np.random.normal(0, 1, (1, latent_dim))
    predicted_scores = generator.predict(noise)
    combined_input = np.hstack([match_encoded, predicted_scores])
    validity = discriminator.predict(combined_input)
    return predicted_scores[0], validity[0][0]

# Generator
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=latent_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Predicts home and away scores
    return model


# Discriminator
def build_discriminator(X_train):
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1] + 2, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(128, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(1, activation='sigmoid'))  # Outputs probability of being real
    return model


def gan(X_train, y_train):
    # GAN setup
    latent_dim = 10

    # Build and compile the discriminator
    discriminator = build_discriminator(X_train)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Build the generator
    generator = build_generator(latent_dim)

    # GAN model (combined generator and discriminator)
    z = Input(shape=(latent_dim,))
    match_input = Input(shape=(X_train.shape[1],))
    fake_scores = generator(z)
    combined_input = concatenate([match_input, fake_scores])
    discriminator.trainable = False
    validity = discriminator(combined_input)

    # Define the combined model
    gan = Model([z, match_input], validity)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    # Training
    epochs = 10000
    batch_size = 64

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        matches = X_train[idx]
        real_scores = y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_scores = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(np.hstack([matches, real_scores]), np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(np.hstack([matches, fake_scores]), np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch([noise, matches], np.ones((batch_size, 1)))

        if epoch % 1000 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    return gan, generator, discriminator, latent_dim


def main2():
    df = pd.read_csv('./bulidata.csv')
    # Encode team names
    encoder = OneHotEncoder()
    teams_encoded = encoder.fit_transform(df[['SEASON', 'MATCHDAY_NR', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']]).toarray()

    # Define features and targets
    X = teams_encoded
    y = df[['GOALS_HOME', 'GOALS_AWAY']].values
    y_home = df['GOALS_HOME']
    y_away = df['GOALS_AWAY']

    indices = np.arange(len(df))
    # Split data into training and testing sets
    # X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, train_indices, test_indices = train_test_split(
    #     X, y_home, y_away, indices, test_size=0.025, shuffle=False, stratify=None
    # )
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.025, shuffle=False, stratify=None)
    # Define the neural network model for home team score prediction
    # model_home = Sequential()
    # model_home.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    # model_home.add(BatchNormalization())
    # model_home.add(Dropout(0.1))
    # model_home.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))
    # model_home.add(BatchNormalization())
    # model_home.add(Dropout(0.1))
    # model_home.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    # model_home.add(BatchNormalization())
    # model_home.add(Dropout(0.1))
    # model_home.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    # model_home.add(BatchNormalization())
    # model_home.add(Dense(1, activation='linear'))  # Output layer for home team score

    # Compile the model
    # model_home.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    # Train the model for home team score
    # model_home.fit(X_train, y_home_train, epochs=50, batch_size=32, validation_split=0.2)

    # Define the neural network model for away team score prediction
    # model_away = Sequential()
    # model_away.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    # model_away.add(BatchNormalization())
    # model_away.add(Dropout(0.1))
    # model_away.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    # model_away.add(BatchNormalization())
    # model_away.add(Dropout(0.1))
    # model_away.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    # model_away.add(BatchNormalization())
    # model_away.add(Dropout(0.1))
    # model_away.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    # model_away.add(BatchNormalization())
    # model_away.add(Dense(1, activation='linear'))  # Output layer for away team score

    # Compile the model
    # model_away.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    # Train the model for away team score
    # model_away.fit(X_train, y_away_train, epochs=50, batch_size=32, validation_split=0.2)

    ganResult, generator, discriminator, latent_dim = gan(X_train, y_train)

    # Example prediction
    predictions = []

    # Get the test set indices
    # test_indices = np.array(df.index)[X_test[:, 0] != 0]  # This is a trick to get the correct test indices

    succesCount = 0

    for i in range(X_test.shape[0]):
        index = test_indices[i]
        season = df.iloc[index]['SEASON']
        matchday = df.iloc[index]['MATCHDAY_NR']
        home_team = df.iloc[index]['HOME_TEAM_ID']
        away_team = df.iloc[index]['AWAY_TEAM_ID']
        home_team_name = df.iloc[index]['HOME_TEAM']
        away_team_name = df.iloc[index]['AWAY_TEAM']
        home_goals = df.iloc[index]['GOALS_HOME']
        away_goals = df.iloc[index]['GOALS_AWAY']

        # predicted_home_score, predicted_away_score = predict_match_result(home_team, away_team)
        # predicted_home_score, predicted_away_score = (
        #     predict_match_result(encoder, model_home, model_away, season, matchday, home_team, away_team))
        predicted_home_score, predicted_away_score, validity = (
            predict_match_result_gan(encoder, generator, discriminator, latent_dim, season, matchday, home_team, away_team))

        predictions.append((home_team, away_team, predicted_home_score, predicted_away_score))
        predictedResult = 'home' if roundScore(predicted_home_score) > roundScore(predicted_away_score) else 'draw' if roundScore(predicted_home_score) == roundScore(predicted_away_score) else 'away'
        realResult = 'home' if home_goals > away_goals else 'draw' if home_goals == away_goals else 'away'
        succesCount = succesCount + 1 if realResult == predictedResult else succesCount
        print(f'Predicted score: {home_team_name} {predicted_home_score:.2f} - {predicted_away_score:.2f} {away_team_name}, real score {home_team_name} {home_goals:.2f} - {away_goals:.2f} {away_team_name}, predicted result: {predictedResult}, real result: {realResult}, validity: {validity}')

    # Optionally, convert predictions to a DataFrame for better presentation

    print(f'Succesfull tips: {succesCount} out of {X_test.shape[0]}, success rate is: {succesCount / X_test.shape[0]}')

    predictions_df = pd.DataFrame(predictions,
                                  columns=['Home Team', 'Away Team', 'Predicted Home Score', 'Predicted Away Score'])
    print(predictions_df)
    # df.map(lambda x:
    #     predicted_home_score, predicted_away_score = (
    #     predict_match_result(encoder, model_home, model_away, season, matchday, home_team, away_team))
    #     print(f'Predicted score: {home_team} {predicted_home_score:.2f} - {predicted_away_score:.2f} {away_team}')
    # )
    # season = 2022
    # matchday = 24
    # home_team = 134
    # away_team = 6
    # predicted_home_score, predicted_away_score = (
    #     predict_match_result(encoder, model_home, model_away, season, matchday, home_team, away_team))
    # print(f'Predicted score: {home_team} {predicted_home_score:.2f} - {predicted_away_score:.2f} {away_team}')


def main():
    df = pd.read_csv('./bulidata.csv')
    df.head()

    # Example: Load your preprocessed data
    # Assume df is a pandas DataFrame with features and target columns
    # features: team statistics, previous match results, player information
    # target: home_team_score, away_team_score

    # Define features and targets
    X = df[['SEASON','MATCHDAY_NR','HOME_TEAM_ID', 'AWAY_TEAM_ID']]
    y_home = df['GOALS_HOME']
    y_away = df['GOALS_AWAY']

    # Split data into training and testing sets
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model_home.add(BatchNormalization())
    model_home.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for home team score

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model for home team score
    model.fit(X_train, y_home_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    home_score_pred = model.predict(X_test)
    home_score_mae = model.evaluate(X_test, y_home_test)

    print("Home Score Prediction MAE:", home_score_mae)

    # Repeat the process for away team score
    model_away = Sequential()
    model_away.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model_away.add(Dense(64, activation='relu'))
    model_away.add(Dense(32, activation='relu'))
    model_away.add(Dense(1, activation='linear'))  # Output layer for away team score

    # Compile the model
    model_away.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model for away team score
    model_away.fit(X_train, y_away_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    away_score_pred = model_away.predict(X_test)
    away_score_mae = model_away.evaluate(X_test, y_away_test)

    print("Away Score Prediction MAE:", away_score_mae)

    # Combine home and away predictions for final score prediction
    predicted_scores = pd.DataFrame({
        'home_score_pred': home_score_pred.flatten(),
        'away_score_pred': away_score_pred.flatten()
    })

    print(predicted_scores.head())


if __name__ == "__main__":
    main2()

