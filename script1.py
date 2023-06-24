import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Read the data and sort by date
df = pd.read_csv('raw_total_fight_data.csv', sep=';')
df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime format
df = df.sort_values(by='date')  # Sort the DataFrame by the 'date' column


# Calculate significant strike stats for red fighter (R) and blue fighter (B)
df['R_SIG_STR.'] = df['R_SIG_STR.'].str.extract('(\d+)')
df['B_SIG_STR.'] = df['B_SIG_STR.'].str.extract('(\d+)')
df['R_SIG_STR.'] = df['R_SIG_STR.'].astype(float)
df['B_SIG_STR.'] = df['B_SIG_STR.'].astype(float)

# Calculate takedown stats for red fighter (R) and blue fighter (B)
df['R_TD'] = df['R_TD'].str.extract('(\d+)')
df['B_TD'] = df['B_TD'].str.extract('(\d+)')
df['R_TD'] = df['R_TD'].astype(float)
df['B_TD'] = df['B_TD'].astype(float)

# Calculate significant strike differentials for red fighter (R) and blue fighter (B)
df['R_SIG_STR_DIFF'] = df['R_SIG_STR.'] - df['B_SIG_STR.']
df['B_SIG_STR_DIFF'] = df['B_SIG_STR.'] - df['R_SIG_STR.']

# Calculate takedown differentials for red fighter (R) and blue fighter (B)
df['R_TD_DIFF'] = df['R_TD'] - df['B_TD']
df['B_TD_DIFF'] = df['B_TD'] - df['R_TD']

# Define the Ranking System (ELO Ratings)
ratings = {}

# Function to update the ratings based on the outcome of a match    
def update_rating(rating, outcome, opponent_rating):
    # Update rating based on the outcome and opponent's rating
    expected_outcome = 1 / (1 + 10**((opponent_rating - rating) / 400))
    k = 32 # K-factor determines the rate of rating change
    new_rating = rating + k * (outcome - expected_outcome)
    return new_rating

# Function to calculate the autocorrelation of a series of values
def calculate_autocorrelation(values):
    values = values.dropna().astype(float)  # Convert the values to float and drop missing values
    if len(values) < 2:
        return 0.0  # Return 0 as the default autocorrelation value if there are insufficient data points

    try:
        model = ARIMA(values, order=(1, 0, 0))  # Adjust the order if needed
        model_fit = model.fit()
        _, _, r = model_fit.acorr_ljungbox(lags=1)
        return r[0]
    except (ValueError, np.linalg.LinAlgError):
        return 0.0  # Return 0 as the default autocorrelation value if there are any errors


# Step 5: Calculate Fighter Rankings
for _, row in df.iterrows():
    r_fighter = row['R_fighter']
    b_fighter = row['B_fighter']
    winner = row['Winner']

    # Get or initialize ratings for the fighters
    r_rating = ratings.get(r_fighter, 1500)
    b_rating = ratings.get(b_fighter, 1500)

    # Update ratings based on the outcome
    if winner == 'Red':
        r_rating = update_rating(r_rating, 1, b_rating)
        b_rating = update_rating(b_rating, 0, r_rating)
    elif winner == 'Blue':
        r_rating = update_rating(r_rating, 0, b_rating)
        b_rating = update_rating(b_rating, 1, r_rating)

    # Update ratings in the dictionary
    ratings[r_fighter] = r_rating
    ratings[b_fighter] = b_rating

    # Calculate autocorrelation for fighter statistics
    r_sig_str_autocorr = calculate_autocorrelation(row[['R_SIG_STR.']])
    b_sig_str_autocorr = calculate_autocorrelation(row[['B_SIG_STR.']])
    r_td_autocorr = calculate_autocorrelation(row[['R_TD']])
    b_td_autocorr = calculate_autocorrelation(row[['B_TD']])


# Sort the fighters based on their ratings and other factors to generate the rankings
fighter_rankings = []

for fighter, rating in ratings.items():
    # Gather relevant statistics and factors for the fighter
    sig_str_diff = df[df['R_fighter'] == fighter]['R_SIG_STR_DIFF'].sum() + df[df['B_fighter'] == fighter]['B_SIG_STR_DIFF'].sum()
    td_diff = df[df['R_fighter'] == fighter]['R_TD_DIFF'].sum() + df[df['B_fighter'] == fighter]['B_TD_DIFF'].sum()
    method_of_victory = (len(df[(df['R_fighter'] == fighter) & (df['win_by'] == 'KO/TKO')]) * 4) + \
                   (len(df[(df['R_fighter'] == fighter) & (df['win_by'] == 'Submission')]) * 3.5) + \
                   (len(df[(df['R_fighter'] == fighter) & (df['win_by'] == 'Decision - Unanimous')]) * 3) + \
                   (len(df[(df['R_fighter'] == fighter) & (df['win_by'] == 'Decision - Split')]) * 1) + \
                   (len(df[(df['B_fighter'] == fighter) & (df['win_by'] == 'KO/TKO')]) * 4) + \
                   (len(df[(df['B_fighter'] == fighter) & (df['win_by'] == 'Submission')]) * 3.5) + \
                   (len(df[(df['B_fighter'] == fighter) & (df['win_by'] == 'Decision - Unanimous')]) * 3) + \
                   (len(df[(df['B_fighter'] == fighter) & (df['win_by'] == 'Decision - Split')]) * 1)

    # Calculate a composite score based on the ratings and factors
    composite_score = (1.5 * rating) + (0.05 * sig_str_diff) + (0.1 * td_diff) + (0.15 * method_of_victory)

    # Add the fighter and their composite score to the rankings list
    fighter_rankings.append((fighter, composite_score))

# Sort the fighters based on the composite score
fighter_rankings.sort(key=lambda x: x[1], reverse=True)

# Print the rankings
print("MMA Male Fighter Rankings:")
for i, (fighter, score) in enumerate(fighter_rankings, start=1):
    print(f"{i}. {fighter}: {score}")
