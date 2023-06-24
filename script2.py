import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from trueskill import Rating, rate_1vs1

# Load and prepare the data
data = pd.read_csv('raw_total_fight_data.csv', delimiter=";", parse_dates=['date'])
data = data.sort_values(by='date')

# Extract numerical values from string
for col in ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.', 'R_TD', 'B_TD']:
    data[col] = data[col].apply(lambda x: int(x.split(' of ')[0]))

for col in ['R_SIG_STR_pct', 'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']:
    data[col] = data[col].apply(lambda x: int(x.rstrip('%')) if x != "---" else np.nan)

# Convert time string to seconds
for col in ['R_CTRL', 'B_CTRL']:
    data[col] = data[col].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if x != "--" else 0)

# Define scoring dictionary for outcomes
win_types = {
    'KO/TKO': 4,
    'Submission': 3,
    'Decision - Unanimous': 2,
    'Decision - Split': 1,
    'Other': 0
}
data['win_by'] = data['win_by'].map(win_types)

# Prepare rating system
rating_dict = {}

# Define factors
factors = ['R_SIG_STR.', 'B_SIG_STR.', 'R_SIG_STR_pct', 'B_SIG_STR_pct', 'R_TD', 'B_TD', 'R_CTRL', 'B_CTRL', 'R_KD', 'B_KD', 'win_by']

# Drop rows with NaN in the target
data = data.dropna(subset=['Winner'])

# Machine learning model
X = data[factors]
y = data['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Iterate over each row in the dataframe
for index, row in data.iterrows():
    winner = row['Winner']
    loser = row['R_fighter'] if row['B_fighter'] == winner else row['B_fighter']

    if winner not in rating_dict:
        rating_dict[winner] = Rating()
    if loser not in rating_dict:
        rating_dict[loser] = Rating()

    rating_winner = rating_dict[winner]
    rating_loser = rating_dict[loser]

    new_rating_winner, new_rating_loser = rate_1vs1(rating_winner, rating_loser)

    rating_dict[winner] = new_rating_winner
    rating_dict[loser] = new_rating_loser

# Print out the ratings sorted by rating
for fighter, rating in sorted(rating_dict.items(), key=lambda x: x[1].mu, reverse=True):
    print(fighter, rating)