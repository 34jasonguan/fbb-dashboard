import pandas as pd
import os
import kagglehub
import json
from datetime import datetime, timedelta

# DATA PROCESSING

path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
nba = pd.read_csv(os.path.join(path, "PlayerStatistics.csv"), low_memory=False)
nba['gameDate'] = pd.to_datetime(nba['gameDate'])

with open("player_lookup_cache.json", "r") as f:
    player_lookup = json.load(f)

with open("opponent_strength_cache.json", "r") as f:
    oss_cache = json.load(f)

nba['fp'] = (
    nba['points'] +
    nba['reboundsTotal'] +
    nba['assists'] * 2 -
    nba['turnovers'] * 2 +
    nba['fieldGoalsMade'] * 2 -
    nba['fieldGoalsAttempted'] +
    nba['blocks'] * 4 +
    nba['steals'] * 4 -
    (nba['freeThrowsAttempted'] - nba['freeThrowsMade']) +
    nba['threePointersMade']
)

cutoff = datetime.today() - timedelta(days=30)
nba_recent = nba[nba['gameDate'] >= cutoff].copy()

def get_position(row):
    key = f"{row['firstName']} {row['lastName']}"
    return player_lookup.get(key, {}).get("position")

nba_recent['position'] = nba_recent.apply(get_position, axis=1)
nba_recent = nba_recent[nba_recent['position'].notna() & (nba_recent['numMinutes'] > 0)]

def get_oss(row):
    team = row['opponentteamName']
    pos = row['position'][0]  
    return oss_cache.get(team, {}).get(pos)

nba_recent['opponent_oss'] = nba_recent.apply(get_oss, axis=1)
nba_recent = nba_recent[nba_recent['opponent_oss'].notna()]

nba_recent = nba_recent.sort_values(by=['firstName', 'lastName', 'gameDate'])
nba_recent['recent_avg_fp'] = (
    nba_recent
    .groupby(['firstName', 'lastName'])['fp']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

nba_recent['season_avg_fp'] = (
    nba_recent
    .groupby(['firstName', 'lastName'])['fp']
    .transform(lambda x: x.shift(1).expanding().mean())
)

nba_recent['avg_minutes'] = (
    nba_recent
    .groupby(['firstName', 'lastName'])['numMinutes']
    .transform(lambda x: x.shift(1).expanding().mean())
)

injuries = pd.read_csv("injury_data.csv")
injuries['DATE'] = pd.to_datetime(injuries['DATE'])
injuries['player_name'] = injuries['PLAYER'].apply(lambda x: f"{x.split(', ')[1]} {x.split(', ')[0]}")
avg_min_lookup = nba_recent[['firstName', 'lastName', 'gameDate', 'avg_minutes']]

bfi_scores = []
lookup_dict = nba_recent.set_index(['firstName', 'lastName', 'gameDate'])['avg_minutes'].to_dict()

for _, row in nba_recent.iterrows():
    game_date = row['gameDate'].date()
    team = row['playerteamName']
    position = row['position'][0]

    teammates = nba_recent[
        (nba_recent['playerteamName'] == team) &
        (nba_recent['position'].str.startswith(position))
    ]
    teammate_names = set(teammates.apply(lambda r: f"{r['firstName']} {r['lastName']}", axis=1))

    injured = injuries[
        (injuries['TEAM'] == team) &
        (injuries['STATUS'] == 'Out') &
        (injuries['DATE'] == game_date)
    ]

    injured_names = set(injured['player_name'])

    total_bfi = 0.0
    for name in injured_names:
        parts = name.split(' ')
        if len(parts) < 2:
            continue
        first = parts[0]
        last = ' '.join(parts[1:])
        key = (first, last, row['gameDate'])
        if key in lookup_dict and name in teammate_names:
            total_bfi += lookup_dict[key]

    bfi_scores.append(total_bfi)

nba_recent['bfi'] = bfi_scores

model_data = nba_recent[[
    'firstName', 'lastName', 'playerteamName', 'gameDate', 'numMinutes',
    'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi', 'fp'
]].dropna()

model_data.to_csv("model_training_data.csv", index=False)