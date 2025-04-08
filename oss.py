import pandas as pd
import json
from datetime import datetime, timedelta
import kagglehub
import os

CACHE_FILE = "opponent_strength_cache.json"
BASE_PATH = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
STATS_CSV = os.path.join(BASE_PATH, "PlayerStatistics.csv")

nba = pd.read_csv(STATS_CSV, low_memory=False)
nba['gameDate'] = pd.to_datetime(nba['gameDate'])
nba = nba[nba['gameDate'] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
nba = nba[nba["numMinutes"] > 0]

with open("player_lookup_cache.json", "r") as f:
    player_lookup = json.load(f)

def get_cached_position(row):
    key = f"{row['firstName']} {row['lastName']}"
    return player_lookup.get(key, {}).get("position")

nba['position'] = nba.apply(get_cached_position, axis=1)
nba = nba[nba['position'].notna()]
nba = nba.assign(position=nba['position'].str.split("-"))
nba = nba.explode('position')

grouped = (
    nba.groupby(['opponentteamName', 'position'])['points']
    .mean()
    .reset_index()
    .rename(columns={'points': 'avg_fp_allowed'})
)

pivot = grouped.pivot(index='opponentteamName', columns='position', values='avg_fp_allowed').fillna(0)
oss_dict = pivot.to_dict(orient='index')

with open(CACHE_FILE, "w") as f:
    json.dump(oss_dict, f, indent=2)

print(f"Opponent strength cache saved with {len(oss_dict)} teams → {CACHE_FILE}")
