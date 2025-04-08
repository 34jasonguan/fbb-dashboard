import json
import os
import pandas as pd
import kagglehub

CACHE_FILE = "player_lookup_cache.json"
BASE_PATH = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
STATS_CSV = os.path.join(BASE_PATH, "PlayerStatistics.csv")
PLAYERS_CSV = os.path.join(BASE_PATH, "Players.csv")

nba = pd.read_csv(STATS_CSV, low_memory=False)
players_df = pd.read_csv(PLAYERS_CSV)
nba = nba[nba['gameDate'] >= "2024-10-22"]

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

all_names = nba[['firstName', 'lastName']].drop_duplicates()

def get_position(row):
    roles = []
    if row.get("Guard") == True: roles.append("G")
    if row.get("Forward") == True: roles.append("F")
    if row.get("Center") == True: roles.append("C")
    return "-".join(roles) if roles else None

players_df["position"] = players_df.apply(get_position, axis=1)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        player_lookup = json.load(f)
else:
    player_lookup = {}

updated = False
for _, row in all_names.iterrows():
    full_name = f"{row['firstName']} {row['lastName']}"
    if full_name in player_lookup and all(player_lookup[full_name].get(k) for k in ['player_id', 'image_url', 'position', 'season_fp', 'avg_fp']):
        continue

    match = players_df[(players_df['firstName'] == row['firstName']) & (players_df['lastName'] == row['lastName'])]
    if not match.empty:
        pid = str(match.iloc[0].get("personId"))
        position = match.iloc[0].get("position")
        image_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"

        player_stats = nba[(nba['firstName'] == row['firstName']) & (nba['lastName'] == row['lastName'])]
        total_fp = round(player_stats['fp'].sum(), 1)
        games_played = len(player_stats)
        avg_fp = round(total_fp / games_played, 1) if games_played > 0 else 0.0

        player_lookup[full_name] = {
            "player_id": pid,
            "position": position,
            "image_url": image_url,
            "season_fp": total_fp,
            "avg_fp": avg_fp
        }
        updated = True
        print(f"Cached {full_name}: {pid}, {position}, FP={total_fp}, AVG={avg_fp}")
    else:
        print(f"Skipped {full_name} — not found in Players.csv")

if updated:
    with open(CACHE_FILE, "w") as f:
        json.dump(player_lookup, f, indent=2)
    print(f"Cache saved with {len(player_lookup)} players.")
else:
    print("All players already cached.")
