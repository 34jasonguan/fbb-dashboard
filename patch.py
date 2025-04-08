from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
import json
import pandas as pd
import time

CACHE_FILE = "player_lookup_cache.json"

with open(CACHE_FILE, "r") as f:
    player_lookup = json.load(f)

active_players = players.get_active_players()
active_df = pd.DataFrame(active_players)

def simplify_position(pos_str):
    """Map full position strings to G, F, C."""
    pos_str = pos_str.upper()
    parts = []
    if "GUARD" in pos_str:
        parts.append("G")
    if "FORWARD" in pos_str:
        parts.append("F")
    if "CENTER" in pos_str:
        parts.append("C")
    return "-".join(parts) if parts else None

updated = False
for full_name, info in player_lookup.items():
    if not info.get("position"):
        match = active_df[active_df["full_name"] == full_name]
        if not match.empty:
            player_id = match.iloc[0]["id"]
            try:
                pos_raw = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0].loc[0, "POSITION"]
                pos_clean = simplify_position(pos_raw)
                if pos_clean:
                    player_lookup[full_name]["position"] = pos_clean
                    updated = True
                    print(f"Updated {full_name}: {pos_clean}")
            except Exception as e:
                print(f"Failed to fetch position for {full_name}: {e}")
            time.sleep(1.2)

if updated:
    with open(CACHE_FILE, "w") as f:
        json.dump(player_lookup, f, indent=2)
    print("Cache updated.")
else:
    print("No missing positions found or updated.")
