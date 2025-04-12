import pandas as pd
import json
import mlflow.sklearn
import kagglehub
import os
from datetime import date, datetime, timedelta

path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")

def get_tomorrows_predictions():
    # Load caches and mappings
    with open("player_lookup_cache.json", "r") as f:
        player_lookup = json.load(f)
    with open("opponent_strength_cache.json", "r") as f:
        oss_cache = json.load(f)
    with open("teams.json", "r") as f:
        team_mappings = json.load(f)

    teamid_to_simple = {entry["teamId"]: entry["simpleName"] for entry in team_mappings}
    simple_to_teamid = {entry["simpleName"]: entry["teamId"] for entry in team_mappings}

    # Load schedule and figure out tomorrow (skewed by 16 hours)
    schedule = pd.read_csv(os.path.join(path, "LeagueSchedule24_25.csv"))
    schedule['gameDateTimeEst'] = pd.to_datetime(schedule['gameDateTimeEst'])

    adjusted_now = datetime.now() - timedelta(hours=16)
    # tomorrow = adjusted_now.date() + timedelta(days=1)
    tomorrow = date(2025, 4, 13) # last day of the season, reset this once the new season starts

    tomorrow_games = schedule[schedule['gameDateTimeEst'].dt.date == tomorrow]
    team_ids = set(tomorrow_games['hometeamId']).union(set(tomorrow_games['awayteamId']))
    team_names = {teamid_to_simple[tid] for tid in team_ids if tid in teamid_to_simple}

    # Load historical player game data
    stats = pd.read_csv("model_training_data.csv")
    stats['gameDate'] = pd.to_datetime(stats['gameDate'])

    # Only consider players from teams playing tomorrow
    latest_games = stats[stats['playerteamName'].isin(team_names)].copy()
    latest_games.sort_values(by=['firstName', 'lastName', 'gameDate'], inplace=True)

    # Get last 5 games per player
    last_5 = latest_games.groupby(['firstName', 'lastName']).tail(5).copy()

    # Remove players who logged 0 minutes in 2+ of their last 5 games
    last_5.loc[:, 'played'] = last_5['numMinutes'] > 0
    play_counts = last_5.groupby(['firstName', 'lastName'])['played'].sum().reset_index()
    active_players = play_counts[play_counts['played'] >= 4][['firstName', 'lastName']]

    # Keep only active players' most recent entry
    df = latest_games.drop_duplicates(subset=['firstName', 'lastName'], keep='last')
    df = pd.merge(df, active_players, on=['firstName', 'lastName'], how='inner').copy()

    # Compute average stats
    five_game_avg_fp = (
        last_5.groupby(['firstName', 'lastName'])['fp']
        .mean().reset_index().rename(columns={'fp': 'recent_avg_fp'})
    )
    season_avg_fp = (
        latest_games.groupby(['firstName', 'lastName'])['fp']
        .mean().reset_index().rename(columns={'fp': 'season_avg_fp'})
    )
    avg_minutes = (
        last_5.groupby(['firstName', 'lastName'])['numMinutes']
        .mean().reset_index().rename(columns={'numMinutes': 'numMinutes'})
    )

    # Merge in averages
    df.drop(columns=['numMinutes', 'recent_avg_fp', 'season_avg_fp'], errors='ignore', inplace=True)
    df = df.merge(five_game_avg_fp, on=['firstName', 'lastName'], how='left')
    df = df.merge(season_avg_fp, on=['firstName', 'lastName'], how='left')
    df = df.merge(avg_minutes, on=['firstName', 'lastName'], how='left')

    # Determine opponent team
    def find_opponent(team_name):
        team_id = simple_to_teamid.get(team_name)
        row = tomorrow_games[
            (tomorrow_games['hometeamId'] == team_id) | (tomorrow_games['awayteamId'] == team_id)
        ]
        if not row.empty:
            home = row.iloc[0]['hometeamId']
            away = row.iloc[0]['awayteamId']
            opp_id = away if home == team_id else home
            return teamid_to_simple.get(opp_id)
        return None

    df.loc[:, 'opponentteamName'] = df['playerteamName'].apply(find_opponent)

    # Add opponent OSS
    def get_oss(row):
        key = f"{row['firstName']} {row['lastName']}"
        pos = player_lookup.get(key, {}).get("position")
        team = row["opponentteamName"]
        if pos and team:
            return oss_cache.get(team, {}).get(pos[0])
        return None

    df.loc[:, "opponent_oss"] = df.apply(get_oss, axis=1)
    df = df[df["opponent_oss"].notna()].copy()

    # Add image, position, and OSS message
    def get_image_url(row):
        name = f"{row['firstName']} {row['lastName']}"
        return player_lookup.get(name, {}).get("image_url")

    def get_position(row):
        name = f"{row['firstName']} {row['lastName']}"
        return player_lookup.get(name, {}).get("position")
    
    # compute oss rankings for message
    oss_rankings = {}
    for pos in ['G', 'F', 'C']:
        team_values = [
            (team, values.get(pos))
            for team, values in oss_cache.items()
            if values.get(pos) is not None
        ]
        team_values = sorted(team_values, key=lambda x: x[1], reverse=True)  # Higher = easier matchup
        oss_rankings[pos] = {team: rank + 1 for rank, (team, _) in enumerate(team_values)}

    def get_oss_message(row):
        pos = row.get("position")
        team = row.get("opponentteamName")
        oss = row.get("opponent_oss")

        position_key = pos[0] if isinstance(pos, str) and len(pos) > 0 else None
        if position_key and team and oss:
            rank = oss_rankings.get(position_key, {}).get(team)
            if rank:
                return f"{team} allow {rank}ᵗʰ highest FP to {pos}s"
            else:
                return f"vs. {team}"
        return ""
    
    df.loc[:, "image_url"] = df.apply(get_image_url, axis=1)
    df.loc[:, "position"] = df.apply(get_position, axis=1)
    df["oss_message"] = df.apply(get_oss_message, axis=1)

    # Final features
    df.loc[:, "bfi"] = 0.0  # Placeholder
    features = ['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']
    df.dropna(subset=features, inplace=True)

    X = df[features]

    # Predict
    model = mlflow.sklearn.load_model("mlruns/0/a5cefbc637fe4c24b6d693e303f11826/artifacts/model")
    df.loc[:, "predicted_fp"] = model.predict(X)
    df.loc[:, "diff_from_season_avg"] = df["predicted_fp"] - df["season_avg_fp"]

    # Output
    columns = [
        "firstName", "lastName", "playerteamName", "opponentteamName",
        "predicted_fp", "season_avg_fp", "diff_from_season_avg",
        "image_url", "oss_message"
    ]
    top_predicted = df.sort_values(by="predicted_fp", ascending=False).head(3)[columns]
    top_booms = df.sort_values(by="diff_from_season_avg", ascending=False).head(3)[columns]

    return top_predicted, top_booms

# def get_tomorrows_predictions():
#     # Load caches and mappings
#     with open("player_lookup_cache.json", "r") as f:
#         player_lookup = json.load(f)
#     with open("opponent_strength_cache.json", "r") as f:
#         oss_cache = json.load(f)
#     with open("teams.json", "r") as f:
#         team_mappings = json.load(f)

#     # Build teamId -> simpleName mapping
#     teamid_to_simple = {entry["teamId"]: entry["simpleName"] for entry in team_mappings}
#     simple_to_teamid = {entry["simpleName"]: entry["teamId"] for entry in team_mappings}

#     # Load schedule and filter for tomorrow's games
#     schedule = pd.read_csv(os.path.join(path, "LeagueSchedule24_25.csv"), low_memory=False)
#     schedule['gameDateTimeEst'] = pd.to_datetime(schedule['gameDateTimeEst'])
#     tomorrow = (datetime.now() - timedelta(hours=8)).date() + timedelta(days=1)
#     tomorrow_games = schedule[schedule['gameDateTimeEst'].dt.date == tomorrow]

#     # Get teams playing tomorrow
#     team_ids = set(tomorrow_games['hometeamId']).union(set(tomorrow_games['awayteamId']))
#     team_names = {teamid_to_simple[tid] for tid in team_ids if tid in teamid_to_simple}

#     # Load model training data
#     df = pd.read_csv("model_training_data.csv")
#     df_latest = df.sort_values(by='gameDate', ascending=False).drop_duplicates(subset=['firstName', 'lastName'])
#     df_tomorrow = df_latest[df_latest['playerteamName'].isin(team_names)]

#     # Add avg_minutes
#     avg_minutes = (
#         df.groupby(['firstName', 'lastName'])['numMinutes']
#         .mean().reset_index()
#     )
#     df_tomorrow = df_tomorrow.drop(columns=['numMinutes'], errors='ignore')
#     df_tomorrow = df_tomorrow.merge(avg_minutes, on=['firstName', 'lastName'], how='left')

#     # Determine opponentteamName
#     def find_opponent(team_name):
#         team_id = simple_to_teamid.get(team_name)
#         row = tomorrow_games[
#             (tomorrow_games['hometeamId'] == team_id) | (tomorrow_games['awayteamId'] == team_id)
#         ]
#         if not row.empty:
#             home = row.iloc[0]['hometeamId']
#             away = row.iloc[0]['awayteamId']
#             opp_id = away if home == team_id else home
#             return teamid_to_simple.get(opp_id)
#         return None

#     df_tomorrow['opponentteamName'] = df_tomorrow['playerteamName'].apply(find_opponent)

#     # Load trained model
#     # model = mlflow.sklearn.load_model("mlruns/0/af2b1d37cbd44718ab497471c47deef9/artifacts/model")
#     model = mlflow.sklearn.load_model("mlruns/0/a5cefbc637fe4c24b6d693e303f11826/artifacts/model")


#     # Prepare features for prediction
#     features = ['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']
#     X = df_tomorrow[features].dropna()
#     df_tomorrow = df_tomorrow.loc[X.index]

#     df_tomorrow["predicted_fp"] = model.predict(X)
#     df_tomorrow["diff_from_season_avg"] = df_tomorrow["predicted_fp"] - df_tomorrow["season_avg_fp"]

#     # Add image and OSS info
#     def get_image_url(row):
#         name = f"{row['firstName']} {row['lastName']}"
#         return player_lookup.get(name, {}).get("image_url")

#     def get_position(row):
#         name = f"{row['firstName']} {row['lastName']}"
#         return player_lookup.get(name, {}).get("position")

#     def get_oss_message(row):
#         position = get_position(row)
#         team = row['opponentteamName']
#         oss = row['opponent_oss']
#         if position and team and oss:
#             return f"{team} allow more fantasy points to {position}s"
#         return ""

#     df_tomorrow["image_url"] = df_tomorrow.apply(get_image_url, axis=1)
#     df_tomorrow["oss_message"] = df_tomorrow.apply(get_oss_message, axis=1)

#     # Final output columns (now includes season_avg_fp)
#     columns = [
#         "firstName", "lastName", "playerteamName", "opponentteamName",
#         "predicted_fp", "season_avg_fp", "diff_from_season_avg",
#         "image_url", "oss_message"
#     ]

#     top_predicted = df_tomorrow.sort_values(by="predicted_fp", ascending=False).head(3)[columns]
#     top_booms = df_tomorrow.sort_values(by="diff_from_season_avg", ascending=False).head(3)[columns]

#     return top_predicted, top_booms