import pandas as pd
import json
import mlflow.sklearn
import kagglehub
import os
from datetime import datetime, timedelta

path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")

def get_tomorrows_predictions():
    # Load caches and mappings
    with open("player_lookup_cache.json", "r") as f:
        player_lookup = json.load(f)
    with open("opponent_strength_cache.json", "r") as f:
        oss_cache = json.load(f)
    with open("teams.json", "r") as f:
        team_mappings = json.load(f)

    # Build teamId -> simpleName mapping
    teamid_to_simple = {entry["teamId"]: entry["simpleName"] for entry in team_mappings}
    simple_to_teamid = {entry["simpleName"]: entry["teamId"] for entry in team_mappings}

    # Load schedule and filter for tomorrow's games
    schedule = pd.read_csv(os.path.join(path, "LeagueSchedule24_25.csv"), low_memory=False)
    schedule['gameDateTimeEst'] = pd.to_datetime(schedule['gameDateTimeEst'])
    tomorrow = (datetime.now() - timedelta(hours=8)).date() + timedelta(days=1)
    tomorrow_games = schedule[schedule['gameDateTimeEst'].dt.date == tomorrow]

    # Get teams playing tomorrow
    team_ids = set(tomorrow_games['hometeamId']).union(set(tomorrow_games['awayteamId']))
    team_names = {teamid_to_simple[tid] for tid in team_ids if tid in teamid_to_simple}

    # Load model training data
    df = pd.read_csv("model_training_data.csv")
    df_latest = df.sort_values(by='gameDate', ascending=False).drop_duplicates(subset=['firstName', 'lastName'])
    df_tomorrow = df_latest[df_latest['playerteamName'].isin(team_names)]

    # Add avg_minutes
    avg_minutes = (
        df.groupby(['firstName', 'lastName'])['numMinutes']
        .mean().reset_index()
    )
    df_tomorrow = df_tomorrow.drop(columns=['numMinutes'], errors='ignore')
    df_tomorrow = df_tomorrow.merge(avg_minutes, on=['firstName', 'lastName'], how='left')

    # Determine opponentteamName
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

    df_tomorrow['opponentteamName'] = df_tomorrow['playerteamName'].apply(find_opponent)

    # Load trained model
    model = mlflow.sklearn.load_model("mlruns/0/af2b1d37cbd44718ab497471c47deef9/artifacts/model")

    # Prepare features for prediction
    features = ['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']
    X = df_tomorrow[features].dropna()
    df_tomorrow = df_tomorrow.loc[X.index]

    df_tomorrow["predicted_fp"] = model.predict(X)
    df_tomorrow["diff_from_season_avg"] = df_tomorrow["predicted_fp"] - df_tomorrow["season_avg_fp"]

    # Add image and OSS info
    def get_image_url(row):
        name = f"{row['firstName']} {row['lastName']}"
        return player_lookup.get(name, {}).get("image_url")

    def get_position(row):
        name = f"{row['firstName']} {row['lastName']}"
        return player_lookup.get(name, {}).get("position")

    def get_oss_message(row):
        position = get_position(row)
        team = row['opponentteamName']
        oss = row['opponent_oss']
        if position and team and oss:
            return f"{team} allow more fantasy points to {position}s"
        return ""

    df_tomorrow["image_url"] = df_tomorrow.apply(get_image_url, axis=1)
    df_tomorrow["oss_message"] = df_tomorrow.apply(get_oss_message, axis=1)

    # Final output columns (now includes season_avg_fp)
    columns = [
        "firstName", "lastName", "playerteamName", "opponentteamName",
        "predicted_fp", "season_avg_fp", "diff_from_season_avg",
        "image_url", "oss_message"
    ]

    top_predicted = df_tomorrow.sort_values(by="predicted_fp", ascending=False).head(3)[columns]
    top_booms = df_tomorrow.sort_values(by="diff_from_season_avg", ascending=False).head(3)[columns]

    return top_predicted, top_booms