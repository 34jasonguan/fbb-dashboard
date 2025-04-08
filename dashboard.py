import kagglehub
import os
import pandas as pd
from datetime import datetime, timedelta
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
from nba_api.stats.static import players

# Download latest version of the NBA dataset, read data
path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
nba = pd.read_csv(os.path.join(path, "PlayerStatistics.csv"), low_memory=False)

# Convert date
nba['gameDate'] = pd.to_datetime(nba['gameDate'])

# Create with relevant data for fantasy
fantasy_stats = nba[nba['gameDate'] >= pd.Timestamp("2024-10-22")][
    ['firstName', 'lastName', 'gameDate', 'playerteamName', 'opponentteamName', 'win',
    'numMinutes', 'points', 'assists', 'blocks', 'steals',
    'fieldGoalsAttempted', 'fieldGoalsMade',
    'reboundsTotal', 'turnovers', 'threePointersMade',
    'freeThrowsAttempted', 'freeThrowsMade']
]

fantasy_stats['fp'] = (
    fantasy_stats['points'] +
    fantasy_stats['reboundsTotal'] +
    fantasy_stats['assists'] * 2 -
    fantasy_stats['turnovers'] * 2 +
    fantasy_stats['fieldGoalsMade'] * 2 -
    fantasy_stats['fieldGoalsAttempted'] +
    fantasy_stats['blocks'] * 4 +
    fantasy_stats['steals'] * 4 -
    (fantasy_stats['freeThrowsAttempted'] - fantasy_stats['freeThrowsMade']) +
    fantasy_stats['threePointersMade']
)

# player ID function
def get_player_id(first_name, last_name):
    full_name = f"{first_name} {last_name}"
    result = players.find_players_by_full_name(full_name)
    if result:
        return result[0]['id']
    return None

# find player image
def build_player_image_url(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

# Find yesterday's top performers
yesterday = datetime.today().date() - timedelta(days=1)
games_yesterday = fantasy_stats[fantasy_stats['gameDate'].dt.date == yesterday]
top_fantasy_players = games_yesterday.sort_values(by='fp', ascending=False).head(5)
# print(top_fantasy_players[['firstName', 'lastName', 'playerteamName', 'fp']])

# Add player_id and image_url columns
top_fantasy_players = top_fantasy_players.copy()
top_fantasy_players["player_id"] = top_fantasy_players.apply(
    lambda row: get_player_id(row["firstName"], row["lastName"]), axis=1
)

top_fantasy_players["image_url"] = top_fantasy_players["player_id"].apply(
    lambda pid: build_player_image_url(pid) if pid else None
)

# display players
def create_player_card(player):
    return html.Div([
        html.Img(src=player["image_url"], style={"width": "100px", "border-radius": "10px"}),
        html.H4(f"{player['firstName']} {player['lastName']}"),
        html.P(f"FP: {round(player['fp'], 1)}")
    ], style={
        "border": "1px solid #ccc",
        "padding": "10px",
        "margin": "5px",
        "textAlign": "center",
        "borderRadius": "10px",
        "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
        "width": "150px"
    })

player_cards = [
    create_player_card(row)
    for _, row in top_fantasy_players.iterrows()
]

# Creating the actual dashboard using Dash
app = dash.Dash(__name__)
app.title = "Fantasy Basketball Dashboard"

app.layout = html.Div([
    html.H1("Top Fantasy Performers - Yesterday"),
    
    html.Div(player_cards, style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "center",
        "flexWrap": "wrap",
        "gap": "10px"
    })
])

if __name__ == "__main__":
    app.run(debug=True)