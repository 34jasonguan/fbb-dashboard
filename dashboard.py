import kagglehub
import os
import pandas as pd
import json
from datetime import datetime, timedelta
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
from nba_api.stats.static import players

# ---------------
# DATA PROCESSING
# ---------------

# Download latest version of the NBA dataset, read data
path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
nba = pd.read_csv(
    os.path.join(path, "PlayerStatistics.csv"), 
    nrows=10000,
    low_memory=False
)

# Convert date
nba['gameDate'] = pd.to_datetime(nba['gameDate'])

# Create new df with relevant data for fantasy
fantasy_stats = nba[nba['gameDate'] >= pd.Timestamp("2024-10-22")][
    ['firstName', 'lastName', 'gameDate', 'playerteamName', 'opponentteamName', 'win',
     'numMinutes', 'points', 'assists', 'blocks', 'steals',
     'fieldGoalsAttempted', 'fieldGoalsMade', 'reboundsTotal', 'turnovers',
     'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade']
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

# cache money
CACHE_FILE = "player_lookup_cache.json"
def get_player_id(first_name, last_name):
    full_name = f"{first_name} {last_name}"
    result = players.find_players_by_full_name(full_name)
    if result:
        return result[0]['id']
    return None

def build_player_image_url(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

# Find yesterday's top performers
yesterday = datetime.today().date() - timedelta(days=1)
games_yesterday = fantasy_stats[fantasy_stats['gameDate'].dt.date == yesterday]
top_fantasy_players = games_yesterday.sort_values(by='fp', ascending=False).head(5)

# Find top 3 performers in the last 5 games
sorted_stats = fantasy_stats.sort_values(by=['firstName', 'lastName', 'gameDate'])
last_5_games = sorted_stats.groupby(['firstName', 'lastName']).tail(5)

player_totals = last_5_games.groupby(['firstName', 'lastName']).agg({
    'fp': 'sum',
    'playerteamName': 'last'
}).reset_index()

top_3 = player_totals.sort_values(by='fp', ascending=False).head(3)
top_3_names = top_3[['firstName', 'lastName']]
merged = pd.merge(last_5_games, top_3_names, on=['firstName', 'lastName'], how='inner')

# -------------
# CACHED LOOKUP
# -------------

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        player_lookup = json.load(f)
else:
    player_lookup = {}

display_names = pd.concat([
    top_fantasy_players[['firstName', 'lastName']],
    top_3[['firstName', 'lastName']]
]).drop_duplicates()

# lookup missing players if necessary
updated = False
for _, row in display_names.iterrows():
    full_name = (row['firstName'], row['lastName'])
    key = f"{full_name[0]} {full_name[1]}"
    if key not in player_lookup:
        pid = get_player_id(*full_name)
        img_url = build_player_image_url(pid) if pid else None
        player_lookup[key] = {'player_id': pid, 'image_url': img_url}
        updated = True

if updated:
    with open(CACHE_FILE, "w") as f:
        json.dump(player_lookup, f)

def get_cached(field, row):
    key = f"{row['firstName']} {row['lastName']}"
    return player_lookup.get(key, {}).get(field)

# Apply to both sets
top_fantasy_players['player_id'] = top_fantasy_players.apply(lambda row: get_cached('player_id', row), axis=1)
top_fantasy_players['image_url'] = top_fantasy_players.apply(lambda row: get_cached('image_url', row), axis=1)

merged['player_id'] = merged.apply(lambda row: get_cached('player_id', row), axis=1)
merged['image_url'] = merged.apply(lambda row: get_cached('image_url', row), axis=1)

# ---------------
# DASH COMPONENTS
# ---------------

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

def create_player_row(player_df):
    player = player_df.iloc[0]
    total_fp = player_df['fp'].sum()
    card = html.Div([
        html.Img(src=player["image_url"], style={"width": "80px", "border-radius": "8px"}),
        html.H4(f"{player['firstName']} {player['lastName']}"),
        html.P(f"Team: {player['playerteamName']}"),
        html.P(f"{total_fp} fantasy points over the past 5 games")
    ], style={
        "width": "150px",
        "padding": "10px",
        "textAlign": "center"
    })

    fig = px.line(
        player_df.sort_values(by='gameDate'),
        x='gameDate',
        y='fp',
        markers=True,
        title=f"{player['firstName']} {player['lastName']} - Last 5 Games",
    )
    chart = dcc.Graph(figure=fig, style={"flex": "1"})

    return html.Div([
        card,
        chart
    ], style={
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "center",
        "padding": "20px",
        "borderBottom": "1px solid #ddd"
    })

player_cards = [create_player_card(row) for _, row in top_fantasy_players.iterrows()]

player_rows = []
for _, player in top_3.iterrows():
    player_history = merged[(merged['firstName'] == player['firstName']) & (merged['lastName'] == player['lastName'])]
    player_rows.append(create_player_row(player_history))

# ---------
# FRONT END
# ---------

app = dash.Dash(__name__)
app.title = "Fantasy Basketball Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("BoxOut", style={
            "color": "#000000",
            "margin": "0",
            "fontSize": "3rem"
        })
    ], style={
        "backgroundColor": "#FFA500",
        "padding": "20px 0",
        "textAlign": "center",
        "boxShadow": "0px 2px 4px rgba(0,0,0,0.1)",
        "marginBottom": "20px"
    }),

    html.H1("Yesterday's Top Performers"),

    html.Div(player_cards, style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "center",
        "flexWrap": "wrap",
        "gap": "10px"
    }), 

    html.H1("Top Players Over The Last 5 Games"),
    html.Div(player_rows)
])

if __name__ == "__main__":
    app.run(debug=True)
