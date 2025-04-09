import kagglehub
import os
import pandas as pd
import json
from datetime import datetime, timedelta
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
from predictor import get_tomorrows_predictions
from plots import create_pred_vs_actual_plot

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
with open("player_lookup_cache.json", "r") as f:
    player_lookup = json.load(f)

def get_cached(field, row):
    key = f"{row['firstName']} {row['lastName']}"
    return player_lookup.get(key, {}).get(field)

# --------------
# TOP PERFORMERS
# --------------

# yesterday's top performers
# yesterday = datetime.today().date() - timedelta(days=1)
now = datetime.now()
adjusted_date = (now - timedelta(hours=8)).date()
yesterday = adjusted_date - timedelta(days=1)
games_yesterday = fantasy_stats[fantasy_stats['gameDate'].dt.date == yesterday].copy()
games_yesterday["player_id"] = games_yesterday.apply(lambda row: get_cached("player_id", row), axis=1)
games_yesterday["image_url"] = games_yesterday.apply(lambda row: get_cached("image_url", row), axis=1)
games_yesterday["position"] = games_yesterday.apply(lambda row: get_cached("position", row), axis=1)
games_yesterday = games_yesterday[games_yesterday["position"].notna()]

top_5 = games_yesterday.sort_values(by='fp', ascending=False).head(5).copy()
top_5["category"] = "All"

top_guards = games_yesterday[games_yesterday["position"].astype(str).str.startswith("G")] \
    .sort_values(by="fp", ascending=False).head(5).copy()
top_guards["category"] = "Guards"

top_forwards = games_yesterday[games_yesterday["position"].astype(str).str.startswith("F")] \
    .sort_values(by="fp", ascending=False).head(5).copy()
top_forwards["category"] = "Forwards"

top_centers = games_yesterday[games_yesterday["position"].astype(str).str.startswith("C")] \
    .sort_values(by="fp", ascending=False).head(5).copy()
top_centers["category"] = "Centers"

top_fantasy_players = pd.concat([top_5, top_guards, top_forwards, top_centers], ignore_index=True)

# Top 3 performers over their last 5 games
sorted_stats = fantasy_stats.sort_values(by=['firstName', 'lastName', 'gameDate'])
last_5_games = sorted_stats.groupby(['firstName', 'lastName']).tail(5)

player_totals = last_5_games.groupby(['firstName', 'lastName']).agg({
    'fp': 'sum',
    'playerteamName': 'last'
}).reset_index()

top_3 = player_totals.sort_values(by='fp', ascending=False).head(3)
top_3_names = top_3[['firstName', 'lastName']]
merged = pd.merge(last_5_games, top_3_names, on=['firstName', 'lastName'], how='inner')

merged['player_id'] = merged.apply(lambda row: get_cached('player_id', row), axis=1)
merged['image_url'] = merged.apply(lambda row: get_cached('image_url', row), axis=1)
merged['position'] = merged.apply(lambda row: get_cached('position', row), axis=1)

# -----------------
# BUY LOW/SELL HIGH
# -----------------

last_5 = (
    fantasy_stats
    .sort_values(by='gameDate')
    .groupby(['firstName', 'lastName'])
    .tail(5)
)

play_counts = (
    last_5
    .assign(played=lambda df: df['numMinutes'] > 0)
    .groupby(['firstName', 'lastName'])['played']
    .sum()
    .reset_index()
    .rename(columns={'played': 'played_5_count'})
)

recent_avg = (
    last_5
    .groupby(['firstName', 'lastName'])['fp']
    .mean()
    .reset_index()
    .rename(columns={'fp': 'recent_avg_fp'})
)

recent_game_info = (
    last_5
    .sort_values(by='gameDate', ascending=False)
    .groupby(['firstName', 'lastName'])
    .head(1)[['firstName', 'lastName', 'playerteamName']]
)

recent_summary = recent_avg.merge(play_counts, on=['firstName', 'lastName'])
recent_summary = recent_summary.merge(recent_game_info, on=['firstName', 'lastName'])

lookup_df = (
    pd.DataFrame.from_dict(player_lookup, orient='index')
    .reset_index()
    .rename(columns={'index': 'full_name'})
)
lookup_df[['firstName', 'lastName']] = lookup_df['full_name'].str.split(' ', n=1, expand=True)

candidates = (
    pd.merge(lookup_df, recent_summary, on=['firstName', 'lastName'], how='inner')
    .assign(diff=lambda df: df['recent_avg_fp'] - df['avg_fp'])
    .query('avg_fp >= 30 and played_5_count >= 4')
)

buy_low_candidates = candidates.sort_values(by='diff').head(2)
sell_high_candidates = candidates.sort_values(by='diff', ascending=False).head(2)

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
    position = get_cached("position", player)
    card = html.Div([
        html.Img(src=player["image_url"], style={"width": "80px", "border-radius": "8px"}),
        html.H4(f"{player['firstName']} {player['lastName']}"),
        html.P(f"{position}, {player['playerteamName']}"),
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

def create_buy_sell_card(row):
    image_url = get_cached('image_url', row)
    position = get_cached("position", row)
    return html.Div([
        html.Img(src=image_url, style={"width": "80px", "border-radius": "8px"}),
        html.H4(f"{row['firstName']} {row['lastName']}"), 
        html.P(f"{position}, {row['playerteamName']}", style={"margin": "0"})
    ], style={"textAlign": "center"})

def create_fp_bar_chart(row):
    df = pd.DataFrame({
        "Category": ["Last 5 Avg", "Season Avg"],
        "FP": [row['recent_avg_fp'], row['avg_fp']]
    })
    fig = px.bar(df, x="Category", y="FP", title=f"{row['firstName']} {row['lastName']} FP Comparison", height=500)
    return dcc.Graph(figure=fig)

buy_low_section = html.Div([
    html.H1("Buy Low Candidates", style={"textAlign": "center"}),
    html.Div([
        create_buy_sell_card(buy_low_candidates.iloc[0]),
        create_fp_bar_chart(buy_low_candidates.iloc[0]),
        create_buy_sell_card(buy_low_candidates.iloc[1]),
        create_fp_bar_chart(buy_low_candidates.iloc[1]),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "20px",
        "backgroundColor": "#E0F7FA",
        "padding": "20px",
        "marginBottom": "40px"
    })
])

sell_high_section = html.Div([
    html.H1("Sell High Candidates", style={"textAlign": "center"}),
    html.Div([
        create_buy_sell_card(sell_high_candidates.iloc[0]),
        create_fp_bar_chart(sell_high_candidates.iloc[0]),
        create_buy_sell_card(sell_high_candidates.iloc[1]),
        create_fp_bar_chart(sell_high_candidates.iloc[1]),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "20px",
        "backgroundColor": "#FFEBEE",
        "padding": "20px",
        "marginBottom": "40px"
    })
])

top_preds, top_booms = get_tomorrows_predictions()

def create_prediction_card(row):
    return html.Div([
        html.Img(src=row["image_url"], style={"width": "80px", "border-radius": "8px"}),
        html.H4(f"{row['firstName']} {row['lastName']}"),
        html.P(f"Predicted FP: {round(row['predicted_fp'], 1)}"),
        html.P(f"Recent Average: {round(row['season_avg_fp'], 1)}"), 
        html.P(row["oss_message"], style={"margin": "0", "fontStyle": "italic"})
    ], style={
        "border": "1px solid #ccc",
        "padding": "10px",
        "margin": "5px",
        "textAlign": "center",
        "borderRadius": "10px",
        "width": "150px"
    })

prediction_section = html.Div([
    html.H1("Tomorrow's Top Predictions"),
    html.H4("Top 3 Predicted Performers", style={"textAlign": "center"}),
    html.Div([create_prediction_card(row) for _, row in top_preds.iterrows()],
             style={"display": "flex", "justifyContent": "center", "gap": "10px"}),

    html.H4("Top 3 Boom Candidates", style={"textAlign": "center", "marginTop": "30px"}),
    html.Div([create_prediction_card(row) for _, row in top_booms.iterrows()],
             style={"display": "flex", "justifyContent": "center", "gap": "10px"})
])

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

    html.Div([
        html.Button("All", id="btn-all", n_clicks=0),
        html.Button("Guard", id="btn-guard", n_clicks=0),
        html.Button("Forward", id="btn-forward", n_clicks=0),
        html.Button("Center", id="btn-center", n_clicks=0)
    ], style={
        "textAlign": "center",
        "marginBottom": "20px",
        "gap": "10px",
        "display": "flex",
        "justifyContent": "center"
    }),

    html.Div(id="top-player-cards", style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "center",
        "flexWrap": "wrap",
        "gap": "10px"
    }),

    html.Div(prediction_section),

    html.H1("Model Accuracy: Predicted vs Actual"),
    create_pred_vs_actual_plot(), 

    html.H1("Top Players Over The Last 5 Games"),
    html.Div(player_rows), 

    html.Div(buy_low_section), 

    html.Div(sell_high_section)
])

@app.callback(
    Output("top-player-cards", "children"),
    [Input("btn-all", "n_clicks"),
     Input("btn-guard", "n_clicks"),
     Input("btn-forward", "n_clicks"),
     Input("btn-center", "n_clicks")]
)
def update_top_performers(all_clicks, guard_clicks, forward_clicks, center_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        category = "All"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        category = {
            "btn-all": "All",
            "btn-guard": "G",
            "btn-forward": "F",
            "btn-center": "C"
        }.get(button_id, "All")

    df = games_yesterday.copy()
    df["player_id"] = df.apply(lambda row: get_cached("player_id", row), axis=1)
    df["image_url"] = df.apply(lambda row: get_cached("image_url", row), axis=1)
    df["position"] = df.apply(lambda row: get_cached("position", row), axis=1)

    if category != "All":
        df = df[df["position"].astype(str).str.startswith(category)]

    df = df.sort_values(by="fp", ascending=False).head(5)

    return [create_player_card(row) for _, row in df.iterrows()]

if __name__ == "__main__":
    app.run(debug=True)