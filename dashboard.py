import kagglehub
import os
import pandas as pd
from datetime import datetime, timedelta
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output

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

# Find yesterday's top performers
yesterday = datetime.today().date() - timedelta(days=1)
games_yesterday = fantasy_stats[fantasy_stats['gameDate'].dt.date == yesterday]
top_fantasy_players = games_yesterday.sort_values(by='fp', ascending=False).head(5)
# print(top_fantasy_players[['firstName', 'lastName', 'playerteamName', 'fp']])

# Creating the actual dashboard using Dash
app = dash.Dash(__name__)
app.title = "Fantasy Basketball Dashboard"

app.layout = html.Div([
    html.H1("Top Fantasy Performers - Yesterday"),
    
    dcc.Graph(
        id="top-fantasy-bar",
        figure=px.bar(
            top_fantasy_players,
            x="lastName",
            y="fp",
            hover_data=["firstName", "playerteamName"],
            title="Top 5 Fantasy Performers",
        )
    )
])

if __name__ == "__main__":
    app.run(debug=True)