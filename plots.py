import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
from dash import dcc

# def create_pred_vs_actual_plot():
#     # Load data and model
#     data = pd.read_csv("model_training_data.csv")
#     X = data[['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']]
#     y = data['fp']

#     model = mlflow.sklearn.load_model("mlruns/0/af2b1d37cbd44718ab497471c47deef9/artifacts/model")

#     # Split for evaluation
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     y_pred = model.predict(X_test)

#     # Compute metrics
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)

#     # Create dataframe for plotting
#     df = pd.DataFrame({
#         'Actual Fantasy Points': y_test,
#         'Predicted Fantasy Points': y_pred
#     })

#     # Create interactive scatter plot with trendline
#     fig = px.scatter(
#         df,
#         x='Actual Fantasy Points',
#         y='Predicted Fantasy Points',
#         title="Predicted vs Actual Fantasy Points (Interactive)",
#         labels={"Actual Fantasy Points": "Actual FP", "Predicted Fantasy Points": "Predicted FP"},
#         trendline="ols"
#     )

#     # Add annotation with R² and MAE
#     fig.add_annotation(
#         xref="paper", yref="paper",
#         x=0.01, y=0.99, showarrow=False,
#         text=f"R² = {r2:.3f}<br>MAE = {mae:.2f}",
#         bordercolor="black",
#         borderwidth=1,
#         bgcolor="white",
#         font=dict(size=12)
#     )

#     # Optional: improve layout
#     fig.update_layout(
#         height=600,
#         margin=dict(t=60, b=40, l=60, r=40)
#     )

#     return dcc.Graph(figure=fig)

def create_pred_vs_actual_plot():
    # Load and sort the data by time
    data = pd.read_csv("model_training_data.csv")
    data['gameDate'] = pd.to_datetime(data['gameDate'])
    data = data.sort_values(by='gameDate')

    # Drop rows with missing or lagged features (due to .shift or rolling avg)
    feature_cols = ['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']
    data = data.dropna(subset=feature_cols + ['fp'])

    X = data[feature_cols]
    y = data['fp']

    # Load the MLflow model
    # model = mlflow.sklearn.load_model("mlruns/0/af2b1d37cbd44718ab497471c47deef9/artifacts/model")
    model = mlflow.sklearn.load_model("mlruns/0/a5cefbc637fe4c24b6d693e303f11826/artifacts/model")

    # Use TimeSeriesSplit: take the last split
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        pass  # This gets us the final (most realistic) train/test split

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Actual Fantasy Points': y_test.values,
        'Predicted Fantasy Points': y_pred
    })

    # Interactive Plotly scatter plot with regression line
    fig = px.scatter(
        df,
        x='Actual Fantasy Points',
        y='Predicted Fantasy Points',
        title="Predicted vs Actual Fantasy Points (Time-Aware Split)",
        labels={"Actual Fantasy Points": "Actual FP", "Predicted Fantasy Points": "Predicted FP"},
        trendline="ols"
    )

    # Add annotation with R² and MAE
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99, showarrow=False,
        text=f"R² = {r2:.3f}<br>MAE = {mae:.2f}",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=12)
    )

    fig.update_layout(
        height=600,
        margin=dict(t=60, b=40, l=60, r=40)
    )

    return dcc.Graph(figure=fig)