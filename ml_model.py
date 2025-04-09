import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

data = pd.read_csv("model_training_data.csv")
X = data[['numMinutes', 'opponent_oss', 'recent_avg_fp', 'season_avg_fp', 'bfi']]
y = data['fp']

# random forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# alt model
model = HistGradientBoostingRegressor(random_state=42)

# cross validation yippee
r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
mse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

print("Cross-validated R² scores:", r2_scores)
print("Mean R²:", r2_scores.mean())
print("Mean MAE:", mae_scores.mean())
print("Mean MSE:", mse_scores.mean())

model.fit(X, y)

# log results
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("cv_r2_mean", r2_scores.mean())
    mlflow.log_metric("cv_r2_std", r2_scores.std())
    mlflow.log_metric("cv_mae_mean", mae_scores.mean())
    mlflow.log_metric("cv_mse_mean", mse_scores.mean())
    mlflow.sklearn.log_model(model, "model")