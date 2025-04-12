# Introducing BoxOut
BoxOut is an interactive dashboard created for fantasy basketball enthusiasts and/or addicts (like myself) to view real-time performance insights and player trends in a clear, straightforward manner. It displays the top current fantasy performers and tracks patterns from recent NBA games to guide fantasy managers through potential trades and waiver moves. 

BoxOut also uses a tree-based gradient boosting model (trained using scikit-learn and managed using MLflow) to predict top performers and pop-off candidates for future NBA games using data from the box scores of every single nba game this season. 

Note: since we are sadly reaching the end of the NBA (and fantasy basketball) regular season, the predictor has been hard-coded to display predictions for the last game of the regular season as there are no more games to predict for fantasy purposes. BoxOut will be back for the 2025-2026 regular season!

Built using Dash & pandas

The dataset used for this project can be found at https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

The following features are in the works!
- improved aesthetics using CSS
- more complex buy-low/sell-high breakdowns
- detailed injury impact modeling & visualization
