from main import agg_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

## TIME SERIES ANALYSIS
# Set 'ActivityDate' as the index of the DataFrame
agg_df.set_index('ActivityDate', inplace=True)

# Resample data to a daily frequency (assuming the data is not already daily)
# You can choose other frequencies like 'W' for weekly or 'M' for monthly as needed
daily_df = agg_df.resample('D').mean()

# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(daily_df.index, daily_df['Calories'], label='Calories Burned', color='blue')
plt.title('Calories Burned Over Time')
plt.xlabel('Date')
plt.ylabel('Calories')
plt.legend()
plt.show()

# Time Series Decomposition (Trend, Seasonality, Residual)
decomposition = sm.tsa.seasonal_decompose(daily_df['Calories'], model='additive')

# Plot the decomposition components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend', color='blue')
plt.legend()
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend()
plt.subplot(413)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend()
plt.tight_layout()

# Create lag features for time series analysis (same as before)
num_lags = 7
for i in range(1, num_lags + 1):
    daily_df[f'Lag_{i}_Calories'] = daily_df['Calories'].shift(i)
daily_df.dropna(inplace=True)

# Split the data into training and test sets
train_size = int(0.8 * len(daily_df))
train_data, test_data = daily_df[:train_size], daily_df[train_size:]

# Define a SARIMA model function
def sarima_estimator(order, seasonal_order):
    model = SARIMAX(train_data['Calories'], order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

# Define a function to evaluate SARIMA models
def evaluate_sarima(order, seasonal_order):
    model = sarima_estimator(order, seasonal_order)
    y_pred = model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
    rmse = np.sqrt(mean_squared_error(test_data['Calories'], y_pred))
    return rmse

# Define the SARIMA hyperparameter grid
order_grid = [(1, 1, 1), (1, 1, 2)]
seasonal_order_grid = [(1, 1, 1, 7)]  # Weekly seasonality

best_rmse = float('inf')
best_order = None
best_seasonal_order = None

# Loop through hyperparameter combinations and find the best model
for order in order_grid:
    for seasonal_order in seasonal_order_grid:
        rmse = evaluate_sarima(order, seasonal_order)
        if rmse < best_rmse:
            best_rmse = rmse
            best_order = order
            best_seasonal_order = seasonal_order

print("Best Hyperparameters:")
print("Order:", best_order)
print("Seasonal Order:", best_seasonal_order)

# Train the best SARIMA model on the entire training data
best_model = sarima_estimator(best_order, best_seasonal_order)

# Forecast on the test data
y_pred = best_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Calculate R-squared
r_squared = r2_score(test_data['Calories'], y_pred)
print("R-squared with Best Estimator:", r_squared)

# Calculate MAE
mae = mean_absolute_error(test_data['Calories'], y_pred)
print("Mean Absolute Error with Best Estimator:", mae)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data['Calories'], y_pred))
print("Root Mean Squared Error with Best Estimator:", rmse)

# You can't set hyperparameters for GridSearchCV with SARIMA as 
# statsmodels doesn't have a set_params method like some other scikit-learn estimators.
# Model Interpretability is not applicable for SARIMA
# This code manually loops through the hyperparameter combinations and selects the best SARIMA model based on RMSE. 
# It then evaluates the model on the test data and calculates performance metrics. 
# Note that SARIMA models don't have feature importances, so that section is not applicable.

# Some more examples of code which can be added to this project.

# Load your time series data here

# ... Existing code for resampling, lag features, splitting data, and SARIMA model ...

# Additional Feature Engineering: Rolling Statistics
window_size = 7  # Choose an appropriate window size
daily_df['Rolling_Mean'] = daily_df['Calories'].rolling(window=window_size).mean()
daily_df['Rolling_Std'] = daily_df['Calories'].rolling(window=window_size).std()

# Exploratory Data Analysis (EDA)
# Visualize autocorrelation plot
sm.graphics.tsa.plot_acf(daily_df['Calories'], lags=40)
plt.show()

# Model Selection: Try an Exponential Smoothing model (Holt-Winters)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
exp_smooth = ExponentialSmoothing(train_data['Calories'], trend='add', seasonal='add', seasonal_periods=7)
exp_smooth_fit = exp_smooth.fit()
exp_smooth_pred = exp_smooth_fit.forecast(len(test_data))

# Hyperparameter Tuning: For Exponential Smoothing, there are alpha, beta, and gamma parameters that can be tuned

# Ensemble Models: Combine SARIMA and Exponential Smoothing forecasts
ensemble_forecast = (y_pred + exp_smooth_pred) / 2

# Cross-Validation: Use Walk-Forward Validation for time series data

# Model Evaluation: Calculate additional metrics like MAPE and SMAPE

# Forecast Visualization: Plot the forecasted values with confidence intervals

# Model Monitoring and Maintenance: Implement a retraining schedule

# Error Analysis: Analyze model errors to identify patterns and improve the models

# Advanced Techniques: Explore Bayesian structural time series (BSTS) or deep learning models

# Alerting and Reporting: Implement alerts or reporting mechanisms
