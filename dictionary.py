# Modules
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet

# def trend_and_subplot(df, config):
#     x_column = config.get("x_column", None)
#     y_column = config.get("y_column", [])
#     granularity = config.get("granularity", None)
#     title_prefix = config.get("title_prefix", "")
#     x_label = config.get("x_label", "")
#     y_label = config.get("y_label", "")
#     rotate_x_ticks = config.get("rotate_x_ticks", True)
#     by_id = config.get("by_id", False)
#     plot_type = config.get("plot_type", "line")

#     unique_ids = df['Id'].unique()
#     num_subplots = len(unique_ids)
#     num_rows = (num_subplots + 1) // 2  # Add 1 to account for the x_column
#     num_cols = 2  # Set default value to 2

#     for idx, column in enumerate(y_column):
#     plt.bar(subset_df[x_column], subset_df[column], label=y_label[idx])

#     plt.figure(figsize=(16, 8 * num_rows))  # Set figure size

#     for idx, unique_id in enumerate(unique_ids):
#         subset_df = df[df['Id'] == unique_id]
#         title = f"{title_prefix} for ID {unique_id}"
#         plt.subplot(num_rows, num_cols, idx + 1)  # Increment idx by 1

#         if plot_type == "line":
#             for column in y_column:
#                 plt.plot(subset_df[x_column], subset_df[column], label=column)  # Line plot
#             plt.title(title)
#             plt.xlabel(x_label)
#             plt.ylabel(', '.join(y_column))  # Join multiple y_column labels
#             plt.legend()  # Add legend to differentiate y_columns

#         elif plot_type == "bar":
#             for column in y_column:
#                 plt.bar(subset_df[x_column], subset_df[column], label=column)  # Bar plot
#             plt.title(title)
#             plt.xlabel(x_label)
#             plt.ylabel(', '.join(y_column))  # Join multiple y_column labels
#             plt.legend()  # Add legend to differentiate y_columns

#         elif plot_type == "scatter":
#             for column in y_column:
#                 plt.scatter(subset_df[x_column], subset_df[column], label=column)  # Scatter plot
#             plt.title(title)
#             plt.xlabel(x_label)
#             plt.ylabel(', '.join(y_column))  # Join multiple y_column labels
#             plt.legend()  # Add legend to differentiate y_columns

#         if granularity == "daily":
#             plt.xticks(rotation=rotate_x_ticks)

#     plt.tight_layout(h_pad=15.0, w_pad=1.0)
#     plt.show()

# # Define configurations for different instances
# config = {
#     "heatmap": {
#         "x_column": "Id",
#         "y_column": "TotalSteps",
#         "granularity": "daily",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True
#         # "plot_type": heatmap
#     },
#     "bar_plot": {
#         "x_column": "Id",
#         "y_column": ["TotalSteps", "TotalDistance"],
#         "granularity": "daily",
#         "title_prefix": "Bar Plot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True,
#         "plot_type": "bar"
#     },
#     "histogram": {
#         "x_column": "Id",
#         "y_column": "TotalSteps",
#         "granularity": "daily",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True,
#         "bins": 20,
#         "plot_type": "hist"
#     },
#     "boxplot": {
#         "x_column": "Id",
#         "y_column": "TotalSteps",
#         "granularity": "daily",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True,
#         "plot_type": "box"
#     },
#     "scatter_plot": {
#         "x_column": "",
#         "y_column": "TotalSteps",
#         "granularity": "daily",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True
#     },
#     "trend_over_time_steps": {
#         "x_column": "Calories",
#         "y_column": ["TotalSteps", "TotalDistance"],
#         "granularity": "DayOfWeek",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True,
#         "plot_type": "bar"
#     },
#      "trend_over_time_minutes": {
#         "x_column": "DayOfWeek",
#         "y_column": "TotalSteps",
#         "granularity": "daily",
#         "title_prefix": "Boxplot for",
#         "x_label": "ID",
#         "y_label": "Total Steps",
#         "rotate_x_ticks": True,
#         "by_id": True
#     }
# }

# Define the file path
file_path = r'C:\Users\ljfit\Desktop\Random Coding\Fitbit Project\sql_tables\fourth_stage.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Feature Engineering: Total Distance
distances = ['TotalDistance', 'TrackerDistance', 'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']

# Feature Engineering: Total active minutes
df['TotalActiveMinutes'] = df['VeryActiveMinutes'] + df['FairlyActiveMinutes'] + df['LightlyActiveMinutes']

# Convert 'ActivityDate' column to datetime type
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], format='%d/%m/%Y')
# df = df.drop(columns=Date)

# Feature Engineering: Split ActivityDate into year, month, day, day of week and weekend columns
df['Day'] = df['ActivityDate'].dt.day
df['DayOfWeek'] = df['ActivityDate'].dt.dayofweek
df['Weekend'] = np.where(df['DayOfWeek'] > 4, 1, 0) # Create weekend column (0 = Mon-Fri, 1 = Sat-Sun)

# # # Call the function and pass the configuration using nested keys
# trend_and_subplot(df, config)

print(df.columns)