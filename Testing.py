import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path
file_path = r'C:\Users\ljfit\Desktop\Random Coding\Fitbit Project\sql_tables\fourth_stage.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Convert 'ActivityDate' column to datetime type
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], format='%d/%m/%Y')

columns_to_remove = ['ActivityDate', 'Id']
df_main = df.drop(columns=columns_to_remove)

# Distances scatter plot
distances = ['TotalDistance', 'TrackerDistance', 'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']

# Loop through columns and plot with steps
for column in distances:

    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Scatter plot with 'steps' on the x-axis and the 'column' variable on the y-axis
    sns.scatterplot(data=df_main, x='TotalSteps', y=column)

    # Add title
    plt.title(f"Scatter Plot of Steps and {column}")

    # Show plot
    plt.show()