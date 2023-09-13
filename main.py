import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Class import DataAnalyzer
from scipy import stats
import calendar

# Define the file path
file_path = r'C:\Users\ljfit\Desktop\Random Coding\Fitbit Project\sql_tables\fourth_stage.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.dtypes)

# Create a list of the column names
column_names = list(df.columns)
# Print the column names
print(column_names)

# # Check for any missing values
# print(df.isnull().sum())

# # Loop through each column and print the number of unique values
# for column in df.columns:
#     unique_values = df[column].nunique()
#     print(f"Number of unique values in '{column}': {unique_values}")

# # Describe the DataFrame
# print(df.describe())

# # Frequency Counts for ID
# print(df['Id'].value_counts())

# # Check for any duplicate rows
# print(df.duplicated().sum())
# Drop any duplicate rows
df = df.drop_duplicates()

# # Save the cleaned DataFrame to a CSV file
# df.to_csv('fourth_stage_cleaned.csv', index=False)

# Convert 'ActivityDate' column to datetime type
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], format='%d/%m/%Y')

# Feature Engineering: Total active minutes
df['TotalActiveMinutes'] = df['VeryActiveMinutes'] + df['FairlyActiveMinutes'] + df['LightlyActiveMinutes']

# Feature Engineering: Total Distance
distances = ['TotalDistance', 'TrackerDistance', 'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']

# Feature Engineering: Split ActivityDate into year, month, day, day of week, weekend columns
df['Day'] = df['ActivityDate'].dt.day
df['DayOfWeek'] = df['ActivityDate'].dt.dayofweek
df['Weekend'] = np.where(df['DayOfWeek'] > 4, 1, 0) # Create weekend column (0 = Mon-Fri, 1 = Sat-Sun)

# Create an instance of the DataAnalyzer class
data_analyzer = DataAnalyzer(df)

# # Example 1: Plot a single column
# data_analyzer.trend_and_subplot_by_id(
#     x_column="Day",
#     y_column="TotalSteps",
#     title_prefix="Total Steps Trend",
#     x_label="Date",
#     y_label="Total Steps",
#     plot_type="line"
# )

# # Example 2: Plot multiple columns (pass them as a list)
# data_analyzer.trend_and_subplot_by_id(
#     x_column="Day",
#     y_column=['VeryActiveMinutes','FairlyActiveMinutes','LightlyActiveMinutes'],
#     title_prefix="Steps and Distance Trend",
#     x_label="Date",
#     y_label="Values",
#     plot_type="line"
# )


# Calculate the correlation between total steps and calories burnt
correlation_steps_calories = df['TotalSteps'].corr(df['Calories'])

# Calculate the correlation between very active minutes and calories burnt
correlation_active_minutes_calories = df['VeryActiveMinutes'].corr(df['Calories'])

print('Correlation between total steps and calories burnt: ', correlation_steps_calories)
print('Correlation between very active minutes and calories burnt: ', correlation_active_minutes_calories)

# Descriptive Statistics of sedentary minutes
print(df['SedentaryMinutes'].describe())

# Divide the data into segments based on sedentary behavior.
# Low sedentary: Less than 4 hours.
# Moderate sedentary: 4-8 hours.
# High sedentary: More than 8 hours.

# Discern sedentary categories
def sedentary_segment(minutes):
    if minutes < 4*60:  # Less than 4 hours
        return "Low sedentary"
    elif 4*60 <= minutes <= 8*60:  # 4-8 hours
        return "Moderate sedentary"
    else:  # More than 8 hours
        return "High sedentary"


# Apply the function to the sedentaryminutes column
df['Sedentary_Segment'] = df['SedentaryMinutes'].apply(sedentary_segment)
average_caloric_burn_per_segment = df.groupby('Sedentary_Segment')['Calories'].mean().rename('Average Caloric Burn').reset_index()
print(average_caloric_burn_per_segment)

# Group average steps, calories, and sleep by ID
daily_avg = df.groupby('Id').agg({
    'TotalSteps': 'mean',
    'Calories': 'mean',
    'TotalMinutesAsleep': 'mean'  
}).reset_index()

# Rename columns for clarity
daily_avg.columns = ['id', 'mean_daily_steps', 'mean_daily_calories', 'mean_daily_sleep']

daily_avg.head()

# Customer Segmentation Analysis
def classify_by_steps(average_steps):
    if average_steps < 5000:
        return "Sedentary"
    elif 5000 <= average_steps < 7500:
        return "Lightly active"
    elif 7500 <= average_steps < 10000:
        return "Fairly active"
    else:
        return "Very active"

# Apply the function to the 'mean_daily_steps' column to create a new 'Activity_Category' column
daily_avg['Activity_Category'] = daily_avg['mean_daily_steps'].apply(classify_by_steps)

# Display the updated dataframe with the new 'Activity_Category' column
daily_avg.head()

# Group by 'Activity_Category' and count each group
user_type_percent = daily_avg.groupby('Activity_Category').size().reset_index(name='total')

# Calculate the total number of users
user_type_percent['totals'] = user_type_percent['total'].sum()

# Calculate the percentage for each user type
user_type_percent['total_percent'] = user_type_percent['total'] / user_type_percent['totals']

# Convert the percentage to a readable format
user_type_percent['percentage%'] = user_type_percent['total_percent'].apply(lambda x: f"{x*100:.2f}%")

# Reorder the levels of the user type
ordered_categories = ["Very active", "Fairly active", "Lightly active", "Sedentary"]
user_type_percent['Activity_Category'] = pd.Categorical(user_type_percent['Activity_Category'], categories=ordered_categories, ordered=True)
user_type_percent = user_type_percent.sort_values('Activity_Category')

# draw a bar graph for each category
print(user_type_percent)

# Data for the pie chart
labels = user_type_percent['Activity_Category']
sizes = user_type_percent['total_percent']

# Plotting the pie chart
plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b2ff','#99ff99','#ffcc99'])
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of User Activity Levels')
plt.show()

# Testing statistical difference in caloric burn between the different user types based on the one-way ANOVA test
# Create a list of the different user types
# Group data by Activity_Category
groups = [daily_avg['mean_daily_calories'][daily_avg['Activity_Category'] == category] for category in ordered_categories]

# Perform ANOVA
f_statistic, p_value = stats.f_oneway(*groups)
print("F statistic: ",f_statistic,"p-value: ", p_value)

# #Trend analysis over specific days of the week
# # Group by 'day_of_week' and count each group
# day_of_week = df.groupby('DayOfWeek').size().reset_index(name='total')

# # Calculate the total number of users
# day_of_week['totals'] = day_of_week['total'].sum()

# # Calculate the percentage for each user type
# day_of_week['total_percent'] = day_of_week['total'] / day_of_week['totals']

# # Convert the percentage to a readable format
# day_of_week['percentage%'] = day_of_week['total_percent'].apply(lambda x: f"{x*100:.2f}%")

# # Convert the DatetimeIndex to day names
# day_of_week['DayOfWeek'] = day_of_week['DayOfWeek'].apply(lambda x: calendar.day_name[x])

# # Group by these day names and calculate the mean steps, calories, and sleep
# day_of_week = df.groupby('DayOfWeek').agg({
#     'TotalSteps': 'mean',
#     'Calories': 'mean',
#     'TotalMinutesAsleep': 'mean'
# }).reset_index()

# # Rename columns for clarity
# day_of_week.columns = ['DayOfWeek', 'mean_daily_steps', 'mean_daily_calories', 'mean_daily_sleep']

# # Reorder the levels of the day of the week
# ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# day_of_week['DayOfWeek'] = pd.Categorical(day_of_week['DayOfWeek'], categories=ordered_days, ordered=True)
# day_of_week = day_of_week.sort_values('DayOfWeek')

# # Draw a bar graph for each category
# day_of_week

# # Data for the pie chart
# labels = day_of_week['DayOfWeek']
# sizes = day_of_week['mean_daily_steps']

# # Plotting the pie chart
# plt.figure(figsize=(10, 7))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b2ff','#99ff99','#ffcc99'])
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Distribution of User Activity Levels')
# plt.show()

