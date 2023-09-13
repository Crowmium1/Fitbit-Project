import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_frame):
        self.df = data_frame

    def trend_and_subplot_by_id(self, x_column, y_column, granularity=None, title_prefix="", x_label="", y_label="", rotate_x_ticks=False, plot_type="line"):
        unique_ids = self.df['Id'].unique()
        num_subplots = len(unique_ids)
        num_rows = (num_subplots + 1) // 2  # Add 1 to account for the x_column
        num_cols = 2  # Set default value to 2

        plt.figure(figsize=(16, 8 * num_rows))  # Set figure size

        for idx, unique_id in enumerate(unique_ids):
            subset_df = self.df[self.df['Id'] == unique_id]
            title = f"{title_prefix} for ID {unique_id}"
            plt.subplot(num_rows, num_cols, idx + 1)  # Increment idx by 1

            if plot_type == "line":
                if isinstance(y_column, list):
                    for column in y_column:
                        plt.plot(subset_df[x_column], subset_df[column], label=column)
                else:
                    plt.plot(subset_df[x_column], subset_df[y_column], label=y_column)
            elif plot_type == "bar":
                if isinstance(y_column, list):
                    for column in y_column:
                        plt.bar(subset_df[x_column], subset_df[column], label=column)
                else:
                    plt.bar(subset_df[x_column], subset_df[y_column], label=y_column)
            elif plot_type == "scatter":
                if isinstance(y_column, list):
                    for column in y_column:
                        plt.scatter(subset_df[x_column], subset_df[column], label=column)
                else:
                    plt.scatter(subset_df[x_column], subset_df[y_column], label=y_column)
            elif plot_type == "histogram":
                if isinstance(y_column, list):
                    for column in y_column:
                        plt.hist(subset_df[column], bins=20, label=column)
                else:
                    plt.hist(subset_df[y_column], bins=20, label=y_column)
            elif plot_type == "heatmap":
                if isinstance(y_column, list):
                    pivot_table = subset_df.pivot(index='DayOfWeek', columns='Weekend', values=y_column[0])
                else:
                    pivot_table = subset_df.pivot(index='DayOfWeek', columns='Weekend', values=y_column)
                sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")

            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            if rotate_x_ticks:
                plt.xticks(rotation=45)

        plt.tight_layout(h_pad=15.0, w_pad=0.5)
        plt.legend()
        plt.show()