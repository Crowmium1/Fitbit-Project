## Regression Analysis
# Import necessary modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from main import agg_df

# List of IDs to be removed
ids_to_remove = [1624580081, 1644430081, 1844505072, 1927972279, 2022484408]

# Create a new DataFrame without the specified IDs
filtered_agg_df = agg_df[~agg_df['Id'].isin(ids_to_remove)]

# Define the features and target
X = filtered_agg_df[['TotalSteps', 'TotalMinutesAsleep', 'TotalActiveMinutes']]
y = filtered_agg_df['Calories']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Create a pipeline with preprocessing and modeling steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['TotalSteps', 'TotalMinutesAsleep', 'TotalActiveMinutes']),
        ('imputer', SimpleImputer(strategy='mean'), ['TotalSteps', 'TotalMinutesAsleep', 'TotalActiveMinutes'])
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(r2_score(y_test, y_pred)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Get feature names from the ColumnTransformer
feature_names = (
    list(pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['TotalSteps', 'TotalMinutesAsleep', 'TotalActiveMinutes'])) +
    ['TotalSteps', 'TotalMinutesAsleep', 'TotalActiveMinutes']
)

# Get coefficients from the LinearRegression model
coefficients = pipeline.named_steps['regressor'].coef_

# Create a DataFrame of the features and coefficients
feature_coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Print the feature coefficients
print(feature_coef_df)

# Plot the coefficients
plt.figure(figsize=(10, 7))
plt.bar(df_coef['features'], df_coef['coefficients'])
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Feature Coefficients')
plt.show()

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'regressor__fit_intercept': [True, False],
    'regressor__normalize': [True, False],
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and its hyperparameters
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

# Predict on the test data using the best estimator
y_pred = best_estimator.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print("R-squared with Best Estimator:", r_squared)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error with Best Estimator:", mae)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error with Best Estimator:", rmse)

# Print the coefficients and their corresponding feature names
for coef, feature in zip(coefficients, X.columns):
    print(f"{feature}: {coef}")

import matplotlib.pyplot as plt

# Scatter plot of a single feature vs. target
plt.scatter(X['TotalSteps'], y)
plt.xlabel('TotalSteps')
plt.ylabel('Calories')
plt.title('Scatter Plot of TotalSteps vs. Calories')
plt.show()

# Calculate residuals
y_pred = reg_all.predict(X)
residuals = y - y_pred

# Plot residuals
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

import statsmodels.api as sm

# Fit the model using statsmodels for p-values
X = sm.add_constant(X)  # Add a constant for the intercept term
model = sm.OLS(y, X).fit()

# Access p-values
p_values = model.pvalues

# Print feature names and their p-values
for feature, p_value in zip(X.columns, p_values):
    print(f"{feature}: {p_value}")

