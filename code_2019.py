# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

# Load the dfset
df = pd.read_csv('AB_NYC_2019.csv')

# Select columns of interest (Complication_risk and the 15 independent variables)
columns_of_interest = [
    'price', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'neighbourhood'
]

df = df[columns_of_interest]

# Remove duplicate rows
df = df.drop_duplicates()

# Check for null values
null_counts = df.isnull().sum()
# print("Null value counts:\n", null_counts)
# Since the only nulls are in 'reviews_per_month' and they're null because 
# they don't have any reviews yet, we'll replace these nulls with 0
df['reviews_per_month'].fillna(0, inplace=True)

# Handle outliers for numerical df using RobustScaler
numerical_columns = ['price', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365']

def handle_outliers(df, col, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = df[col].quantile(lower_percentile)
    upper_bound = df[col].quantile(upper_percentile)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

for col in numerical_columns:
    df = handle_outliers(df, col)

#print("Preprocessed df:\n", df.head())

# %% [markdown]
# Transformations

# %%
# Encode categorical variables
cat_cols = ['room_type', 'neighbourhood_group', 'neighbourhood']
one_hot_encoded_data = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
df_dropped = df.drop(columns=cat_cols)
df_encoded = pd.concat([df_dropped, one_hot_encoded_data], axis=1)
df_encoded.to_csv('cleaned_AB_NYC_2019.csv', index=False)

# %% [markdown]
# Initial Model

# %%
# Add constant for intercept
X = sm.add_constant(df_encoded.drop(columns='price'))

# Dependent variable
y = df_encoded['price']

# Convert DataFrame to NumPy array
X_array = X.to_numpy()
y_array = y.to_numpy()

# Fit the initial model
initial_model = sm.OLS(y_array, X_array).fit()

# Display model summary
print(initial_model.summary())

# %%
def backward_elimination(X, y, significance_level=0.05):
    while True:
        model = sm.OLS(y, X).fit()
        max_p_value = model.pvalues.max()
        if max_p_value > significance_level:
            max_p_var_label = model.pvalues.idxmax()
            print("Max p-value variable label:", max_p_var_label)
            X = X.drop(max_p_var_label, axis=1)
            print("Dropped variable:", max_p_var_label)
        else:
            break
    return model

reduced_model = backward_elimination(X, y)
print(reduced_model.summary())

# %%
# Calculate residuals for the reduced model
reduced_residuals = reduced_model.resid

# Calculate the residual standard error (RSE)
rse = np.sqrt(np.sum(reduced_residuals ** 2) / (len(y) - reduced_model.df_model - 1))
print("Residual Standard Error:", rse)

# %%
# Create a residual plot
plt.figure(figsize=(10, 6))
sns.residplot(x=reduced_model.fittedvalues, y=reduced_residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Residual Plot")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()


