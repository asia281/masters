# -*- coding: utf-8 -*-
"""hw2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lV_NYoGFrmIcsqdwWIwrchrLZYd4zqFH
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

"""1. Exploration"""

X_train_all = pd.read_csv('X_train.csv', sep=';')
y_train_all = pd.read_csv('y_train.csv', sep=';')
X_test = pd.read_csv('X_test.csv', sep=';')

print("X_train shape:", X_train_all.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train_all.shape)

y_train_all.head()

X_train_all.head()

X_test.head()

"""(a) Check how many observations and variables are there in the loaded training and test data. Take a look at the types of the variables and, if necessary, make the appropriate conversions before further analysis. Make sure the data is complete."""

print("Data type of y_train:", y_train_all.dtypes)
nan_columns_train = X_train_all.columns[X_train_all.isnull().any()].tolist()
if nan_columns_train:
    print(f"Columns with missing values in X_train: {nan_columns_train}")
else:
    print("No missing values in X_train.")

nan_columns_test = X_test.columns[X_test.isnull().any()].tolist()
if nan_columns_train:
    print(f"Columns with missing values in X_test: {nan_columns_test}")
else:
    print("No missing values in X_train.")

"""b) Investigate the empirical distribution of the response variable (e.g., present some basic statistics, attach a histogram or graph of the density estimator to the analysis). Discuss the results."""

print(y_train_all.describe())

skewness = y_train_all['CD36_Y'].skew()
kurtosis = y_train_all['CD36_Y'].kurt()
print(skewness, kurtosis)

from scipy.stats import boxcox
y_transformed = y_train_all.copy()
y_transformed['boxcox'] = boxcox(y_train_all['CD36_Y'] + 0.8)[0]
y_transformed['log'] = np.log(y_train_all['CD36_Y'] + 0.8)
y_transformed['1/3'] = (y_train_all['CD36_Y'] + 0.8) ** (1/3)
y_transformed['arcsin'] = np.arcsinh(y_train_all['CD36_Y'])
from scipy.stats import yeojohnson
y_transformed['yeo'] = yeojohnson(y_train_all['CD36_Y'])[0]
print(y_transformed.describe())
skewness = y_transformed.skew()
kurtosis = y_transformed.kurt()
print('Skewness:', skewness, '\nKurtosis:', kurtosis)

fig = px.histogram(y_train_all, x='CD36_Y', nbins=100, title='Distribution of the CD36_Y')
fig.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(y_train_all['CD36_Y'], color='green', fill=True)
plt.title('Kernel Density Estimate of Response Variable')
plt.xlabel('Response Variable')
plt.ylabel('Density')
plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.kdeplot(y_transformed['log'], color='green', fill=True)
# plt.title('Kernel Density Estimate of logarithmic Response Variable')
# plt.xlabel('Response Variable')
# plt.ylabel('Density')
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.kdeplot(y_transformed['log'], color='green', fill=True)
# plt.title('Kernel Density Estimate of logarithmic Response Variable')
# plt.xlabel('Response Variable')
# plt.ylabel('Density')
# plt.show()

"""(c) Compute the appropriate correlation coefficients between the predictors and the response variable. Visualize the results in the compact way (we suggest violin plots). Select the 100 independent variables that are the most correlated with the response variable. Calculate the correlations for each pair of these variables, and provide a compact visualization of your choice (e.g., a heatmap) in search of multicolinearity. Discuss the results."""

correlations = X_train_all.corrwith(y_train_all['CD36_Y'])

correlations.isnull().any()

top_correlated_vars = correlations.abs().nlargest(100).index
print(top_correlated_vars)

# subset_df = X_train_all[top_correlated_vars]
# correlations_subset = subset_df.corr()

# plt.figure(figsize=(20, 8))
# sns.violinplot(x=subset_df.index, y=y_train_all['CD36_Y'])
# plt.xticks(rotation=90)
# plt.title('Correlation with Response Variable (Violin Plot)')
# plt.show()

# plt.figure(figsize=(20, 15))
# sns.heatmap(correlations_subset, cmap='coolwarm', annot=True, fmt=".2f")
# plt.title('Correlation Heatmap for Top 100 Variables')
# plt.show()

# from statsmodels.stats.outliers_influence import variance_inflation_factor

# vif_data = pd.DataFrame()
# vif_data["Variable"] = subset_df.columns
# vif_data["VIF"] = [variance_inflation_factor(subset_df.values, i) for i in range(subset_df.shape[1])]

# multi = vif_data[vif_data["VIF"] > 10]
# print(multi.shape, multi)

"""We picked the variables with the VIF > 10 and inspect their correlations."""

# smaller = correlations_subset[multi["Variable"]].corr()

# plt.figure(figsize=(20, 15))
# sns.heatmap(smaller, cmap='coolwarm', annot=True, fmt=".2f")
# plt.title('Correlation Heatmap for Top 32')
# plt.show()

"""2. ElasticNet (6p.) The first model to train is ElasticNet. During the lecture, we introduced its special cases: ridge regression and lasso.

(b) Define a grid of hyperparameters based on at least three values for each hyperparameter. Make sure that you included the hyperparameter configurations corresponding to the ridge and lasso regression. Use cross-validation to select appropriate hyperparameters (the number of subsets used in cross-validation is up to you to decide, but you have to justify your choice).

At first I'll perform gridsearch on 3 parameters alpha = [0.1, 1.0, 10.0] and 3 l1_ratio = [0.0, 0.5, 1.0]. The last 3 correspond to the most popular types of penalised regression: Ridge, Mix, and Lasso. Parameters alpha are choosen to see the performance for small, medium and large alpha.
"""

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# param_grid = {
#     'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
#     'l1_ratio': [0.0, 0.5, 1.0]  # Corresponding to Ridge, Mix, and Lasso
# }

# elasticnet_model = ElasticNet()

# cv = KFold(n_splits=5, shuffle=True, random_state=42)

# grid_search = GridSearchCV(elasticnet_model, param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error', verbose=3)
# grid_search.fit(X_train, y_train)

# best_alpha = grid_search.best_params_['alpha']
# best_l1_ratio = grid_search.best_params_['l1_ratio']

# """(c) Specify the training and validation error of the model (the result should be averaged over all subsets from the the cross-validation)."""

# best_elasticnet_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
# best_elasticnet_model.fit(X_train, y_train)

# train_error = -grid_search.best_score_  # neg mean squared error
# validation_error = mean_squared_error(y_validation, best_elasticnet_model.predict(X_validation))

# print(f"Best Hyperparameters - alpha: {best_alpha}, l1_ratio: {best_l1_ratio}")
# print(f"Training Error: {train_error}")
# print(f"Validation Error: {validation_error}")

"""3. Random forest (6p.) In this part of the project, you train the random forest model and compare its performance with the ElasticNet model from the previous task.

(a) From the many hyperparameters that characterize the random forest model, choose three different ones. Define a three-dimensional grid of hyperparameter combinations to be searched and select their optimal (in the context of the prediction) values using cross- validation. The data division used for cross-validation should be the same as in the case of ElasticNet model.

I chose 3 hyperparameters, with 3 possible values each: 'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]. These are the most significant values, 3 in each type will provide us with considerable grid search.
"""

from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
# param_grid_rf = {
#     'n_estimators': [50],
#     'max_depth': [None,],
#     'min_samples_split': [2]
# }
random_forest_model = RandomForestRegressor(random_state=42)

"""As fitting so many times (5 * 27 = 135), the following cell will take a lot of time:"""

cv_rf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_rf = GridSearchCV(random_forest_model, param_grid_rf, cv=cv_rf, n_jobs=-1, scoring='neg_mean_squared_error', verbose=4)
grid_search_rf.fit(X_train, y_train['CD36_Y'])

best_n_estimators = grid_search_rf.best_params_['n_estimators']
best_max_depth = grid_search_rf.best_params_['max_depth']
best_min_samples_split = grid_search_rf.best_params_['min_samples_split']

print(best_n_estimators, best_max_depth, best_min_samples_split)

"""(b) Provide a tabular summary of the cross-validation results of the methods in both models under consideration. (This comparison is why we make you use the same divisions.) Specify which model seems to be the best (justify your choice). Include a basic reference model for the comparison, which assigns the arithmetic mean of the dependent variable to any independent variable values."""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

def compute_stats(model, x_test, y_test, plot=False):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Perform cross-validation to evaluate generalization performance
    cv_scores = cross_val_score(lasso_model, x_test, y_test, cv=5, scoring='neg_mean_squared_error')
    print("Compute stats:")
    mean_cv_score = np.mean(cv_scores)
    print(f'mse: {mse}, R2: {r2}, mean CV score: {-mean_cv_score}')

elasticnet_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
elasticnet_model.fit(X_train, y_train)
compute_stats(elasticnet_model, X_train, y_train)

random_forest_model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                            min_samples_split=best_min_samples_split, random_state=42)
random_forest_model.fit(X_train, y_train)
compute_stats(random_forest_model, X_train, y_train)

# Calculate mean for the reference model
reference_model_prediction = y_train.mean()
reference_model_validation_error = mean_squared_error(y_validation, [reference_model_prediction] * len(y_validation))

print(reference_model_validation_error)

"""4. Prediction on a test set (10p.) Use the training data to choose the ”best” predictive model, and then use it to predict values of the dependent variable in the test set. The methods of selecting and building the models, as well as the motivations behind such choices, should be described in the report. The number of points you earn will depend on the quality of prediction, measured by the root of the mean squared error, RMSE."""





from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_scaled, y_train_all, test_size=0.2, random_state=42)

alpha = 0.1
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_all, y_train_all)



coefficients = lasso_model.coef_
print(coefficients[coefficients != 0])
idx_biggest_coeffs = np.where(coefficients != 0)[0] # np.sort(np.abs(coefficients))[-100:]

print(idx_biggest_coeffs.astype(int))
biggest_coeffs = coefficients[idx_biggest_coeffs]
column_names = X_train_all.columns[idx_biggest_coeffs.astype(int)]
print(biggest_coeffs, column_names)
compute_stats(lasso_model, X_train_all, y_train_all)



# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.model_selection import cross_val_score

# model = LinearRegression()

# sfs_forward = SequentialFeatureSelector(model, n_features_to_select=None, direction='forward', cv=10)
# sfs_forward.fit(X_train, y_train)
# selected_feature_indices = sfs_forward.get_support(indices=True)
# selected_features = X_train.columns[selected_feature_indices]
# print("Selected features from forward model selection:", selected_features)

features_selected = X_train[selected_features]

features_selected_model = sm.OLS(y_train_all, features_selected).fit()

print(features_selected_model.summary())

