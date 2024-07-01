
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import RobustScaler
import optuna
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 500)
# pd.set_option('display.float_format', lambda x: '%.5f' % x


train = pd.read_csv("African_Zindi/Datasets/Train_Africa.csv")
test = pd.read_csv("African_Zindi/Datasets/Test_Africa.csv")
df = train._append(test, ignore_index=True)
train.head()
train.shape
test.shape


##################################
# Capturing Numeric and Categorical Variables
##################################

def grab_col_names(dataframe, cat_th=12, car_th=20):
    """
    Extract column names for a given dataframe.

    param dataframe: The dataframe to analyze.
    param cat_th: Threshold for numerical columns to be considered categorical.
    param car_th: Threshold for categorical columns to be considered as cardinal.
    return: Lists of categorical columns, categorical but cardinal columns, and numerical columns.
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() <= cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = number of variables.
    # num_but_cat is already included in cat_cols.
    # Therefore, all variables will be selected with these three lists: cat_cols + num_cols + cat_but_car.
    # num_but_cat is provided only for reporting purposes.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)




######################################
# Missing Value Analysis and Imputation
######################################

# Function to Create a Table of Missing Values
def missing_values_table(dataframe, na_name=True):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df)


num_cols = [col for col in num_cols if col not in "pm2_5"]

# Filling Missing Values by Site ID Over Time
for col in num_cols:
    for site_id in df.site_id.unique():
      df.loc[df["site_id"] == site_id, col] = df.loc[df["site_id"] == site_id, col].bfill().ffill()


na_columns = missing_values_table(df)

# Filling Remaining Missing Values with Mean
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

na_columns = missing_values_table(df)



######################################
# Feature Extraction
######################################

# Time Series Features
df["NEW_date"] = df["date"].astype(str) + "-" + df["hour"].astype(str)
df["NEW_date"] = pd.to_datetime(df["NEW_date"])
df['NEW_day'] = df['NEW_date'].dt.day
df["NEW_week"] = df["NEW_date"].dt.dayofweek
df["NEW_week_of_year"] = df['NEW_date'].dt.isocalendar().week
df['NEW_quarter'] = df['NEW_date'].dt.quarter
df['NEW_year'] = df['NEW_date'].dt.year
df['NEW_day_of_year'] = df['NEW_date'].dt.dayofyear
df["NEW_time_no"] = (df["NEW_date"] - df["NEW_date"].min()) // dt.timedelta(days=1)

df["NEW_start_dow_sin"] = np.sin(2 * np.pi * df["NEW_date"].dt.dayofweek / 6.0)
df["NEW_start_dow_cos"] = np.cos(2 * np.pi * df["NEW_date"].dt.dayofweek / 6.0)
df['NEW_start_hour_sin'] = np.sin(2 * np.pi * df["NEW_date"].dt.hour / 23.0)
df['NEW_start_hour_cos'] = np.cos(2 * np.pi * df["NEW_date"].dt.hour / 23.0)
df['NEW_start_month_sin'] = np.sin(2 * np.pi * df["NEW_date"].dt.month / 12)
df['NEW_start_month_cos'] = np.cos(2 * np.pi * df["NEW_date"].dt.month / 12)
df["NEW_dayofyear_sin"] = np.sin(2 * np.pi * df["NEW_day_of_year"] / df["NEW_day_of_year"].max())
df["NEW_dayofyear_cos"] = np.cos(2 * np.pi * df["NEW_day_of_year"] / df["NEW_day_of_year"].max())

df["is_weekend"] = df["NEW_date"].apply(lambda x: x.weekday() > 5).astype(int)
df["NEW_is_month_start"] = df.NEW_date.dt.is_month_start.astype(int)
df["NEW_is_month_end"] = df.NEW_date.dt.is_month_end.astype(int)
df["is_quarter_start"] = df["NEW_date"].dt.is_quarter_start.astype(int)
df["is_quarter_end"] = df["NEW_date"].dt.is_quarter_end.astype(int)
df["is_year_start"] = df["NEW_date"].dt.is_year_start.astype(int)
df["is_year_end"] = df["NEW_date"].dt.is_year_end.astype(int)


# Solar hour length according to coordinate information
from suntime import Sun
import pytz

tz = {"Nigeria" : pytz.timezone('Africa/Lagos'),
      "Kenya" : pytz.timezone('Africa/Nairobi'),
      "Burundi" : pytz.timezone('Africa/Bujumbura'),
      "Uganda" : pytz.timezone('Africa/Kampala'),
      "Ghana" : pytz.timezone('Africa/Accra'),
      "Cameroon": pytz.timezone('Africa/Douala')}

df['NEW_length_of_day'] = 0

for i in range(len(df)):
    sun = Sun(df.loc[i, 'site_latitude'], df.loc[i, 'site_longitude'])
    country = df.loc[i, "country"]
    sunrise = sun.get_sunrise_time(df.loc[i, 'NEW_date'], time_zone=tz[country]).hour
    sunset = sun.get_sunset_time(df.loc[i, 'NEW_date'], time_zone=tz[country]).hour
    df.loc[i, 'NEW_length_of_day'] = sunset - sunrise


# Season feature
def find_season(month):
    season_map = {12: 'WINTER', 1: 'WINTER', 2: 'WINTER',
                  3: 'SPRING', 4: 'SPRING', 5: 'SPRING',
                  6: 'SUMMER', 7: 'SUMMER', 8: 'SUMMER',
                  9: 'AUTUMN', 10: 'AUTUMN', 11: 'AUTUMN'}
    return season_map[month]

df['NEW_season'] = df['month'].apply(find_season)

# Country Indicator Columns and Data Cleanup
df['is_Nigeria'] = (df['country'] == 'Nigeria').astype(int)
df["is_Kenya"] = (df["country"] == "Kenya").astype(int)
df["is_Burundi"] = (df["country"] == "Burundi").astype(int)
df["is_Uganda"] = (df["country"] == "Uganda").astype(int)

df.drop(["country", "city", "NEW_date", "date", "id"], axis=1, inplace=True)


# Correlation-Based Feature Engineering and Quality Calculation
df_corr = df[num_cols].corrwith(df["pm2_5"]).sort_values(ascending=False)
df_corr = pd.DataFrame({"Variables" : df_corr.index,
                        "Values": df_corr.values})

corr_pos = df_corr.loc[(df_corr["Values"] > 0.1),"Variables"].values.tolist()
corr_neg = df_corr.loc[(df_corr["Values"] < -0.1),"Variables"].values.tolist()


for col in (corr_pos + corr_neg):
    df["NEW_Cat_" + col] = pd.qcut(df[col], q=5, labels=["verylow", "low", "medium", "high", "veryhigh"])

pos_category_mapping = {
    'verylow': 5,
    'low': 4,
    'medium': 3,
    'high': 2,
    'veryhigh': 1
}

for col in corr_pos:
  df['NEW_Cat_'+ col +'_num'] = df["NEW_Cat_" + col].map(pos_category_mapping)

neg_category_mapping = {
    'verylow': 1,
    'low': 2,
    'medium': 3,
    'high': 4,
    'veryhigh': 5
}

for col in corr_neg:
  df['NEW_Cat_'+ col +'_num'] = df["NEW_Cat_" + col].map(neg_category_mapping)


Numeric_features = df.columns[-26:].tolist()
df["NEW_Quality"] = df[Numeric_features].sum(axis=1)
df.drop(Numeric_features, axis=1, inplace=True)

for col in (corr_pos + corr_neg):
    df.drop("NEW_Cat_" + col, axis=1, inplace=True)

df.head()


##################
# Scaling & One - Hot Encoding
##################
cat_cols, cat_but_car, num_cols = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["pm2_5"]]
cat_cols = cat_cols + ["site_id"]

# Apply RobustScaler to Numerical Columns
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Apply One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape

##################################
# ROBUST PLS Modeling
##################################

# Note: This dataset, characterized by high noise, high correlation, and the presence of outliers, has led to the selection of
# the PLS model (Partial Least Squares) within RANSAC as an appropriate choice

train_df = df[df['pm2_5'].notnull()]
test_df = df[df['pm2_5'].isnull()]

y = train_df['pm2_5']
X = train_df.drop(["pm2_5"], axis=1)


##################################
# Optimization with Optuna
##################################

# def objective(trial):
#    min_samples = trial.suggest_int('min_samples', 1, int(X.shape[0] * 0.5)) #
#    residual_threshold = trial.suggest_float('residual_threshold', 1.0, 10.0)
#    n_components = trial.suggest_int('n_components', 1, min(X.shape[1], 20))
#    pls = PLSRegression(n_components=n_components)
#    ransac = RANSACRegressor(estimator=pls, min_samples=min_samples, residual_threshold=residual_threshold, random_state=42)

#    kf = KFold(n_splits=2, shuffle=True, random_state=42)
#    rmse_scores = []

#    for train_index, val_index in kf.split(X):
#        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
#        ransac.fit(X_train, y_train)
#        y_pred = ransac.predict(X_val)
#        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

#    return np.mean(rmse_scores)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=75)
# best_params = study.best_params

best_params = {'min_samples': 4032,
 'residual_threshold': 9.972481710177503,
  'n_components': 18}


################################################################
# KFold Validation ve Permutation Importance
################################################################

# Initialize KFold and Model
n_splits = 2
group_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

pls = PLSRegression(n_components=best_params['n_components'])
best_model = RANSACRegressor(estimator=pls, min_samples=best_params['min_samples'],
                             residual_threshold=best_params['residual_threshold'], random_state=42)

# Perform K-Fold Cross-Validation
fold_rmses = []
fold_rmses_train = []
test_preds = []
i=0
for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X)):
     # Split into training and testing sets
     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train the model
     best_model.fit(X_train, y_train)
     if i==0:
         perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, scoring='neg_mean_absolute_error',random_state=42)
     # Predict on the test set
     y_pred = best_model.predict(X_test)
     y_pred_train = best_model.predict(X_train)
     y_pred_test = best_model.predict(test_df.drop("pm2_5", axis=1))
     test_preds.append(y_pred_test)

     # Calculate the RMSE for the current fold
     fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     fold_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
     fold_rmses.append(fold_rmse)
     fold_rmses_train.append(fold_rmse_train)

     # Print RMSE for the current fold
     print(f"Fold {fold + 1} RMSE train: {fold_rmse_train:.4f}, RMSE test: {fold_rmse:.4f}")
     i+=1
print(f"\nMean train: {np.mean(fold_rmses_train):.4f}, Mean test: {np.mean(fold_rmses):.4f}")


# Calculate Feature Importances
feature_names = X.columns.tolist()
importances_df = pd.DataFrame({'Features': feature_names,
    'Importance Score Mean': perm_importance.importances_mean,
    'Importance Score Std': perm_importance.importances_std})

importances_df = importances_df.sort_values(by='Importance Score Mean', ascending=False)
feats_filter = importances_df[importances_df['Importance Score Mean'] > 0]['Features'].tolist()


#  Plot Feature Importances
def plot_feature_importance(importances_df, top_n=None):
    if top_n is not None:
        importances_df = importances_df.head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance Score Mean', y='Features', data=importances_df, palette='viridis')
    plt.title('Permutation Importance of Features')
    plt.xlabel('Importance Score Mean')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

plot_feature_importance(importances_df, top_n=70)



################################################################
# KFold Prediction
################################################################
X = train_df[feats_filter]
y = train_df["pm2_5"]

# Random States and Model Definition
random_states = [13, 31, 42, 69, 72, 99, 131, 55, 172, 27, 18, 230, 155, 118, 135, 169, 142, 11, 119, 61]
pls = PLSRegression(n_components=best_params['n_components'])
best_model = RANSACRegressor(estimator=pls, min_samples=best_params['min_samples'],
                                 residual_threshold=best_params['residual_threshold'], random_state=42)

# Defining Lists for Performance Metrics
train_rmse_scores = []
val_rmse_scores = []
test_preds = []
# Model Training
for random_state in random_states:
      kf = KFold(n_splits=2, shuffle=True, random_state=random_state)
      for train_index, val_index in kf.split(X):
          X_train, X_val = X.iloc[train_index], X.iloc[val_index]
          y_train, y_val = y.iloc[train_index], y.iloc[val_index]
          best_model.fit(X_train, y_train)
          # Predictions and Error Calculations
          y_train_pred = best_model.predict(X_train)
          y_val_pred = best_model.predict(X_val)
          y_test_pred = best_model.predict(test_df[feats_filter])
          train_rmse_scores.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
          val_rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
          test_preds.append(y_test_pred)
print(f'Mean Train RMSE: {np.mean(train_rmse_scores)}')
print(f'Mean Validation RMSE: {np.mean(val_rmse_scores)}')


# Note: To prevent overfitting and produce more generalizable predictions, we obtained predictions using 20 random states in a 2-split k-fold cross-validation
result_pred = np.mean(test_preds, axis=0)
result_pred = pd.DataFrame({"pm2_5_pred": result_pred})


##################################
# Submitting Predictions
#################################
result_df = pd.DataFrame({
    "id": test['id'],
    "pm2_5": result_pred["pm2_5_pred"]
})

result_df.head()
result_df.to_csv("African_Zindi_Predictions.csv.csv", index=False)

import os
os.getcwd()






