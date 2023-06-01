
#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from hyperopt import fmin, tpe, hp, STATUS_OK
#%%

#%% 
#Read in data
airport = 'KATL'
train_label = f'data/prescreened_train_labels/prescreened_train_labels_{airport}.csv.bz2'
pushback = pd.read_csv(train_label, compression='bz2', parse_dates=['timestamp'])
etd_label = f'data/{airport}/{airport}_etd.csv.bz2'
etd = pd.read_csv(etd_label, compression='bz2', parse_dates=['timestamp', 'departure_runway_estimated_time'])
lamp_label = f'data/{airport}/{airport}_lamp.csv.bz2'
lamp = pd.read_csv(lamp_label, compression='bz2', parse_dates=['timestamp', 'forecast_timestamp'])
mfs_label = f'data/{airport}/{airport}_mfs.csv.bz2'
mfs = pd.read_csv(mfs_label, compression='bz2')
#%%

#%%
#Clean Data
lamp = lamp.drop('wind_direction', axis=1)
mfs = mfs[mfs['isdeparture'] == True]

#Make new timestamps and forecast timestamps to prep for merges
lamp['timestamp_plus_30'] = lamp['timestamp'] + pd.Timedelta(minutes=30)
lamp['timestamp_plus_45'] = lamp['timestamp'] + pd.Timedelta(minutes=45)
lamp['timestamp_plus_60'] = lamp['timestamp'] + pd.Timedelta(minutes=60)
lamp['timestamp_plus_75'] = lamp['timestamp'] + pd.Timedelta(minutes=75)

lamp['forecast_plus_15'] = lamp['forecast_timestamp'] + pd.Timedelta(minutes=15)
lamp['forecast_plus_30'] = lamp['forecast_timestamp'] + pd.Timedelta(minutes=30)
lamp['forecast_plus_45'] = lamp['forecast_timestamp'] + pd.Timedelta(minutes=45)

#Drop timestamp column
lamp = lamp.drop('timestamp', axis=1)

#Filter timestamps in lamp that I want to merge with pushback
lamp1 = lamp[lamp['timestamp_plus_30'] == lamp['forecast_timestamp']]
lamp2 = lamp[lamp['timestamp_plus_45'] == lamp['forecast_plus_15']]
lamp3 = lamp[lamp['timestamp_plus_60'] == lamp['forecast_plus_30']]
lamp4 = lamp[lamp['timestamp_plus_75'] == lamp['forecast_plus_45']]

# merge the dataframes based on multiple conditions
# Inner Merge, note that this inner merge exlucdes all values in pushback where there is no lamp data 
merge1 = pd.merge(pushback, lamp1, how='inner', left_on=['timestamp'], right_on=['timestamp_plus_30'])
merge2 = pd.merge(pushback, lamp2, how='inner', left_on=['timestamp'], right_on=['timestamp_plus_45'])
merge3 = pd.merge(pushback, lamp3, how='inner', left_on=['timestamp'], right_on=['timestamp_plus_60'])
merge4 = pd.merge(pushback, lamp4, how='inner', left_on=['timestamp'], right_on=['timestamp_plus_75'])
merged_df = pd.concat([merge1, merge2, merge3, merge4], ignore_index=True)

#Get the rows from pushback with no lamp data (lamp data is filled with nan)
    #Take the columns from the merged_df that I want to merge with pushback
temp1 = merged_df[['gufi', 'timestamp', 'minutes_until_pushback']]
    #Outer merge pushback with temp1 so I can see which rows are in pushback but not in merged_df
temp2 = pd.merge(pushback, temp1, how='outer', on=['gufi', 'timestamp', 'minutes_until_pushback'], indicator=True)
    #Get only the rows where lamp data is not available
result_df = temp2[temp2['_merge'] == 'left_only'].drop('_merge', axis=1)
    #Now I have all original rows from the original pushback, and if lamp data is not available, those are filled with nan
pushback = pd.merge(merged_df, result_df, on=['gufi', 'timestamp', 'airport', 'minutes_until_pushback'], how='outer', indicator=True).drop('_merge', axis=1)
    #Remove columns I added for merging purposes
pushback = pushback[['gufi', 'timestamp', 'airport', 'minutes_until_pushback', 'forecast_timestamp', 'temperature', 'wind_speed', 'wind_gust', 'cloud_ceiling', 'visibility', 'cloud', 'lightning_prob', 'precip']]

#Merge mfs onto pushback
    #drop unnecessary column    
mfs = mfs.drop('isdeparture', axis=1)
    #Merge
pushback = pd.merge(pushback, mfs, how='left', left_on=['gufi'], right_on=['gufi'])

#Rounds timestamp to nearest 15 minutes
etd['rounded_timestamp'] = etd['timestamp'].dt.round('15min')

#I want all values to round up so if rounded_timestamp is less than timestamp, it adds 15 minutes
etd.loc[etd['timestamp'] > etd['rounded_timestamp'], 'rounded_timestamp'] += pd.Timedelta(minutes=15)

# Drop Duplicates and keep more recent predicted time 
    # Sort etd by timestamp in descending order (note that drop_duplicates keeps first occurence)
etd = etd.sort_values('timestamp', ascending=False)
    # Drop duplicates
etd.drop_duplicates(inplace=True, subset=['gufi', 'rounded_timestamp'])

#Drop timestamp column 
etd = etd.drop('timestamp', axis=1)
etd = etd.reset_index(drop=True)

#Merge etd onto pushback
    #Left Merge pushback and etd on gufi
pushback = pd.merge(pushback, etd, on='gufi', how='left')
    #Take out all observations that rounded_timestamp occurs after timestamp
pushback = pushback[pushback['timestamp'] >= pushback['rounded_timestamp']]
    #Sort values and drop duplicates
pushback = pushback.sort_values('rounded_timestamp', ascending=False)
pushback.drop_duplicates(inplace=True, subset=['gufi', 'timestamp', 'minutes_until_pushback'])
    #Drop rounded_timestamp
pushback = pushback.drop('rounded_timestamp', axis=1)
pushback = pushback.sort_values(['gufi', 'timestamp'])
pushback = pushback.reset_index(drop=True)

# Pushback is typically about 15 minutes before departure so
    #Estimate the time pushback occurs
pushback['benchmark_pushback_estimated_time'] = pushback['departure_runway_estimated_time'] - pd.Timedelta(minutes=15)
    #Create a benchmark time
pushback['benchmark_pushback'] = (pushback['benchmark_pushback_estimated_time'] - pushback['timestamp']) / pd.Timedelta(minutes=1)

# Create new variables for day of week and month
pushback['day_of_week'] = pushback['departure_runway_estimated_time'].dt.day_name()
pushback['month'] = pushback['departure_runway_estimated_time'].dt.strftime('%B')
pushback['hour'] = pushback['departure_runway_estimated_time'].dt.hour
pushback['year'] = pushback['departure_runway_estimated_time'].dt.year

#Fill categorical NA 
pushback['aircraft_type'].fillna('OTHER', inplace=True)
pushback['flight_type'].fillna('OTHER', inplace=True)
pushback['major_carrier'].fillna('OTHER', inplace=True)

#drop uneccessary variables
drop_cols = ['timestamp', 'airport', 'forecast_timestamp', 'departure_runway_estimated_time', 'benchmark_pushback_estimated_time']
pushback = pushback.drop(drop_cols, axis=1)
pushback = pushback.reset_index(drop=True)
#%%

#%%
#One-hot encoding
categorical_cols = ['cloud', 'lightning_prob', 'precip', 'day_of_week', 'month', 'aircraft_engine_class', 'flight_type', 'aircraft_type', 'major_carrier', 'year']

pushback_onehot = pushback
for col in categorical_cols:
    pushback_onehot[col] = pushback_onehot[col].astype('category')

#Drop Rows with NaN lamp data
pushback_onehot = pushback_onehot.dropna()
#%%


#%% 
#Train-test Split
# define the features and target variables
X = pushback_onehot.drop(['minutes_until_pushback', 'gufi'], axis=1)
y = pushback_onehot['minutes_until_pushback']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

#Create train and test data for gbm model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
#%%

#%%
#Get baseline mae
baseline_mae = mean_absolute_error(pushback['minutes_until_pushback'], pushback['benchmark_pushback'])
baseline_mae2 = mean_absolute_error(y_test, X_test['benchmark_pushback'])
#%%

#%%
#make integers flaots
pushback_onehot['hour'] = pushback_onehot['hour'].astype(float)
pushback_onehot['minutes_until_pushback'] = pushback_onehot['minutes_until_pushback'].astype(float)
#%%

#%%Tuning
# Define the hyperparameter search space
depth_start = 10

space = {
    'num_leaves': hp.quniform('num_leaves', 400, 1000, 25),
    'learning_rate': hp.loguniform('learning_rate', -4, 0),
    'max_depth': hp.choice('max_depth', range(depth_start, 21)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1)
}

# Define the objective function
def objective(params):
    # Convert integer parameters to integer values
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['n_jobs'] = -1
    params['min_data_in_leaf'] = 250
    params['boosting'] = 'goss'
    params['feature_pre_filter'] = False
    
    
    # Train the LightGBM model with the specified hyperparameters
    model = lgb.train(params, train_data, valid_sets=test_data, num_boost_round=1000, early_stopping_rounds=10) #, verbose_eval=False)
    
    # Predict on the validation set and compute the mean squared error
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Return the loss to be minimized
    return {'loss': mae, 'status': STATUS_OK}

# Define the optimization algorithm
tpe_algo = tpe.suggest

# Run the optimization
best = fmin(objective, space, algo=tpe_algo, max_evals=25)

# Print the best hyperparameters
print('Best hyperparameters:', best)

# Train the final model with the best hyperparameters
best_params = {
    'num_leaves': int(best['num_leaves']),
    'learning_rate': best['learning_rate'],
    'max_depth': int(best['max_depth']) + depth_start,
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda'],
    'feature_fraction' : best['feature_fraction'],
    'min_data_in_leaf': 200,
    'n_jobs': -1,
    'boosting' : 'goss',
    'feature_pre_filter' : False
}
#%%

#%% 
# Final Model

final_model = lgb.train(best_params, train_data, valid_sets=test_data, num_boost_round=1000, early_stopping_rounds=10, verbose_eval=True)
train_predictions = final_model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_predictions)
final_predictions = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_predictions)

print("Baseline1: ", baseline_mae)
print("Baseline2", baseline_mae2)
print("Train mae:", train_mae)
print("Final mae:", final_mae)
print(best['max_depth'] + depth_start)
print(best)

#%%
