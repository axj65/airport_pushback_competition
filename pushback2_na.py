#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
import json
from pathlib import Path
from typing import Any
#%%

#%% Read in Data
#Read in data for train data
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

#%% Clean Function
def clean_data(pushback, etd, lamp, mfs):
    #Take out unwanted columns
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
    # Drop timestamp column
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
    #Concat 
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

    #Merges etd ont pushback
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

    # Create new variables for day of week, month, hour, and year
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

    return pushback
#%%One Hot encoding
def onehot (pushback):
    #One-hot encoding
    categorical_cols = ['cloud', 'lightning_prob', 'precip', 'day_of_week', 'month', 'aircraft_engine_class', 'flight_type', 'aircraft_type', 'major_carrier', 'year']
    
    pushback_onehot = pushback
    for col in categorical_cols:
        pushback_onehot[col] = pushback_onehot[col].astype('category')
    #Drop Rows with NaN lamp data
    pushback_onehot = pushback_onehot.dropna(axis=1)
    return pushback_onehot
#%%

#%% Apply functions to data
#Clean Data
pushback = clean_data(pushback, etd, lamp, mfs)
#One-hot encode data
pushback_onehot = onehot(pushback)
#%%

#%%
benchmark = mean_absolute_error(pushback['minutes_until_pushback'], pushback['benchmark_pushback'])
#%%

#%% assign parameters
params = {
    'boosting_type': 'goss',
    'objective': 'regression',
    'metric': 'mae',
    'max_bins' : 1000,
    'verbose': 10,
    'min_data_in_leaf' : 200,
    'n_jobs': -1,
    'learning_rate': 0.08369562410523884,
    'max_depth': 13,
    'num_leaves': 360,
    'reg_alpha' : 0.3254746622061918,
    'reg_lambda' : 0.3196716454118168,
    'feature_fraction' : 0.8360831187256348,
    'feature_pre_filter' : False
}
#%%

#%% Model
X = pushback_onehot.drop('minutes_until_pushback', axis=1)
X = X.drop('gufi', axis=1)
y = pushback_onehot['minutes_until_pushback']

gbm_model_final = lgb.train(params,
                lgb.Dataset(X, y),
                num_boost_round = 1000)
#%%

#%% Test Model

#Read in data for test data
submission_format = f'data/code_execution_development_data/test_labels.csv'
pushback_predict = pd.read_csv(submission_format, parse_dates=['timestamp'])
pushback_predict = pushback_predict[pushback_predict['airport'] == f"{airport}"]

etd_label = f'data/code_execution_development_data/{airport}/{airport}_etd.csv.bz2'
etd = pd.read_csv(etd_label, compression='bz2', parse_dates=['timestamp', 'departure_runway_estimated_time'])

lamp_label = f'data/code_execution_development_data/{airport}/{airport}_lamp.csv.bz2'
lamp = pd.read_csv(lamp_label, compression='bz2', parse_dates=['timestamp', 'forecast_timestamp'])

mfs_label = f'data/code_execution_development_data/{airport}/{airport}_mfs.csv.bz2'
mfs = pd.read_csv(mfs_label, compression='bz2')

#Clean Test Data
pushback_predict = clean_data(pushback_predict, etd, lamp, mfs)
pushback = pushback.assign(test=0)
pushback_predict = pushback_predict.assign(test=1)
pushback_combined = pd.concat([pushback, pushback_predict])
pushback_combined = onehot(pushback_combined)
pushback_predict_onehot = pushback_combined[pushback_combined['test']==1]
pushback_predict_onehot = pushback_predict_onehot.drop('test', axis=1)
#%%

#%%
# define the features and target variables
X = pushback_predict_onehot.drop('minutes_until_pushback', axis=1)
X_test = X.drop('gufi', axis=1)
y = pushback_predict_onehot['minutes_until_pushback']

#Predict
y_pred_gbm = gbm_model_final.predict(X_test, num_iteration=gbm_model_final.best_iteration)
model_mae = mean_absolute_error(y, y_pred_gbm)

print("Benchmark MAE", benchmark)
print("Model MAE:",  model_mae)
#%%

#Save the model into a .txt file
gbm_model_final.save_model(f"{airport}_na_model.txt")


