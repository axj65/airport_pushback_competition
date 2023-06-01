import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from typing import Any
from loguru import logger


#%%Load Model Function
def load_model(solution_directory: Path) -> Any:
    """Load a specific model asset from disk."""
    airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA', 'KATL_na', 'KCLT_na', 'KDEN_na', 'KDFW_na', 'KJFK_na', 'KMEM_na', 'KMIA_na', 'KORD_na', 'KPHX_na', 'KSEA_na']
    models = {}
    for airport in airports:
        model_filename = f"{airport}_model.txt"
        model = lgb.Booster(model_file=str(solution_directory / model_filename))
        models[airport] = model    


    return models
#%%


def predict(
    config: pd.DataFrame,
    etd: pd.DataFrame,
    first_position: pd.DataFrame,
    lamp: pd.DataFrame,
    mfs: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tbfm: pd.DataFrame,
    tfm: pd.DataFrame,
    airport: str,
    prediction_time: pd.Timestamp,
    partial_submission_format: pd.DataFrame,
    model: Any,
    solution_directory: Path,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time."""
    logger.debug("Computing prediction")
    
    #Handles when partial_submission_format is emtpy
    if len(partial_submission_format) == 0:
        return partial_submission_format

    #%% Clean Data Function
    def clean_data(pushback, etd, lamp, mfs):

        #Make copies of all data so that dont throw errors
        etd=etd.copy()
        pushback = pushback.copy()
        lamp = lamp.copy()
        #Ensure timestampes are formatted correctly
        pushback['timestamp'] = pd.to_datetime(pushback['timestamp'])
        lamp['timestamp'] = pd.to_datetime(lamp['timestamp'])
        lamp['forecast_timestamp'] = pd.to_datetime(lamp['forecast_timestamp'])
        etd['timestamp'] = pd.to_datetime(etd['timestamp'])
        
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
        #Drop timestamp column
        lamp = lamp.drop('timestamp', axis=1)
        #Filter timestamps in lamp that I want to merge with pushback
        lamp1 = lamp[lamp['timestamp_plus_30'] == lamp['forecast_timestamp']]
        lamp2 = lamp[lamp['timestamp_plus_45'] == lamp['forecast_plus_15']]
        lamp3 = lamp[lamp['timestamp_plus_60'] == lamp['forecast_plus_30']]
        lamp4 = lamp[lamp['timestamp_plus_75'] == lamp['forecast_plus_45']]

        # merge the dataframes based on multiple conditions
        # Inner Merge, note that this inner merge excludes all values in pushback where there is no lamp data 
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
            
        #Merges etd onto pushback
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

        return pushback
    #%%

    #%% Onehot encoding Function
    def onehot (pushback):
        
        #One-hot encoding
        categorical_cols = ['cloud', 'lightning_prob', 'precip', 'day_of_week', 'month', 'aircraft_engine_class', 'flight_type', 'aircraft_type', 'major_carrier', 'year']

        pushback_onehot = pushback

        for col in categorical_cols:
            pushback_onehot[col] = pushback_onehot[col].astype('category')

        return pushback_onehot
    #%%

    #%%
    #Clean Test Data and onehot encode test data
    pushback = clean_data(partial_submission_format, etd, lamp, mfs)
    airport_test = onehot(pushback)
    #%%
    

    #%% Load the Correct model based on if weather data is present
    if airport_test.isna().any().any():
        my_model = model[airport + '_na']
        airport_test = airport_test.dropna(axis=1)
    else:
        my_model = model[airport]
    #%%


    #Create response and predictor variables
    X = airport_test.drop('minutes_until_pushback', axis=1)
    X = X.drop('gufi', axis=1)

    #Predict
    prediction = partial_submission_format.copy()
    y_pred = my_model.predict(X).round().astype(int)

    #Add predictions to partial_prediction_format
    prediction['minutes_until_pushback'] = y_pred
    return prediction

    #%%