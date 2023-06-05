import pandas as pd
import numpy as np

def datetime_features(df_temp):
    """
    Generates calendar features (both generic and business related ones)
    """
    df_temp['month'] = df_temp['Date'].dt.month
    df_temp['year'] = df_temp['Date'].dt.year
    df_temp['dayofweek'] = df_temp['Date'].dt.dayofweek
    df_temp['dayofyear'] = df_temp['Date'].dt.dayofyear
    df_temp['quarter'] = df_temp['Date'].dt.quarter
    df_temp['dayofmonth'] = df_temp['Date'].dt.day
    df_temp['weekofyear'] = df_temp['Date'].dt.weekofyear
    df_temp['month_middle'] = np.where((df_temp['Date'].dt.day == 15), 1, 0)
    df_temp['month_end'] = np.where(df_temp['Date'].dt.is_month_end, 1, 0)
    df_temp['month_start'] = np.where(df_temp['Date'].dt.is_month_start, 1, 0)
    
    df_temp['is_weekday'] = (df_temp["dayofweek"] <= 4).astype(int)

    df_temp['nth_weekday_of_month'] = df_temp.groupby(["year", "month"])["is_weekday"].cumsum() + 1
    df_temp['nth_weekday_of_quarter'] = df_temp.groupby(["year", "quarter"])["is_weekday"].cumsum() + 1
    df_temp['nth_weekday_of_year'] = df_temp.groupby(["year"])["is_weekday"].cumsum() + 1
    
    df_temp.drop(labels=["is_weekday", "year"], axis=1, inplace=True)
    
    df_temp.loc[df_temp.dayofweek==0,'weekfirstdate'] = 1
    df_temp.loc[df_temp.dayofweek==4,'weeklastdate'] = 1
    
    df_temp[['weekfirstdate','weeklastdate']] =\
        df_temp[['weekfirstdate','weeklastdate']].fillna(0).astype(int)
    
    date_features = ['month','dayofweek','quarter','dayofmonth', 'dayofyear', 'weekofyear',
                     'month_middle','month_end', 'month_start', 'weekfirstdate','weeklastdate',
                     'nth_weekday_of_month', 'nth_weekday_of_quarter', 'nth_weekday_of_year'
                    ]
    
    return df_temp,date_features

def lag_features(df_temp,
                 columns,
                 lags):
    
    for col in columns:
        for lag in lags:
            df_temp[f'lag_{lag}_{col}'] = df_temp[col].shift(lag)
    return df_temp