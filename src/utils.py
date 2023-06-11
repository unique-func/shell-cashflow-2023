import pandas as pd
import numpy as np
from zipfile import ZipFile
from catboost import CatBoostRegressor


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
    df_temp['weekofyear'] = df_temp['Date'].dt.isocalendar().week
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
    
    df_temp[['weekfirstdate','weeklastdate']] = df_temp[['weekfirstdate','weeklastdate']].fillna(0).astype(int)
    
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

def date_like_features_func(df_temp):
    
    date_like_features = []
    for i in [1,2,3]:
         for col in [
             'month_end',
             'month_start',
             'month_middle',
             'weekfirstdate',
             'weeklastdate',
         ]:
                df_temp[f"is_{col}_in_next_{i}_days"] = df_temp[col].rolling(i).sum().shift(-i).fillna(-1)
                df_temp[f"is_{col}_in_past_{i}_days"] = df_temp[col].rolling(i).sum().shift(1).fillna(-1)
                date_like_features.extend([f"is_{col}_in_next_{i}_days" , f"is_{col}_in_past_{i}_days"])
    return df_temp, date_like_features
                
def usd_normalizer(df_temp):
    df_temp["usd_diff"] = df_temp["USD SATIŞ"] - df_temp["USD ALIŞ"]
    df_temp["eur_diff"] = df_temp["EUR SATIŞ"] - df_temp["EUR ALIŞ"]
    df_temp["gbp_diff"] = df_temp["GBP SATIŞ"] - df_temp["GBP ALIŞ"]
    
    df_temp["ab_mult_usd"] = df_temp["AB Piyasa Fiyatı"] * df_temp["Dolar Kuru (Satış)"]
    df_temp["ab_gap"] = df_temp["AB Piyasa Fiyatı- Yüksek"] - df_temp["AB Piyasa Fiyatı- Düşük"]
    df_temp["ab_gap_mult_usd"] = df_temp["ab_gap"] - df_temp["Dolar Kuru (Satış)"]
    
    for col in ['Customers - DDS', 'Customers - EFT', 'T&S Collections',
            'FX Sales', 'Other operations', 'Tüpraş', 'Other Oil',
            'Gas', 'Import payments (FX purchases)', 'Tax',
            'Operatioınal and Admin. Expenses', 'VIS Buyback Payments',
            'Total Inflows', 'Total Outflows']:
    
        df_temp[col] /= df_temp["USD ALIŞ"]
        
    return df_temp

# opening the zip file in READ mode
def read_models(zip_path):
    """
    Read models from zip
    """
    with ZipFile(zip_path, 'r') as zip:
        all_model_names = zip.namelist()
        inflow_model_names = [model for model in all_model_names if 'inflow' in model]
        outflow_model_names = [model for model in all_model_names if 'outflow' in model]
        
        inflow_models = [zip.read(model_name) for model_name in inflow_model_names]
        outflow_models = [zip.read(model_name) for model_name in outflow_model_names]
        
    return inflow_models, outflow_models


def predict_fn(models, inference_df):
    """
    load from blob and predict future values using catboost models
    """
    preds = []
    for model in models:
        cat = CatBoostRegressor()
        cat.load_model(blob=model)
        pred = (cat.predict(inference_df) * inference_df['ref_col']).values
        preds.append(pred)
    return np.mean(preds, axis=0)