import streamlit as st
import pandas as pd
from datetime import timedelta
from catboost import CatBoostRegressor
from pandas.tseries.offsets import BDay
from src.utils import datetime_features, date_like_features_func, usd_normalizer, lag_features, read_models, predict_fn
import plotly.graph_objects as go
#from src.constants import CFG
class CFG:
    target = 'Net Cashflow from Operations'
    targets = ['Total Outflows', 'Total Inflows']
    cash_flow_columns = ['Date','Total Inflows','Customers - DDS','Customers - EFT','T&S Collections','FX Sales','Other operations','Total Outflows','Tüpraş','Other Oil','Gas','Import payments (FX purchases)','Tax','Operatioınal and Admin. Expenses','VIS Buyback Payments','Net Cashflow from Operations']
    usd_columns = ['Yıl','Tarih','USD ALIŞ','USD SATIŞ','EUR ALIŞ','EUR SATIŞ','GBP ALIŞ','GBP SATIŞ']
    brent_columns = ['Tarih','Ürün','Avrupa Birliği Para Birimi','AB Piyasa Fiyatı','AB Piyasa Fiyatı- Yüksek','AB Piyasa Fiyatı- Düşük','Dolar Kuru (Satış)']
    forecast_period = 70
    scaler_col = 'ref_col'
    lags = [forecast_period]
    
cash_flow = pd.DataFrame()
brent = pd.DataFrame()
usd = pd.DataFrame()

st.set_page_config(page_title="Shell Datathon 2023")
st.markdown("<h1 style='text-align:center;'>Shell Datathon 2023 Cashflow Inference Tool</h1>", unsafe_allow_html=True)
uploaded_csv_files = st.file_uploader("Son 70 günde gerçekleşen inflow-outflow, USD ve Brent verilerini içeren '.csv' formatlı dosyaları yükleyiniz.", accept_multiple_files=True)

if uploaded_csv_files is not None:
    dfs = []
    for uploaded_file in uploaded_csv_files:
        try:
            dfs.append(pd.read_csv(uploaded_file))
        except:
            raise "Dosya formatını kontrol ediniz."
        
    for df_temp in dfs:
        if CFG.target in df_temp.columns:
            cash_flow = df_temp.copy()
            cash_flow = cash_flow[CFG.cash_flow_columns]
            print(f"cash_flow datası başarıyla yüklendi, shape: {cash_flow.shape}")
        elif CFG.usd_columns[0] in df_temp.columns.tolist():
            usd = df_temp.copy()
            usd = usd[CFG.usd_columns]
            print(f"usd datası başarıyla yüklendi, shape: {usd.shape}")
        elif CFG.brent_columns[0] in df_temp.columns.tolist():
            brent = df_temp.copy()
            brent = brent[CFG.brent_columns]
            print(f"brent datası başarıyla yüklendi, shape: {brent.shape}")
        else:
            st.error("Hatalı giriş yapıldı, lütfen yüklenen csv uzantılı dosyalardaki sütun isimlerini kontrol ediniz.")
    
    if not (cash_flow.empty and usd.empty and brent.empty):
        st.success('Veriler başarılı bir şekilde yüklendi!', icon="✅")
        brent = brent.rename(columns={'Tarih':'Date'})
        usd = usd.rename(columns={'Tarih':'Date'})

        cash_flow['Date'] = pd.to_datetime(cash_flow['Date'])
        brent['Date'] = pd.to_datetime(brent['Date'])
        usd['Date'] = pd.to_datetime(usd['Date'])
        brent = brent.drop(['Ürün', 'Avrupa Birliği Para Birimi'],axis=1)
        usd = usd.drop(['Yıl'], axis=1)
        cash_flow = cash_flow.sort_values('Date').reset_index(drop=True)
        
        print(f"Yüklenen data başlangıç tarihi: {cash_flow['Date'].iloc[0]}")
        
        forecast_start_date = (cash_flow['Date'].iloc[-1] + BDay(1)).strftime('%Y-%m-%d')
        forecast_end_date = (cash_flow['Date'].iloc[-1] + BDay(CFG.forecast_period+2)).strftime('%Y-%m-%d')
        print(f'Forecast Start Date: {forecast_start_date},Forecast End Date:{forecast_end_date}')

        base_date = pd.DataFrame(pd.date_range(start=cash_flow['Date'].iloc[0], end=forecast_end_date, freq="D"),columns=['Date'])
        base_date, date_features = datetime_features(base_date)
        print(date_features)
        
        base_date, date_like_features = date_like_features_func(base_date)
        base_date = base_date[base_date.dayofweek<=4].reset_index(drop=True)
        
        
        df = base_date.merge(cash_flow, how='left', on='Date')
        df = df.merge(brent, how='left', on='Date')
        df = df.merge(usd, how='left', on='Date')
        df = usd_normalizer(df)
        
        shift_cols = df.drop(['Date'] + date_features + date_like_features, axis=1).columns.to_list()

        lags = [CFG.forecast_period]
        df = lag_features(df_temp=df,
                          columns=shift_cols,
                          lags=CFG.lags
                 )
        exclude_cols = ['Date', CFG.target] + CFG.targets
        inference_df = df.drop(shift_cols + exclude_cols,axis=1)
        # NaN handling for lag features
        inference_df = inference_df.iloc[max(lags):,:].tail(CFG.forecast_period+2).reset_index(drop=True)
        
        inference_df[CFG.scaler_col] = inference_df[f"lag_{CFG.forecast_period}_USD ALIŞ"]
        inference_df[CFG.scaler_col] = inference_df[CFG.scaler_col].ffill()
        
        #Inference
        inflow_models,outflow_models = read_models(zip_path='./model/models.zip')
        print("Modeller başarıyla yüklendi.")
        print("Inference shape: ", inference_df.shape)
        inflow_forecast, outflow_forecast = predict_fn(inflow_models,inference_df), predict_fn(outflow_models,inference_df)
        target_forecast = inflow_forecast + outflow_forecast
        
        #st.dataframe(inference_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date.tail(len(target_forecast)),
                                    y=target_forecast,
                                    mode='lines',
                          name='Prediction'))
        st.plotly_chart(fig, use_container_width=True)
