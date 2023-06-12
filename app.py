import streamlit as st
import pandas as pd
from datetime import timedelta
from catboost import CatBoostRegressor

from src.utils import preprocess_fn, prepare_base_date, datetime_features, date_like_features_func, usd_normalizer, lag_features, read_models, predict_fn, download_button
import plotly.graph_objects as go
from src.constants import CFG
from PIL import Image
import base64

cash_flow = pd.DataFrame()
brent = pd.DataFrame()
usd = pd.DataFrame()

st.set_page_config(page_title="Shell Datathon 2023")

# Logo
col1, col2, col3 = st.columns([4, 1, 4])
with col1:
    st.write(' ')
with col2:
    image = Image.open("./src/images/shell.png")
    st.image(image)
with col3:
    st.write(' ')

# Title
st.markdown("<h2 style='text-align:center;'>Shell Datathon 2023 Cashflow Inference Tool</h2>", unsafe_allow_html=True)
uploader_holder = st.empty()
uploaded_csv_files = uploader_holder.file_uploader("Son 70 iş gününde gerçekleşen inflow-outflow, USD ve Brent verilerini içeren **.csv** formatlı dosyaları yükleyiniz.",
                                                   accept_multiple_files=True)


# File Upload
if uploaded_csv_files is not None:
    dfs = []
    for uploaded_file in uploaded_csv_files:
        # Sanity Check
        try:
            dfs.append(pd.read_csv(uploaded_file))
        except:
            raise "Dosya formatını kontrol ediniz."

    # File-format Check
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

    # Successful Upload Routine
    if not (cash_flow.empty or usd.empty or brent.empty):
        uploader_holder.empty()

        load_success_holder = st.empty()
        load_success_holder.success('Veriler başarılı bir şekilde yüklendi!', icon="✅")
            
        brent, usd, cash_flow = preprocess_fn(brent, usd, cash_flow)
        base_date, date_features, date_like_features = prepare_base_date(cash_flow)
        
        # Merge dfs to base_date
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
        inference_df = df.drop(shift_cols + exclude_cols, axis=1)

        # NaN handling for lag features
        inference_df = inference_df.iloc[max(lags):,:].tail(CFG.forecast_period+2).reset_index(drop=True)
        inference_df[CFG.scaler_col] = inference_df[f"lag_{CFG.forecast_period}_USD ALIŞ"]
        inference_df[CFG.scaler_col] = inference_df[CFG.scaler_col].ffill()

        # Inference
        predict_holder = st.empty()
        if predict_holder.button('Tahmin Et'):
            with st.spinner(text="Tahmin ediliyor..."):
                load_success_holder.empty()
                inflow_models, outflow_models = read_models(zip_path='./model/models.zip')
                print("Modeller başarıyla yüklendi.")
                print("Inference shape: ", inference_df.shape)
                inflow_forecast, outflow_forecast = predict_fn(inflow_models, inference_df),\
                    predict_fn(outflow_models, inference_df)
                target_forecast = inflow_forecast + outflow_forecast
                predict_holder.empty()

                inference_df_output = pd.DataFrame(df.Date.tail(len(target_forecast)))
                inference_df_output['Predicted Total Inflow'] = inflow_forecast
                inference_df_output['Predicted Total Outflow'] = outflow_forecast
                inference_df_output['Net Cashflow from Operations'] = target_forecast
                # output_csv = inference_df_output.to_csv(index=False).encode('utf-8')

                download_button_str = download_button(object_to_download=inference_df_output,
                                download_filename='forecast.csv',
                                button_text="Tahmini İndir")

                st.markdown(download_button_str, unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["Inflow Tahmini", "Outflow Tahmini", "Net Cashflow Tahmini"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.Date.tail(len(inflow_forecast)),
                                             y=inflow_forecast,
                                             mode='lines',
                                             name='Inflow Tahmini',
                                             line=dict(color="red")))
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.Date.tail(len(outflow_forecast)),
                                             y=outflow_forecast,
                                             mode='lines',
                                             name='Outflow Tahmini',
                                             line=dict(color="red")))
                    st.plotly_chart(fig, use_container_width=True)
                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.Date.tail(len(target_forecast)),
                                             y=target_forecast,
                                             mode='lines',
                                             name='Net Cashflow Tahmini',
                                             line=dict(color="red")))
                    st.plotly_chart(fig, use_container_width=True)

