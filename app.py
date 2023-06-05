import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor


st.set_page_config(page_title="Shell Datathon 2023")
st.markdown("<h1 style='text-align:center;'>Shell Datathon 2023 Cashflow Inference Tool</h1>",unsafe_allow_html=True)

uploaded_csv_file = st.file_uploader("Upload last 70 days USD file in '.csv' format")
if uploaded_csv_file is not None:
    usd = pd.read_csv(uploaded_csv_file)
    st.write(usd)

