import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="IPL Broadcast Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    path = "Dataset/ipl_bbb_2021_25.csv"
    df = pd.read_csv(path)
    return df

df = load_data()

st.title("📊 IPL 2021–2025 Broadcast Dashboard")
st.write("Data loaded successfully. Preview below 👇")

st.dataframe(df.head())
