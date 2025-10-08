import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  


st.set_page_config(page_title='IPL Performance Analysis Portal', layout='wide')
st.title('IPL Performance Analysis Portal')
@st.cache_data
def load_data():
    path = "Dataset/ipl_bbb_2021_25.csv"
    df = pd.read_csv(path)
    return df

df = load_data()
