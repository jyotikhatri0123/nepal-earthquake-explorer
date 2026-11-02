import io

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Nepal Earthquake Data Explorer", page_icon="🌏",
                   layout="centered", initial_sidebar_state="collapsed")

# Phone-friendly spacing and smaller headline numbers
st.markdown(
    """<style>
.block-container{padding-top:1.5rem;padding-left:1rem;padding-right:1rem}
h1{font-size:1.7rem !important;line-height:1.2}
[data-testid="stMetricValue"]{font-size:1.5rem}
</style>""",
    unsafe_allow_html=True,
)

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BBOX = dict(minlatitude=26.3, maxlatitude=30.5, minlongitude=80.0, maxlongitude=88.3)
MIN_MAG = 3.5
START_YEAR = 1950

# Rough, simplified outline of Nepal as (longitude, latitude). NOT an official border.
NEPAL_OUTLINE = [(80.06, 28.83), (80.5, 29.7), (81.1, 30.2), (81.9, 30.35), (82.7, 29.9),
                 (83.5, 29.3), (84.2, 28.9), (85.2, 28.35), (85.7, 28.3), (86.4, 28.15), (87.0, 27.97),
                 (88.15, 27.9), (88.1, 26.5), (87.3, 26.4), (86.0, 26.65), (84.7, 27.0),
                 (83.3, 27.4), (82.0, 28.1), (81.3, 28.35), (80.5, 28.6)]

# West-to-east longitude bands (approximate; these are NOT official provinces)
ZONE_BINS = [80.0, 81.5, 83.0, 84.5, 86.0, 87.5, 88.3]