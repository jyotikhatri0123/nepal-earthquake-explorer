import io

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Nepal Earthquake Data Explorer", page_icon="🌏", layout="wide")


USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BBOX = dict(minlatitude=26.3, maxlatitude=30.5, minlongitude=80.0, maxlongitude=88.3)


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Downloading earthquake data from USGS (first load takes about a minute)...")
def load_data() -> pd.DataFrame:
    frames = []
    this_year = pd.Timestamp.now().year + 1
    for y in range(1950, this_year, 5):
        params = {"format": "csv", "starttime": f"{y}-01-01", "endtime": f"{min(y + 5, this_year)}-01-01",
                  "minmagnitude": 3.5, "orderby": "time-asc", **BBOX}
        r = requests.get(USGS_URL, params=params, timeout=60)
        if r.status_code == 204 or not r.text.strip():
            continue
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text)))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    df = df[["time", "latitude", "longitude", "depth", "mag", "magType", "place"]].dropna(
        subset=["mag", "latitude", "longitude"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["year"] = df["time"].dt.year
    return df

