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


def mag_color(m: float) -> str:
    if m >= 6:
        return "#b71c1c"
    if m >= 5:
        return "#f57c00"
    if m >= 4:
        return "#fbc02d"
    return "#7cb342"


st.title("🌏 Nepal Earthquake Data Explorer")
st.caption(
    "Historical records from the USGS earthquake catalog. "
    "This dashboard describes past data only. It does not predict future earthquakes."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not download data from USGS. Please refresh the page in a minute. ({e})")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_d, max_d = df["time"].min().date(), df["time"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
mag_range = st.sidebar.slider(
    "Magnitude", float(df["mag"].min()), float(df["mag"].max()),
    (float(df["mag"].min()), float(df["mag"].max())), step=0.1,
)

if len(date_range) != 2:
    st.info("Select both a start and end date.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
f = df[(df["time"] >= start) & (df["time"] < end) & df["mag"].between(*mag_range)]

if f.empty:
    st.warning("No earthquakes match these filters.")
    st.stop()

# ---------- Summary stats ----------
strongest = f.loc[f["mag"].idxmax()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", f"{len(f):,}")
c2.metric("Average magnitude", f"{f['mag'].mean():.2f}")
c3.metric("Strongest", f"M{strongest['mag']:.1f}")
c4.metric("Strongest date", strongest["time"].strftime("%Y-%m-%d"))

tab_map, tab_trends, tab_regions, tab_data = st.tabs(["🗺️ Map", "📈 Trends", "📍 Regions", "📋 Data"])

# ---------- Map ----------
with tab_map:
    view = st.radio("Map view", ["Markers", "Density heatmap"], horizontal=True)
    m = folium.Map(location=[28.4, 84.1], zoom_start=7, tiles="CartoDB positron")
    if view == "Markers":
        shown = f.nlargest(2000, "mag")  # cap markers for performance
        for r in shown.itertuples():
            folium.CircleMarker(
                [r.latitude, r.longitude],
                radius=max(2, (r.mag - 2.5) * 2.5),
                color=mag_color(r.mag), fill=True, fill_opacity=0.6, weight=1,
                tooltip=f"M{r.mag:.1f} | {r.time:%Y-%m-%d} | depth {r.depth:.0f} km<br>{r.place}",
            ).add_to(m)
        if len(f) > len(shown):
            st.caption(f"Showing the {len(shown):,} strongest of {len(f):,} events. Narrow the filters to see more.")
    else:
        HeatMap(f[["latitude", "longitude"]].values.tolist(), radius=12, blur=15).add_to(m)
    st_folium(m, height=550, use_container_width=True, returned_objects=[])

# ---------- Trends ----------
with tab_trends:
    per_year = f.groupby("year").size().reset_index(name="events")
    st.plotly_chart(px.bar(per_year, x="year", y="events", title="Recorded events per year"),
                    use_container_width=True)
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.histogram(f, x="mag", nbins=30, title="Magnitude distribution"),
                       use_container_width=True)
    col_b.plotly_chart(px.scatter(f, x="time", y="mag", color="mag", opacity=0.6,
                                  title="Magnitude over time"), use_container_width=True)
    st.caption(
        "Note: more events in recent years often reflects better seismic monitoring, "
        "not necessarily more earthquakes."
    )

# ---------- Regions ----------
with tab_regions:
    g = f.copy()
    g["cell"] = g["latitude"].round().astype(int).astype(str) + "°N, " + g["longitude"].round().astype(int).astype(str) + "°E"
    cells = g.groupby("cell").agg(events=("mag", "size"), max_mag=("mag", "max")).reset_index()
    cells = cells.sort_values("events", ascending=False).head(15)
    st.plotly_chart(
        px.bar(cells, x="events", y="cell", orientation="h", hover_data=["max_mag"],
               title="Top 1°×1° grid cells by recorded events").update_yaxes(autorange="reversed"),
        use_container_width=True,
    )
    st.caption("Grid cells are approximate areas (about 100 km wide), centered on the labeled coordinates.")

# ---------- Data ----------
with tab_data:
    st.dataframe(f.drop(columns="year").sort_values("time", ascending=False), use_container_width=True)
    st.download_button("Download filtered CSV", f.to_csv(index=False), "nepal_earthquakes_filtered.csv", "text/csv")

st.divider()
st.caption(
    "Data: USGS Earthquake Catalog. The search area is a rectangle around Nepal, "
    "so it includes some events in neighbouring India and Tibet."
)