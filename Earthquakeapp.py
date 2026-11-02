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
ZONE_LABELS = ["80.0-81.5°E (far west)", "81.5-83.0°E", "83.0-84.5°E",
               "84.5-86.0°E (Kathmandu area)", "86.0-87.5°E (Everest area)", "87.5-88.3°E (far east)"]


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Downloading earthquake data from USGS (first load takes about a minute)...")
def load_data() -> pd.DataFrame:
    frames = []
    this_year = pd.Timestamp.now().year + 1
    for y in range(START_YEAR, this_year, 5):
        params = {"format": "csv", "starttime": f"{y}-01-01", "endtime": f"{min(y + 5, this_year)}-01-01",
                  "minmagnitude": MIN_MAG, "orderby": "time-asc", **BBOX}
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
    return df.reset_index(drop=True)


def inside_outline(lons, lats, poly=NEPAL_OUTLINE):
    """Ray-casting point-in-polygon test for arrays of points."""
    xs = np.array([p[0] for p in poly])
    ys = np.array([p[1] for p in poly])
    inside = np.zeros(len(lons), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        crosses = (ys[i] > lats) != (ys[j] > lats)
        x_at = (xs[j] - xs[i]) * (lats - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]
        inside ^= crosses & (lons < x_at)
        j = i
    return inside


def mag_color(m: float) -> str:
    if m >= 6:
        return "#b71c1c"
    if m >= 5:
        return "#f57c00"
    if m >= 4:
        return "#fbc02d"
    return "#7cb342"


def tidy(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10))
    return fig


st.title("🌏 Nepal Earthquake Data Explorer")
st.caption("Historical earthquake records from the USGS catalog. This dashboard describes past data only. "
           "It does not predict future earthquakes.")
st.info("**Area covered:** a rectangle from 26.3°N to 30.5°N and 80.0°E to 88.3°E around Nepal. "
        "It also includes parts of India and Tibet. Details are in the About tab.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not download data from USGS. Please refresh the page in a minute. ({e})")
    st.stop()

# ---------- Filters (on the main page so they are easy to find on a phone) ----------
with st.expander("🔎 Filters", expanded=True):
    min_d, max_d = df["time"].min().date(), df["time"].max().date()
    date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    mag_lo, mag_hi = float(df["mag"].min()), float(df["mag"].max())
    mag_range = st.slider("Magnitude", mag_lo, mag_hi, (mag_lo, mag_hi), step=0.1)
    only_nepal = st.checkbox("Only events inside the approximate outline of Nepal", value=False,
                             help="Uses a rough, simplified outline, not the official border.")

if len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start = pd.Timestamp(date_range[0])
end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
mask = (df["time"] >= start) & (df["time"] < end) & df["mag"].between(*mag_range)
if only_nepal:
    mask &= inside_outline(df["longitude"].to_numpy(), df["latitude"].to_numpy())
f = df[mask].copy()

area_text = "inside the approximate Nepal outline" if only_nepal else "inside the rectangle around Nepal"
st.caption(f"Selection: {date_range[0]:%Y-%m-%d} to {date_range[1]:%Y-%m-%d}, magnitude "
           f"{mag_range[0]:.1f} to {mag_range[1]:.1f}, {area_text}. "
           f"{len(f):,} of {len(df):,} events in the full dataset.")

if f.empty:
    st.warning("No earthquakes match these filters. Try a wider date range or magnitude range.")
    st.stop()

# ---------- Summary stats (all calculated from the selection above) ----------
strongest = f.loc[f["mag"].idxmax()]
a, b = st.columns(2)
a.metric("Events in selection", f"{len(f):,}")
b.metric("Average magnitude", f"{f['mag'].mean():.2f}")
c, d = st.columns(2)
c.metric("Strongest magnitude", f"M{strongest['mag']:.1f}")
d.metric("Strongest event date", strongest["time"].strftime("%Y-%m-%d"))
st.caption(f"Strongest event location: {strongest['place']}")

tab_map, tab_trends, tab_regions, tab_data, tab_about = st.tabs(["🗺️ Map", "📈 Trends", "📍 Regions", "📋 Data", "ℹ️ About"])

# ---------- Map ----------
with tab_map:
    view = st.radio("Map view", ["Markers", "Density heatmap"], horizontal=True)
    m = folium.Map(location=[28.4, 84.1], zoom_start=7, tiles="OpenStreetMap")
    folium.Rectangle(bounds=[[BBOX["minlatitude"], BBOX["minlongitude"]], [BBOX["maxlatitude"], BBOX["maxlongitude"]]],
                     color="#555555", weight=1, dash_array="6", fill=False, tooltip="Area covered by the data").add_to(m)
    if only_nepal:
        folium.Polygon([(lat, lon) for lon, lat in NEPAL_OUTLINE], color="#1565c0", weight=2, fill=False,
                       tooltip="Approximate outline of Nepal (simplified)").add_to(m)
    if view == "Markers":
        shown = f.nlargest(2000, "mag")
        for r in shown.itertuples():
            folium.CircleMarker(
                [r.latitude, r.longitude], radius=max(2, (r.mag - 2.5) * 2.5),
                color=mag_color(r.mag), fill=True, fill_opacity=0.6, weight=1,
                tooltip=f"M{r.mag:.1f} | {r.time:%Y-%m-%d} | depth {r.depth:.0f} km<br>{r.place}",
            ).add_to(m)
        if len(f) > len(shown):
            st.caption(f"Showing the {len(shown):,} strongest of {len(f):,} events. Narrow the filters to see more.")
    else:
        HeatMap(f[["latitude", "longitude"]].values.tolist(), radius=12, blur=15).add_to(m)
    st_folium(m, height=420, use_container_width=True, returned_objects=[])
    st.markdown(
        "**How to read this map:** each dot is one recorded earthquake. Bigger dots are stronger. "
        "Colors: 🟢 under M4, 🟡 M4 to 4.9, 🟠 M5 to 5.9, 🔴 M6 and above. Tap a dot for its date, depth and place. "
        "The dashed rectangle is the area covered by the data. The heatmap shows where many events are close together."
    )

# ---------- Trends ----------
with tab_trends:
    per_year = f.groupby("year").size().reset_index(name="events")
    peak = per_year.loc[per_year["events"].idxmax()]
    st.plotly_chart(tidy(px.bar(per_year, x="year", y="events", title="Recorded events per year")),
                    use_container_width=True)
    st.markdown(
        f"**What it shows:** how many earthquakes were recorded in each year of your selection. "
        f"Taller bars mean more recorded events. In this selection the busiest year is **{int(peak['year'])}** "
        f"with **{int(peak['events'])}** events. A year with a very large earthquake often has a tall bar because of "
        f"its aftershocks. A rise over the decades partly reflects better instruments, not necessarily more earthquakes."
    )

    st.plotly_chart(tidy(px.histogram(f, x="mag", nbins=30, title="How strong the earthquakes were")),
                    use_container_width=True)
    st.markdown(
        f"**What it shows:** how many earthquakes fall at each magnitude. Small earthquakes are far more common than "
        f"large ones, so the bars shrink to the right. The middle (median) magnitude here is **M{f['mag'].median():.1f}**, "
        f"and **{(f['mag'] >= 5).mean() * 100:.1f}%** of events are M5 or higher. The data starts at M3.5, so the left side "
        f"is cut off on purpose."
    )

    st.plotly_chart(tidy(px.scatter(f, x="time", y="mag", color="mag", opacity=0.6, title="Magnitude over time")),
                    use_container_width=True)
    st.markdown(
        "**What it shows:** each dot is one earthquake, with time across and magnitude up. Dots near the top are the "
        "strongest events. A burst of dots right after a high dot is usually its aftershocks. Fewer small dots in the "
        "early years mostly means weaker earthquakes were not detected back then."
    )

# ---------- Regions ----------
with tab_regions:
    st.markdown(
        "**Important:** the USGS data has coordinates but no province or district names, so official province "
        "numbers can't be calculated from it alone. Instead, events are grouped into six **west-to-east bands of "
        "longitude**, which roughly follow Nepal's shape. These bands are **not** official provinces."
    )
    g = f.copy()
    g["zone"] = pd.cut(g["longitude"], bins=ZONE_BINS, labels=ZONE_LABELS, include_lowest=True)
    zt = g.groupby("zone", observed=False).agg(
        events=("mag", "size"), strongest=("mag", "max"), average=("mag", "mean")).reset_index()
    zt["share_%"] = (zt["events"] / zt["events"].sum() * 100).round(1)
    zt["strongest"] = zt["strongest"].round(1)
    zt["average"] = zt["average"].round(2)

    fig = px.bar(zt, x="events", y="zone", orientation="h", text="events", title="Events by west-to-east band")
    fig.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(tidy(fig, 340), use_container_width=True)
    top = zt.loc[zt["events"].idxmax()]
    st.markdown(
        f"**What it shows:** how many recorded events fall in each band, from far west (top) to far east (bottom). "
        f"The busiest band in this selection is **{top['zone']}** with **{int(top['events'])}** events "
        f"({top['share_%']}% of the selection). More recorded events does not mean more dangerous: "
        f"one very large earthquake and its aftershocks can fill a single band."
    )
    st.dataframe(zt.rename(columns={"zone": "Band", "events": "Events", "strongest": "Strongest M",
                                    "average": "Average M", "share_%": "Share %"}),
                 hide_index=True, use_container_width=True)

    g["cell"] = g["latitude"].round().astype(int).astype(str) + "°N, " + g["longitude"].round().astype(int).astype(str) + "°E"
    cells = g.groupby("cell").agg(events=("mag", "size"), max_mag=("mag", "max")).reset_index()
    cells = cells.sort_values("events", ascending=False).head(10)
    fig2 = px.bar(cells, x="events", y="cell", orientation="h", hover_data=["max_mag"],
                  title="Top 10 busiest areas (1°×1° squares)")
    fig2.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(tidy(fig2, 360), use_container_width=True)
    st.markdown(
        "**What it shows:** the map is cut into squares about 100 km wide, and these are the 10 squares with the most "
        "recorded events. Each label is the center of the square. Squares near the border may include areas of India or Tibet."
    )

# ---------- Data ----------
with tab_data:
    st.caption("This table shows only the events in your current selection.")
    st.dataframe(f.drop(columns="year").sort_values("time", ascending=False), use_container_width=True)
    st.download_button("Download selection as CSV", f.drop(columns="year").to_csv(index=False),
                       "nepal_earthquakes_selection.csv", "text/csv")

# ---------- About ----------
with tab_about:
    st.markdown(
        f"""
### Where the data comes from
The earthquake records come from the **USGS Earthquake Catalog** (earthquake.usgs.gov), through its public event web
service. The app downloads them automatically and refreshes them once a day.

**What was requested**
- Earthquakes of magnitude **{MIN_MAG} or higher**
- From **1 January {START_YEAR}** until today
- Inside a rectangle: **{BBOX['minlatitude']}°N to {BBOX['maxlatitude']}°N** and **{BBOX['minlongitude']}°E to {BBOX['maxlongitude']}°E**
- Downloaded in 5-year blocks, because the service limits how many events one request can return

**Full dataset after cleaning:** {len(df):,} events, from {df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}.

### How the data was cleaned
1. The 5-year blocks were joined into one table.
2. Duplicate events (same USGS event ID) were removed.
3. Rows missing a magnitude, latitude or longitude were dropped.
4. Times were converted to UTC (world time).
5. Only the columns needed were kept: time, latitude, longitude, depth, magnitude, magnitude type and place name.
6. A year column was added for the charts.

No other events were removed or changed. Outliers were not removed.

### How the numbers are calculated
Every number and chart on this page uses only the events left after your filters (date range, magnitude, and the
optional outline). The count, average and strongest event are all calculated from that selection.

### Geographic boundaries
- The data covers a **rectangle**, not Nepal's exact border, so it includes parts of India and Tibet.
- The optional "inside the approximate outline" filter uses a **rough, simplified outline** I drew from approximate
  coordinates. It is not an official border, so events right at the edge may be wrongly included or left out.
- The Regions tab uses west-to-east longitude bands, not official provinces or districts.

### Limits to keep in mind
- Small earthquakes were detected less well in earlier decades, so old and new counts are not directly comparable.
- Magnitudes come from different methods, and can differ from numbers published by Nepal's own seismology center.
- Locations and depths have some error, especially for older events.
- This is **historical data only**. It cannot tell when or where the next earthquake will happen.
"""
    )