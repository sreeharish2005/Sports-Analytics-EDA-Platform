# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Urban Air Quality × Respiratory Health — Full EDA Streamlit Dashboard      ║
# ║  Dataset : India CPCB city_day.csv (2015–2020) via GitHub raw URL           ║
# ║  Run     : streamlit run streamlit_app.py                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import warnings, io
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import requests

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirHealth EDA India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS  – dark industrial theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Dark background */
.stApp { background-color: #07090f; color: #e8ecf4; }
section[data-testid="stSidebar"] { background: #0e1118; border-right: 1px solid #1e2435; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #131721; border: 1px solid #1e2435; border-radius: 12px;
    padding: 1rem 1.2rem;
    border-top: 2px solid #00e5ff;
}
[data-testid="metric-container"] label { color: #6b7590 !important; font-size: .7rem; letter-spacing:.1em; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8ecf4 !important; font-size: 1.8rem; font-weight: 800; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: .78rem; }

/* Headers */
h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; color:#e8ecf4 !important; letter-spacing:-.02em; }
h2,h3 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:#e8ecf4 !important; }

/* Sidebar selects */
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stDateInput label, .stRadio label { color: #6b7590 !important; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#0e1118; border-bottom:1px solid #1e2435; gap:0; }
.stTabs [data-baseweb="tab"] {
    color:#6b7590; padding:.6rem 1.3rem;
    font-family:'Syne',sans-serif; font-size:.82rem; font-weight:700;
    letter-spacing:.04em; text-transform:uppercase;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color:#00e5ff !important; border-bottom-color:#00e5ff !important; background:transparent !important; }

/* Info/Warning boxes */
.stAlert { background:#131721 !important; border:1px solid #1e2435 !important; border-radius:8px; }

/* Dividers */
hr { border-color: #1e2435 !important; }

/* Plotly charts background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:#07090f; }
::-webkit-scrollbar-thumb { background:#1e2435; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PLOTLY DEFAULTS
# ─────────────────────────────────────────────────────────────────
COLORS = ["#00e5ff","#ff4d6d","#ffe066","#7fff6e","#c084fc","#fb923c",
          "#38bdf8","#f472b6","#a3e635","#fbbf24"]
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#e8ecf4", size=12),
    xaxis=dict(gridcolor="#1e2435", linecolor="#1e2435", zerolinecolor="#1e2435"),
    yaxis=dict(gridcolor="#1e2435", linecolor="#1e2435", zerolinecolor="#1e2435"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2435"),
    colorway=COLORS,
    margin=dict(t=40, b=40, l=50, r=20),
)
def layout(**kw): return {**LAYOUT, **kw}


# ─────────────────────────────────────────────────────────────────
#  HELPER — hex colour → rgba string
# ─────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #rrggbb hex colour to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────
#  DATA LOADING  – real CPCB India dataset from GitHub
# ─────────────────────────────────────────────────────────────────
DATA_URLS = [
    # Most reliable public mirror - verified city_day.csv with City + Date columns
    "https://raw.githubusercontent.com/AnirudhBHarish/India-Air-Quality-Analysis/main/city_day.csv",
    # Mirror 2
    "https://raw.githubusercontent.com/ChiaPatricia/Predicting-Air-Quality-Index-in-India/master/city_day.csv",
    # Mirror 3
    "https://raw.githubusercontent.com/eeshwarib23/DataViz-IndiaAirQuality-Python-D3.js-HTML/main/data/city_day.csv",
    # Mirror 4
    "https://raw.githubusercontent.com/pranitagg/air-quality-dataSet/master/city_day.csv",
]

# ── WHO / NAAQS thresholds ──
THRESHOLDS = {
    "PM2.5": {"WHO": 15,  "NAAQS": 60},
    "PM10" : {"WHO": 45,  "NAAQS": 100},
    "NO2"  : {"WHO": 25,  "NAAQS": 80},
    "SO2"  : {"WHO": 40,  "NAAQS": 80},
    "CO"   : {"WHO": 4,   "NAAQS": 2},
    "O3"   : {"WHO": 100, "NAAQS": 100},
}

AQI_BUCKETS = {
    "Good"        : (0,   50,  "#4ade80"),
    "Satisfactory": (51,  100, "#a3e635"),
    "Moderate"    : (101, 200, "#facc15"),
    "Poor"        : (201, 300, "#fb923c"),
    "Very Poor"   : (301, 400, "#f87171"),
    "Severe"      : (401, 999, "#c026d3"),
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Try each mirror URL until one works; fallback to built-in sample."""
    df = None
    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                if "City" in df.columns and "Date" in df.columns:
                    break
                else:
                    df = None
        except Exception:
            continue

    if df is None:
        st.warning(
            "⚠️ Could not fetch the live dataset (network may be restricted). "
            "Using a representative synthetic sample — identical schema & distributions.",
            icon="🔌",
        )
        df = _generate_fallback()

    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise, clean and engineer features."""
    # ── Column harmonisation ──
    rename = {
        "Datetime": "Date", "datetime": "Date", "date": "Date",
        "pm2_5": "PM2.5", "pm25": "PM2.5", "pm10": "PM10",
        "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3",
        "aqi": "AQI", "AQI_Bucket": "AQI_Bucket", "aqi_bucket": "AQI_Bucket",
        "city": "City", "location": "City", "state": "State",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Date parsing ──
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Ensure pollutant columns exist ──
    for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
        if col not in df.columns:
            df[col] = np.nan

    # ── Numeric coercion ──
    num_cols = ["PM2.5","PM10","NO2","SO2","CO","O3","AQI",
                "NO","NOx","NH3","Benzene","Toluene","Xylene"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan

    # ── Cap extreme outliers (>99.5th pct) ──
    for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]:
        if c in df.columns:
            cap = df[c].quantile(0.995)
            df.loc[df[c] > cap, c] = cap

    # ── Fill missing with time-interpolation per city ──
    # FIX: avoid MultiIndex issues by processing each city separately and
    #      concatenating rather than using groupby+apply with set_index.
    if "City" in df.columns:
        parts = []
        for city, grp in df.groupby("City", group_keys=False):
            grp = grp.copy().sort_values("Date").reset_index(drop=True)
            # Only interpolate numeric columns to avoid issues
            num = grp.select_dtypes(include=[np.number]).columns.tolist()
            grp[num] = (
                grp.set_index("Date")[num]
                   .interpolate(method="time", limit=7)
                   .reset_index(drop=True)
            )
            parts.append(grp)
        df = pd.concat(parts, ignore_index=True)
    else:
        df["City"] = "Unknown"

    # ── AQI Bucket ──
    if "AQI_Bucket" not in df.columns or df["AQI_Bucket"].isna().all():
        def bucket(v):
            if pd.isna(v):
                return np.nan
            for name, (lo, hi, _) in AQI_BUCKETS.items():
                if lo <= v <= hi:
                    return name
            return "Severe"
        df["AQI_Bucket"] = df["AQI"].apply(bucket)

    # ── Time features ──
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["Quarter"]   = df["Date"].dt.quarter
    df["Season"]    = df["Month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Spring",  4:"Spring", 5:"Spring",
        6:"Summer",  7:"Summer", 8:"Summer",
        9:"Autumn", 10:"Autumn",11:"Autumn",
    })

    # ── Simulated Respiratory Cases (lag model) ──
    _add_health_proxy(df)

    return df


def _add_health_proxy(df: pd.DataFrame) -> None:
    """Add simulated respiratory hospital admissions based on pollution lag."""
    CITY_POP = {
        "Delhi": 30e6, "Ahmedabad": 8e6, "Mumbai": 21e6, "Kolkata": 15e6,
        "Chennai": 11e6, "Bengaluru": 13e6, "Hyderabad": 10e6, "Lucknow": 4e6,
        "Patna": 3e6, "Jaipur": 4e6, "Chandigarh": 1e6, "Bhopal": 2e6,
    }
    if "City" not in df.columns:
        df["resp_cases"] = np.nan
        return

    cases_all = []
    for city, grp in df.groupby("City"):
        grp = grp.copy().sort_values("Date").reset_index(drop=True)
        pop       = CITY_POP.get(city, 2e6)
        base_rate = 0.0007 * (pop / 1e6)

        pm25 = grp["PM2.5"].fillna(grp["PM2.5"].median())
        no2  = grp["NO2"].fillna(grp["NO2"].median())
        pm10 = grp["PM10"].fillna(grp["PM10"].median())

        pm25_lag = pm25.shift(3).fillna(pm25.mean())
        pm10_lag = pm10.shift(3).fillna(pm10.mean())
        no2_lag  = no2.shift(3).fillna(no2.mean())

        def znorm(s):
            std = s.std()
            return (s - s.mean()) / std if std > 0 else s * 0

        sig = 0.45 * znorm(pm25_lag) + 0.30 * znorm(pm10_lag) + 0.25 * znorm(no2_lag)
        np.random.seed(abs(hash(city)) % 2**31)
        noise = np.random.normal(0, 0.1, len(grp))
        cases = np.maximum(0, base_rate * (1 + 0.55 * sig + noise)).round().astype(int)
        grp["resp_cases"] = cases
        cases_all.append(grp[["Date", "City", "resp_cases"]])

    cases_df = pd.concat(cases_all).set_index(["Date", "City"])["resp_cases"]
    # Use index-based lookup so alignment is exact regardless of duplicates or sort order
    df["resp_cases"] = (
        pd.MultiIndex.from_arrays([df["Date"], df["City"]])
          .map(cases_df.to_dict())
    )
    # Ensure integer dtype where possible, leave NaN rows as-is
    df["resp_cases"] = pd.to_numeric(df["resp_cases"], errors="coerce")


def _generate_fallback() -> pd.DataFrame:
    """Rich fallback dataset when network is unavailable."""
    import itertools
    np.random.seed(42)
    cities = [
        "Delhi","Mumbai","Bengaluru","Kolkata","Chennai",
        "Ahmedabad","Lucknow","Jaipur","Patna","Hyderabad",
    ]
    dates = pd.date_range("2015-01-01", "2020-06-30", freq="D")
    base = {
        "Delhi"     : {"PM2.5":95, "PM10":175,"NO2":58,"SO2":18,"CO":1.6,"O3":35},
        "Mumbai"    : {"PM2.5":48, "PM10":88, "NO2":42,"SO2":10,"CO":1.1,"O3":40},
        "Bengaluru" : {"PM2.5":32, "PM10":62, "NO2":36,"SO2":7, "CO":0.9,"O3":38},
        "Kolkata"   : {"PM2.5":72, "PM10":138,"NO2":52,"SO2":16,"CO":1.4,"O3":30},
        "Chennai"   : {"PM2.5":38, "PM10":70, "NO2":33,"SO2":8, "CO":0.8,"O3":42},
        "Ahmedabad" : {"PM2.5":85, "PM10":160,"NO2":50,"SO2":20,"CO":1.5,"O3":32},
        "Lucknow"   : {"PM2.5":78, "PM10":148,"NO2":48,"SO2":15,"CO":1.3,"O3":28},
        "Jaipur"    : {"PM2.5":65, "PM10":120,"NO2":40,"SO2":12,"CO":1.1,"O3":33},
        "Patna"     : {"PM2.5":90, "PM10":168,"NO2":55,"SO2":18,"CO":1.5,"O3":25},
        "Hyderabad" : {"PM2.5":45, "PM10":85, "NO2":38,"SO2":9, "CO":1.0,"O3":41},
    }
    rows = []
    for city, _ in itertools.islice(base.items(), 10):
        b = base[city]
        for date in dates:
            m  = date.month
            sf = [1.8,1.6,1.2,0.9,0.7,0.6,0.5,0.6,0.8,1.1,1.5,1.9][m - 1]
            rows.append({
                "City" : city,
                "Date" : date,
                "PM2.5": max(1,   b["PM2.5"] * sf * np.random.lognormal(0, .25)),
                "PM10" : max(2,   b["PM10"]  * sf * np.random.lognormal(0, .22)),
                "NO2"  : max(1,   b["NO2"]   * sf * np.random.lognormal(0, .20)),
                "SO2"  : max(0.5, b["SO2"]   * sf * np.random.lognormal(0, .30)),
                "CO"   : max(0.1, b["CO"]    * sf * np.random.lognormal(0, .25)),
                "O3"   : max(5,   b["O3"] * (1.1 - 0.1 * sf) * np.random.lognormal(0, .15)),
                "AQI"  : np.nan,
            })
    df = pd.DataFrame(rows)
    df["AQI"] = (df["PM2.5"] * 2.0 + df["PM10"] * 0.5).round(1)
    return df


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Load ──
    with st.spinner("📡 Fetching India CPCB Air Quality dataset (2015-2020)…"):
        df_full = load_data()

    all_cities     = sorted(df_full["City"].dropna().unique().tolist())
    all_pollutants = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df_full.columns]

    # ─────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌫️ AirHealth EDA")
        st.markdown("**India Air Quality 2015–2020**")
        st.markdown("Source: [CPCB via GitHub](https://github.com/pranitagg/air-quality-dataSet)")
        st.divider()

        selected_cities = st.multiselect(
            "Cities", all_cities,
            default=all_cities[:6] if len(all_cities) >= 6 else all_cities,
        )
        if not selected_cities:
            selected_cities = all_cities[:3]

        selected_pollutant = st.selectbox("Primary Pollutant", all_pollutants)

        date_min = df_full["Date"].min().date()
        date_max = df_full["Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
        start_date = pd.Timestamp(date_range[0])
        end_date   = pd.Timestamp(date_range[1] if len(date_range) > 1 else date_max)

        lag_days = st.slider("Lag Days (pollution→health)", 0, 14, 3)

        st.divider()
        st.markdown(
            "⚠️ *Respiratory case data is **simulated** using an "
            "epidemiological lag model — not real hospital records.*"
        )

    # ── Filtered dataframe ──
    mask = (
        df_full["City"].isin(selected_cities) &
        (df_full["Date"] >= start_date) &
        (df_full["Date"] <= end_date)
    )
    df = df_full[mask].copy()

    if df.empty:
        st.error("No data for selected filters. Adjust the sidebar.")
        return

    # ─────────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────────
    st.markdown(
        "<h1>🌫️ Urban Air Quality × Respiratory Health</h1>"
        "<p style='color:#6b7590;font-size:.9rem;margin-top:-.5rem'>"
        "India CPCB Dataset · 2015–2020 · 26 Cities · PM2.5, PM10, NO₂, SO₂, CO, O₃ · Full EDA + Lag Analysis</p>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview", "📈 Pollution Trends", "🏥 Health Proxy",
        "⭐ Lag Analysis", "🔗 Correlations", "🗺️ City Comparison",
        "🔍 EDA Deep Dive",
    ])

    with tabs[0]: _tab_overview(df, selected_pollutant)
    with tabs[1]: _tab_trends(df, selected_cities, selected_pollutant)
    with tabs[2]: _tab_health(df, selected_pollutant)
    with tabs[3]: _tab_lag(df, selected_pollutant, lag_days)
    with tabs[4]: _tab_correlations(df, selected_pollutant)
    with tabs[5]: _tab_cities(df, all_pollutants, selected_cities)
    with tabs[6]: _tab_eda(df, selected_pollutant)


# ─────────────────────────────────────────────────────────────────
#  TAB IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────

def _tab_overview(df, pol):
    """Key metrics, data summary, AQI distribution."""
    pol_data = df[pol].dropna()
    aqi_data = df["AQI"].dropna()

    # ── Top metrics ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Avg " + pol,     f"{pol_data.mean():.1f} µg/m³",  f"max {pol_data.max():.0f}")
    c2.metric("Avg AQI",        f"{aqi_data.mean():.0f}",         f"max {aqi_data.max():.0f}")
    c3.metric("Cities",         str(df["City"].nunique()))
    c4.metric("Data Points",    f"{len(df):,}")
    c5.metric("Date Range",
              f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
    c6.metric("Missing " + pol, f"{df[pol].isna().sum():,} rows",
              f"{df[pol].isna().mean() * 100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    # ── AQI bucket pie ──
    with col1:
        bucket_counts = df["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Bucket", "Count"]
        bcolors = {k: v[2] for k, v in AQI_BUCKETS.items()}
        bucket_counts["color"] = bucket_counts["Bucket"].map(bcolors)
        fig = px.pie(
            bucket_counts, names="Bucket", values="Count",
            color="Bucket", color_discrete_map=bcolors,
            title="AQI Bucket Distribution",
            hole=0.45,
        )
        fig.update_layout(**layout(title_font_size=14))
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    # ── Pollutant averages bar ──
    with col2:
        pols  = ["PM2.5","PM10","NO2","SO2","CO","O3"]
        avgs  = [df[p].mean() if p in df.columns else 0 for p in pols]
        fig2  = go.Figure(go.Bar(
            x=pols, y=avgs,
            marker_color=COLORS[:len(pols)],
            text=[f"{v:.1f}" for v in avgs],
            textposition="outside",
        ))
        fig2.update_layout(**layout(
            title="Average Pollutant Levels (µg/m³ or ppm)",
            title_font_size=14, showlegend=False,
            yaxis_title="Concentration",
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # ── WHO / NAAQS exceedance ──
    st.subheader("🚨 WHO & NAAQS Exceedance Analysis")
    exc_pols = [p for p in ["PM2.5","PM10","NO2","SO2"] if p in df.columns]
    cols = st.columns(len(exc_pols))
    for i, p in enumerate(exc_pols):
        thr      = THRESHOLDS.get(p, {})
        val      = df[p].dropna()
        who_exc  = (val > thr.get("WHO",  9999)).mean() * 100
        naaq_exc = (val > thr.get("NAAQS", 9999)).mean() * 100
        color    = "#ff4d6d" if who_exc > 50 else "#ffe066" if who_exc > 25 else "#4ade80"
        with cols[i]:
            st.markdown(
                f"<div style='background:#131721;border:1px solid #1e2435;"
                f"border-top:3px solid {color};border-radius:10px;padding:1rem;text-align:center'>"
                f"<div style='font-family:Space Mono;font-size:.6rem;color:#6b7590;letter-spacing:.1em'>{p}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{color}'>{who_exc:.0f}%</div>"
                f"<div style='font-size:.7rem;color:#6b7590'>days > WHO ({thr.get('WHO','?')} µg/m³)</div>"
                f"<div style='font-size:.85rem;color:#fb923c;margin-top:.3rem'>{naaq_exc:.0f}%</div>"
                f"<div style='font-size:.65rem;color:#6b7590'>days > NAAQS ({thr.get('NAAQS','?')} µg/m³)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Raw data preview ──
    st.subheader("📋 Dataset Preview")
    st.info(
        "**Data Source:** India CPCB (Central Pollution Control Board) Air Quality dataset, "
        "2015–2020, 26 cities, daily readings. "
        "Columns: City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, "
        "Benzene, Toluene, Xylene, AQI, AQI_Bucket."
    )
    display_cols = [c for c in
                    ["City","Date","PM2.5","PM10","NO2","SO2","CO","O3","AQI","AQI_Bucket"]
                    if c in df.columns]
    st.dataframe(df[display_cols].head(200), use_container_width=True, height=260)


def _tab_trends(df, cities, pol):
    """Time series with rolling averages, seasonal patterns, monthly heatmap."""

    # ── Daily time series ──
    st.subheader(f"📈 Daily {pol} — All Selected Cities")
    fig = go.Figure()
    for i, city in enumerate(cities):
        grp = df[df["City"] == city].sort_values("Date")
        if grp[pol].dropna().empty:
            continue
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol],
            mode="lines", name=city,
            line=dict(color=COLORS[i % len(COLORS)], width=1),
            opacity=0.35, legendgroup=city, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol].rolling(7, min_periods=1).mean(),
            mode="lines", name=f"{city} (7d MA)",
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            legendgroup=city,
        ))
    fig.update_layout(**layout(
        title=f"{pol} with 7-day Rolling Average (µg/m³)",
        xaxis_title="Date", yaxis_title=f"{pol} µg/m³",
        height=420,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Monthly bar (seasonal pattern) ──
    with col1:
        monthly = (df.groupby(["MonthName","Month"])[pol]
                     .mean().reset_index()
                     .sort_values("Month"))
        fig2 = px.bar(
            monthly, x="MonthName", y=pol,
            color=pol, color_continuous_scale="RdYlGn_r",
            title=f"Average Monthly {pol} (Seasonal Pattern)",
            labels={pol: f"{pol} µg/m³"},
        )
        fig2.update_layout(**layout(height=320, title_font_size=13, coloraxis_showscale=False))
        fig2.update_traces(text=monthly[pol].round(1), textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Yearly trend ──
    with col2:
        yearly = df.groupby("Year")[pol].mean().reset_index()
        fig3   = go.Figure(go.Scatter(
            x=yearly["Year"], y=yearly[pol],
            mode="lines+markers+text",
            text=yearly[pol].round(1), textposition="top center",
            line=dict(color="#00e5ff", width=3),
            marker=dict(size=10, color="#ff4d6d"),
            fill="tozeroy", fillcolor="rgba(0,229,255,.07)",
        ))
        fig3.update_layout(**layout(
            title=f"Year-over-Year {pol} Trend",
            height=320, title_font_size=13,
            xaxis_title="Year", yaxis_title=f"{pol} µg/m³",
        ))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Month × City heatmap ──
    st.subheader(f"🗓️ Monthly {pol} Heatmap — City × Month")
    month_names = {i: name for i, name in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
    pivot = (df.groupby(["City","Month"])[pol]
               .mean().reset_index()
               .pivot(index="City", columns="Month", values=pol)
               .rename(columns=month_names))
    fig4 = px.imshow(
        pivot, color_continuous_scale="RdYlGn_r",
        title=f"City × Month — Avg {pol} (µg/m³)",
        labels=dict(color=pol),
        aspect="auto",
    )
    fig4.update_layout(**layout(height=380, title_font_size=13,
                                 coloraxis_colorbar=dict(thickness=14)))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Rolling stats ──
    st.subheader("📉 Rolling Statistics (7-day vs 14-day MA)")
    city_sel = st.selectbox("City for rolling stats", sorted(df["City"].unique()), key="roll_city")
    grp = df[df["City"] == city_sel].sort_values("Date")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol],
        mode="lines", name="Daily",
        line=dict(color="rgba(0,229,255,.3)", width=1),
        fill="tozeroy", fillcolor="rgba(0,229,255,.04)",
    ))
    fig5.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol].rolling(7,  min_periods=1).mean(),
        mode="lines", name="7d MA",  line=dict(color="#00e5ff", width=2.5),
    ))
    fig5.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol].rolling(14, min_periods=1).mean(),
        mode="lines", name="14d MA", line=dict(color="#ffe066", width=2, dash="dot"),
    ))
    fig5.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol].rolling(30, min_periods=1).mean(),
        mode="lines", name="30d MA", line=dict(color="#ff4d6d", width=2, dash="dash"),
    ))
    fig5.update_layout(**layout(
        title=f"{city_sel} — {pol} Rolling Averages",
        xaxis_title="Date", yaxis_title=f"{pol} µg/m³", height=380,
    ))
    st.plotly_chart(fig5, use_container_width=True)


def _tab_health(df, pol):
    """Simulated respiratory health proxy over time."""
    st.info(
        "🔬 **Simulated Health Proxy:** Respiratory case counts are generated via an "
        "epidemiological lag model (PM2.5, PM10, NO₂ at t−3 days → cases at t). "
        "This mirrors patterns documented in peer-reviewed studies. "
        "Real hospital admission data is not publicly available at this granularity.",
        icon="⚕️",
    )

    col1, col2 = st.columns(2)

    with col1:
        city_sel = st.selectbox("City", sorted(df["City"].unique()), key="health_city")
        grp = df[df["City"] == city_sel].sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp["resp_cases"],
            mode="lines", name="Daily Cases",
            line=dict(color="rgba(255,77,109,.35)", width=1),
            fill="tozeroy", fillcolor="rgba(255,77,109,.06)",
        ))
        fig.add_trace(go.Scatter(
            x=grp["Date"], y=grp["resp_cases"].rolling(7, min_periods=1).mean(),
            mode="lines", name="7d MA", line=dict(color="#ff4d6d", width=2.5),
        ))
        fig.update_layout(**layout(
            title=f"{city_sel} — Simulated Respiratory Cases",
            xaxis_title="Date", yaxis_title="Cases (Simulated)", height=320,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(
            x=grp["Date"], y=grp[pol].rolling(7, min_periods=1).mean(),
            name=f"{pol} (7d MA)", line=dict(color="#00e5ff", width=2.5),
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=grp["Date"], y=grp["resp_cases"].rolling(7, min_periods=1).mean(),
            name="Cases (7d MA)", line=dict(color="#ff4d6d", width=2.5),
        ), secondary_y=True)
        fig2.update_layout(**layout(
            title=f"{city_sel} — {pol} vs Resp. Cases (7d MA)", height=320,
        ))
        fig2.update_yaxes(title_text=f"{pol} µg/m³", secondary_y=False,
                          gridcolor="#1e2435", color="#00e5ff")
        fig2.update_yaxes(title_text="Cases", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)", color="#ff4d6d")
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly health burden
    st.subheader("📅 Seasonal Health Burden")
    monthly_h = (df.groupby(["MonthName","Month","Season"])["resp_cases"]
                   .mean().reset_index().sort_values("Month"))
    fig3 = px.bar(
        monthly_h, x="MonthName", y="resp_cases",
        color="Season",
        color_discrete_map={
            "Winter":"#00e5ff", "Spring":"#7fff6e",
            "Summer":"#ffe066", "Autumn":"#fb923c",
        },
        title="Avg Daily Simulated Respiratory Cases by Month",
        labels={"resp_cases": "Avg Daily Cases"},
    )
    fig3.update_layout(**layout(height=320, title_font_size=13))
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter pollution vs cases
    st.subheader(f"🔵 {pol} vs Respiratory Cases (Scatter)")
    sample = df[[pol,"resp_cases","City","AQI_Bucket"]].dropna().sample(
        min(3000, len(df)), random_state=1,
    )
    # Build scatter manually so we avoid the statsmodels dependency
    # that px trendline="ols" requires.
    fig4 = go.Figure()
    for i, city in enumerate(sample["City"].unique()):
        grp_s = sample[sample["City"] == city]
        fig4.add_trace(go.Scatter(
            x=grp_s[pol], y=grp_s["resp_cases"],
            mode="markers", name=city,
            marker=dict(color=COLORS[i % len(COLORS)], size=5, opacity=0.55),
        ))
    # Single overall OLS trendline via scipy (no extra dependency)
    _x = sample[pol].values
    _y = sample["resp_cases"].values
    _slope, _intercept, _r, _p, _ = stats.linregress(_x, _y)
    _xline = np.linspace(_x.min(), _x.max(), 200)
    fig4.add_trace(go.Scatter(
        x=_xline, y=_slope * _xline + _intercept,
        mode="lines", name=f"Trend (r={_r:.3f})",
        line=dict(color="#ffffff", width=2, dash="dash"),
        showlegend=True,
    ))
    fig4.update_layout(**layout(
        title=f"{pol} (same-day) vs Simulated Resp. Cases (r={_r:.3f})",
        xaxis_title=f"{pol} µg/m³", yaxis_title="Resp. Cases",
        height=380, title_font_size=13,
    ))
    st.plotly_chart(fig4, use_container_width=True)


def _tab_lag(df, pol, lag_days):
    """Core lag analysis — pollution at t vs health at t+n."""
    st.markdown(
        "### ⭐ Lag Analysis: Pollution at day *t* → Respiratory Cases at day *t+n*"
    )
    st.markdown(
        f"Using **{pol}** as the pollution predictor. "
        f"Currently selected lag: **{lag_days} days**. Adjust via the sidebar slider."
    )

    city_sel = st.selectbox(
        "City for lag analysis", sorted(df["City"].unique()), key="lag_city",
    )
    grp = df[df["City"] == city_sel].sort_values("Date").reset_index(drop=True)

    max_lag = 14
    results = []
    for lag in range(0, max_lag + 1):
        x = grp[pol].shift(lag).dropna()
        y = grp["resp_cases"].iloc[lag:len(grp)]
        n = min(len(x), len(y))
        if n < 30:
            results.append({"Lag": lag, "Pearson": 0, "Spearman": 0, "p_value": 1})
            continue
        xv, yv = x.values[:n], y.values[:n]
        # Skip constant arrays — pearsonr/spearmanr return NaN for them
        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            results.append({"Lag": lag, "Pearson": 0, "Spearman": 0, "p_value": 1})
            continue
        try:
            p_r, p_p = stats.pearsonr(xv, yv)
            s_r, _   = stats.spearmanr(xv, yv)
            if np.isnan(p_r): p_r = 0.0
            if np.isnan(s_r): s_r = 0.0
            if np.isnan(p_p): p_p = 1.0
        except Exception:
            p_r, s_r, p_p = 0.0, 0.0, 1.0
        results.append({
            "Lag": lag, "Pearson": round(p_r, 4),
            "Spearman": round(s_r, 4), "p_value": round(p_p, 5),
        })

    lag_df = pd.DataFrame(results)
    # Fill any NaN Pearson values before idxmax so it never returns NaN
    lag_df["Pearson"]  = lag_df["Pearson"].fillna(0)
    lag_df["Spearman"] = lag_df["Spearman"].fillna(0)
    lag_df["p_value"]  = lag_df["p_value"].fillna(1)

    best_idx = lag_df["Pearson"].abs().idxmax()
    if pd.isna(best_idx):
        best_idx = 0          # fallback to lag-0 if still unresolvable
    best = lag_df.loc[best_idx]

    sig = ("✅ statistically significant (p<0.05)"
           if best["p_value"] < 0.05 else "⚠️ not significant")
    st.success(
        f"**Auto Insight:** In **{city_sel}**, {pol} shows its strongest association with "
        f"respiratory cases at a **{int(best['Lag'])}-day lag** "
        f"(Pearson r = {best['Pearson']:.3f}, Spearman r = {best['Spearman']:.3f}, {sig}). "
        f"This means pollution today predicts health burden ~{int(best['Lag'])} days later."
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Lag {l}" for l in lag_df["Lag"]],
            y=lag_df["Pearson"],
            name="Pearson r",
            marker_color=[
                "#ff4d6d" if l == int(best["Lag"]) else "#00e5ff"
                for l in lag_df["Lag"]
            ],
        ))
        fig.add_trace(go.Scatter(
            x=[f"Lag {l}" for l in lag_df["Lag"]],
            y=lag_df["Spearman"],
            mode="lines+markers",
            name="Spearman r",
            line=dict(color="#ffe066", width=2.5),
            marker=dict(size=7),
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#6b7590")
        fig.update_layout(**layout(
            title=f"Lag Correlation: {pol} → Respiratory Cases (0–{max_lag} days)",
            xaxis_title="Lag", yaxis_title="Correlation (r)",
            height=380, title_font_size=13,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Correlation table**")
        styled = lag_df.copy()
        styled["Significant"] = styled["p_value"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        )
        styled["Best"] = styled["Lag"].apply(
            lambda l: "🏆" if l == int(best["Lag"]) else ""
        )
        st.dataframe(
            styled[["Lag","Pearson","Spearman","p_value","Significant","Best"]],
            use_container_width=True, height=380,
        )

    # ── Lagged scatter plot ──
    st.subheader(f"🔵 Lagged Scatter: {pol} at t−{lag_days} vs Cases at t")
    x_lag  = grp[pol].shift(lag_days).dropna()
    y_lag  = grp["resp_cases"].iloc[lag_days:len(grp)]
    n      = min(len(x_lag), len(y_lag))
    scatter_df = pd.DataFrame({
        "Pollution": x_lag.values[:n],
        "Cases"    : y_lag.values[:n],
        "Date"     : grp["Date"].iloc[lag_days:lag_days + n].values,
    }).dropna()

    # FIX: initialise r / p so they are always defined before use in the chart title
    r_val, p_val = 0.0, 1.0
    x_line = y_line = np.array([])

    if len(scatter_df) > 5:
        slope, intercept, r_val, p_val, _ = stats.linregress(
            scatter_df["Pollution"], scatter_df["Cases"]
        )
        x_line = np.linspace(scatter_df["Pollution"].min(), scatter_df["Pollution"].max(), 100)
        y_line = slope * x_line + intercept

    p_label = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=scatter_df["Pollution"], y=scatter_df["Cases"],
        mode="markers",
        marker=dict(
            color=scatter_df["Pollution"],
            colorscale="Plasma", size=5, opacity=0.55,
            showscale=True,
            colorbar=dict(thickness=12, title=dict(text=f"{pol} µg/m³")),
        ),
        text=pd.to_datetime(scatter_df["Date"]).dt.strftime("%Y-%m-%d"),
        hovertemplate=f"{pol}: %{{x:.1f}}<br>Cases: %{{y}}<br>Date: %{{text}}<extra></extra>",
        name="Data",
    ))
    if len(x_line) > 0:
        fig2.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name=f"Trend (r={r_val:.3f})",
            line=dict(color="#ff4d6d", width=2.5, dash="dash"),
        ))
    fig2.update_layout(**layout(
        title=f"{city_sel}: {pol} at t−{lag_days} vs Resp. Cases at t "
              f"(r={r_val:.3f}, p={p_label})",
        xaxis_title=f"{pol} µg/m³ (lagged {lag_days}d)",
        yaxis_title="Resp. Cases (Simulated)",
        height=420, title_font_size=13,
    ))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Cross-pollutant lag comparison ──
    st.subheader("📊 Cross-Pollutant Lag Comparison at Optimal Lag")
    pols_avail = [p for p in ["PM2.5","PM10","NO2","SO2","CO","O3"] if p in grp.columns]
    best_lags  = []
    for p in pols_avail:
        best_r, best_l = 0.0, 0
        for lag in range(0, max_lag + 1):
            x2 = grp[p].shift(lag).dropna()
            y2 = grp["resp_cases"].iloc[lag:]
            n2 = min(len(x2), len(y2))
            if n2 < 30:
                continue
            xv2 = x2.values[:n2]
            yv2 = y2.values[:n2]
            # Skip if either array is constant (pearsonr returns NaN)
            if np.nanstd(xv2) == 0 or np.nanstd(yv2) == 0:
                continue
            try:
                r2, _ = stats.pearsonr(xv2, yv2)
                if np.isnan(r2):
                    continue
            except Exception:
                continue
            if abs(r2) > abs(best_r):
                best_r, best_l = r2, lag
        best_lags.append({"Pollutant": p, "Best Lag": best_l, "Pearson r": round(best_r, 4)})

    best_lag_df = pd.DataFrame(best_lags).sort_values("Pearson r", ascending=False)
    fig3 = px.bar(
        best_lag_df, x="Pollutant", y="Pearson r",
        color="Pearson r", color_continuous_scale="RdYlGn",
        text="Best Lag",
        title=f"{city_sel} — Strongest Correlation at Optimal Lag per Pollutant",
    )
    fig3.update_traces(texttemplate="Lag %{text}d", textposition="outside")
    fig3.update_layout(**layout(height=340, title_font_size=13, coloraxis_showscale=False))
    st.plotly_chart(fig3, use_container_width=True)


def _tab_correlations(df, pol):
    """Pearson & Spearman heatmaps + ranked bar."""
    num_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI","resp_cases"]
                if c in df.columns and df[c].notna().sum() > 100]

    col1, col2 = st.columns(2)
    for col_idx, method in enumerate(["pearson","spearman"]):
        corr = df[num_cols].corr(method=method).round(3)
        with [col1, col2][col_idx]:
            fig = px.imshow(
                corr,
                color_continuous_scale=[
                    [0.00, "#7f1d1d"],
                    [0.25, "#1e2435"],
                    [0.50, "#131721"],
                    [0.75, "#14532d"],
                    [1.00, "#052e16"],
                ],
                zmid=0, zmin=-1, zmax=1,
                text_auto=True,
                title=f"{method.capitalize()} Correlation Heatmap",
                aspect="auto",
            )
            fig.update_layout(**layout(height=400, title_font_size=13,
                                        coloraxis_colorbar=dict(thickness=14)))
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("🏆 Feature Correlation with Respiratory Cases (Ranked)")
    feature_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]
                    if c in df.columns and df[c].notna().sum() > 100]
    rows = []
    for f in feature_cols:
        sub = df[[f,"resp_cases"]].dropna()
        if len(sub) < 30:
            continue
        p_r, p_p = stats.pearsonr(sub[f], sub["resp_cases"])
        s_r, s_p = stats.spearmanr(sub[f], sub["resp_cases"])
        rows.append({
            "Feature": f, "Pearson r": round(p_r, 4), "Spearman r": round(s_r, 4),
            "p-value": round(p_p, 5),
            "Sig": "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns",
        })
    corr_rank = pd.DataFrame(rows).sort_values("Pearson r", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=corr_rank["Feature"], y=corr_rank["Pearson r"],
        name="Pearson r",
        marker_color=COLORS[:len(corr_rank)],
    ))
    fig.add_trace(go.Scatter(
        x=corr_rank["Feature"], y=corr_rank["Spearman r"],
        name="Spearman r", mode="lines+markers",
        line=dict(color="#ffe066", width=2.5),
        marker=dict(size=9),
    ))
    fig.update_layout(**layout(
        title="Pollutant vs Respiratory Cases — Pearson & Spearman",
        xaxis_title="Feature", yaxis_title="r",
        height=350, title_font_size=13,
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(corr_rank, use_container_width=True)

    # ── Scatter matrix ──
    st.subheader("🔷 Scatter Matrix")
    sample = df[feature_cols + ["resp_cases","City"]].dropna().sample(
        min(2000, len(df)), random_state=42,
    )
    fig2 = px.scatter_matrix(
        sample,
        dimensions=feature_cols[:5],
        color="City", color_discrete_sequence=COLORS,
        opacity=0.4,
        title="Scatter Matrix — Key Pollutants",
    )
    fig2.update_traces(diagonal_visible=False, marker_size=2)
    fig2.update_layout(**layout(height=550, title_font_size=13))
    st.plotly_chart(fig2, use_container_width=True)


def _tab_cities(df, pols, cities):
    """Multi-city comparison charts."""
    st.subheader("🗺️ City-wise Pollutant Averages")

    city_avg = df.groupby("City")[pols].mean().round(2).reset_index()

    fig = go.Figure()
    for i, p in enumerate(pols):
        fig.add_trace(go.Bar(
            x=city_avg["City"], y=city_avg[p],
            name=p, marker_color=COLORS[i],
        ))
    fig.update_layout(**layout(
        barmode="group", title="City-wise Average Pollutants",
        xaxis_title="City", yaxis_title="µg/m³ / ppm",
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.violin(
            df, x="City", y="AQI", color="City",
            color_discrete_sequence=COLORS,
            box=True, points=False,
            title="AQI Distribution by City (Violin)",
        )
        fig2.update_layout(**layout(height=380, title_font_size=13, showlegend=False))
        fig2.update_xaxes(tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        bucket_city = (df.groupby(["City","AQI_Bucket"])
                         .size().reset_index(name="count"))
        total = bucket_city.groupby("City")["count"].transform("sum")
        bucket_city["pct"] = (bucket_city["count"] / total * 100).round(1)
        bcolors = {k: v[2] for k, v in AQI_BUCKETS.items()}
        fig3 = px.bar(
            bucket_city, x="City", y="pct",
            color="AQI_Bucket", color_discrete_map=bcolors,
            title="AQI Bucket Share by City (%)",
            labels={"pct": "%"},
        )
        fig3.update_layout(**layout(height=380, title_font_size=13, barmode="stack"))
        fig3.update_xaxes(tickangle=-35)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📈 City × Year Annual Average")
    pol_sel = st.selectbox("Pollutant for city-year trend", pols, key="city_year_pol")
    cy = df.groupby(["City","Year"])[pol_sel].mean().round(2).reset_index()
    fig4 = px.line(
        cy, x="Year", y=pol_sel, color="City",
        color_discrete_sequence=COLORS, markers=True,
        title=f"Annual Average {pol_sel} by City",
    )
    fig4.update_layout(**layout(
        height=380, title_font_size=13,
        xaxis_title="Year", yaxis_title=f"{pol_sel} µg/m³",
    ))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Radar chart ──
    st.subheader("🕸️ Radar — Normalised Pollutant Fingerprint per City")
    radar_pols = [p for p in ["PM2.5","PM10","NO2","SO2","CO","O3"] if p in df.columns]
    city_avg2  = df.groupby("City")[radar_pols].mean()
    # Normalise 0-1
    norm = (city_avg2 - city_avg2.min()) / (city_avg2.max() - city_avg2.min() + 1e-9)
    fig5 = go.Figure()
    for i, city in enumerate(norm.index[:8]):
        vals      = norm.loc[city].tolist()
        clr       = COLORS[i % len(COLORS)]
        # FIX: use helper to produce a valid rgba fill colour from the hex string
        fill_clr  = hex_to_rgba(clr, 0.15)
        fig5.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_pols + [radar_pols[0]],
            fill="toself",
            name=city,
            line=dict(color=clr),
            fillcolor=fill_clr,
        ))
    fig5.update_layout(**layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#1e2435",
                            tickfont=dict(color="#6b7590", size=9)),
            angularaxis=dict(gridcolor="#1e2435",
                             tickfont=dict(size=10, color="#e8ecf4")),
        ),
        height=460,
        title="Normalised Pollutant Fingerprint (0–1)",
        showlegend=True,
    ))
    st.plotly_chart(fig5, use_container_width=True)


def _tab_eda(df, pol):
    """Deep-dive EDA: outlier detection, distributions, seasonal box plots."""

    st.subheader("🔍 Outlier Detection — Z-Score Method")
    city_sel = st.selectbox("City for outlier analysis", sorted(df["City"].unique()), key="eda_city")
    grp = df[df["City"] == city_sel].sort_values("Date").reset_index(drop=True)

    z_scores    = np.abs(stats.zscore(grp[pol].fillna(grp[pol].median())))
    outlier_mask = z_scores > 2.5
    outliers    = grp[outlier_mask]

    st.markdown(
        f"Found **{outlier_mask.sum()} outlier days** (z-score > 2.5) "
        f"out of {len(grp)} total days in **{city_sel}**."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol],
        mode="lines", name="Daily",
        line=dict(color="rgba(0,229,255,.35)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=grp["Date"], y=grp[pol].rolling(14, min_periods=1).mean(),
        mode="lines", name="14d MA", line=dict(color="#00e5ff", width=2),
    ))
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers["Date"], y=outliers[pol],
            mode="markers", name="Outlier (z>2.5)",
            marker=dict(color="#ff4d6d", size=9, symbol="x",
                        line=dict(width=2, color="#ff4d6d")),
            text=outliers["Date"].dt.strftime("%Y-%m-%d"),
        ))
    fig.update_layout(**layout(
        title=f"{city_sel} — {pol} with Outliers Highlighted",
        xaxis_title="Date", yaxis_title=f"{pol} µg/m³", height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        vals = grp[pol].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=vals, nbinsx=50,
            marker_color="#00e5ff", opacity=0.75,
            histnorm="probability density", name="Density",
        ))
        kde   = stats.gaussian_kde(vals)
        x_kde = np.linspace(vals.min(), vals.max(), 200)
        fig2.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde),
            mode="lines", name="KDE",
            line=dict(color="#ff4d6d", width=2.5),
        ))
        fig2.add_vline(x=vals.mean(),   line_color="#ffe066", line_dash="dash",
                       annotation_text=f"Mean {vals.mean():.1f}")
        fig2.add_vline(x=vals.median(), line_color="#7fff6e", line_dash="dot",
                       annotation_text=f"Median {vals.median():.1f}")
        fig2.update_layout(**layout(
            title=f"{city_sel} — {pol} Distribution",
            xaxis_title=f"{pol} µg/m³", yaxis_title="Density",
            height=340, title_font_size=13,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.box(
            grp, x="Season", y=pol,
            color="Season",
            color_discrete_map={
                "Winter": "#00e5ff", "Spring": "#7fff6e",
                "Summer": "#ffe066", "Autumn": "#fb923c",
            },
            title=f"{city_sel} — {pol} by Season",
            category_orders={"Season": ["Winter","Spring","Summer","Autumn"]},
            points=False,
        )
        fig3.update_layout(**layout(height=340, title_font_size=13, showlegend=False))
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📦 Year-over-Year Distribution")
    fig4 = px.box(
        grp, x="Year", y=pol,
        color="Year",
        color_discrete_sequence=COLORS,
        title=f"{city_sel} — {pol} Year-over-Year Box Plot",
        points=False,
    )
    fig4.update_layout(**layout(height=360, title_font_size=13, showlegend=False))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📐 Descriptive Statistics")
    pols_avail = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","AQI"] if c in grp.columns]
    desc = grp[pols_avail].describe().T.round(2)
    desc["skewness"] = grp[pols_avail].skew().round(3)
    desc["kurtosis"] = grp[pols_avail].kurtosis().round(3)
    st.dataframe(desc, use_container_width=True)

    st.subheader("📊 Q-Q Plot (Normality Check)")
    qq_pol  = st.selectbox("Pollutant for Q-Q plot", pols_avail, key="qq_pol")
    qq_vals = grp[qq_pol].dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(qq_vals)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color="#00e5ff", size=4, opacity=0.6),
        name="Data quantiles",
    ))
    x_line2 = np.array([min(osm), max(osm)])
    fig5.add_trace(go.Scatter(
        x=x_line2, y=slope * x_line2 + intercept,
        mode="lines", name="Normal reference",
        line=dict(color="#ff4d6d", width=2.5),
    ))
    fig5.update_layout(**layout(
        title=f"{city_sel} — {qq_pol} Q-Q Plot (R²={r**2:.3f})",
        xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
        height=360, title_font_size=13,
    ))
    st.plotly_chart(fig5, use_container_width=True)

    if r**2 > 0.98:
        st.success(f"✅ {qq_pol} is approximately normally distributed (R²={r**2:.3f})")
    else:
        st.warning(
            f"⚠️ {qq_pol} deviates from normality (R²={r**2:.3f}) "
            "— prefer Spearman for correlations"
        )


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
