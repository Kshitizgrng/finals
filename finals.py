import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from uuid import uuid4

try:
    from scipy import stats as sps
except Exception:
    sps = None

st.set_page_config(page_title="Week 10 • Full Team Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 0.85rem !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }
div[data-testid="column"] { padding-left: 0.35rem; padding-right: 0.35rem; }
[data-testid="metric-container"] { padding: 0.7rem 0.85rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.10rem !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
    line-height: 1.15 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
.js-plotly-plot .plot-container { width: 100% !important; }
[data-testid="stDataFrame"] { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.title("Lucentara & Dinosty Fossils • Week 10 Deep Dive Dashboard")
st.caption("Price Drivers • Product Mix • Customer Segments • Geography & Channels • Inventory Timing • Ownership • Seasonality • Compliance • Stats • $ CAD")

BASE = pathlib.Path(__file__).parent
DATA_FILE = BASE / "Combined_Sales_2025 (2).csv"

ESSENTIAL = [
    "Sale ID", "Date", "Country", "City", "Channel",
    "Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Shipped Date"
]

@st.cache_data(show_spinner=False)
def load_csv(p: pathlib.Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(p)
    except Exception:
        d = pd.read_csv(p, encoding="utf-8-sig")
    d.columns = d.columns.str.strip()
    return d

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

def normalize_country(x: str) -> str:
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    patches = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.": "United States",
        "us": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    return patches.get(s.lower(), s)

def cad(x, decimals=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    try:
        return f"${float(x):,.{decimals}f} CAD"
    except Exception:
        return "-"

def p_fmt(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "-"
    p = float(p)
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"

def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out

def fig_tight(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def download_html(fig, filename: str):
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button(
        label="Download HTML",
        data=html,
        file_name=filename,
        mime="text/html",
        key=f"dl_{uuid4().hex}"
    )

def heatmap_from_pivot(pv: pd.DataFrame, title: str, ztitle: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=pv.values,
            x=pv.columns.tolist(),
            y=pv.index.tolist(),
            colorbar=dict(title=ztitle)
        )
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        c2 = str(cand).lower()
        if c2 in lower_map:
            return lower_map[c2]
    for cand in candidates:
        pat = str(cand).lower()
        for c in cols:
            if pat in c.lower():
                return c
    return None

if not DATA_FILE.exists():
    st.error("Dataset file not found. Put 'Combined_Sales_2025 (2).csv' in the SAME folder as app.py in your repo.")
    st.stop()

df = load_csv(DATA_FILE)
missing = [c for c in ESSENTIAL if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

text_cols = [
    "Country", "City", "Channel", "Customer Type", "Product Type", "Lead Source",
    "Consignment? (Y/N)", "Species", "Grade", "Finish",
    "COA", "COA #", "COA Provided", "COA Provided? (Y/N)",
    "Export", "Export? (Y/N)", "Export Permit", "Export Permit (PDF link)"
]
for c in text_cols:
    if c in df.columns:
        df[c] = _clean_str(df[c])

df["Country"] = df["Country"].apply(normalize_country)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Shipped Date"] = pd.to_datetime(df["Shipped Date"], errors="coerce")

num_cols = ["Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)",
            "Color Count (#)", "length", "width", "weight"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["Net Sales (CAD)"] = (df["Price (CAD)"] - df["Discount (CAD)"]).clip(lower=0)
df["Total Collected (CAD)"] = (
    df["Net Sales (CAD)"]
    + df["Shipping (CAD)"].fillna(0)
    + df["Taxes Collected (CAD)"].fillna(0)
).clip(lower=0)

df["Discount Rate"] = np.where(df["Price (CAD)"] > 0, df["Discount (CAD)"] / df["Price (CAD)"], np.nan)
df["Ship Lag Raw (days)"] = (df["Shipped Date"] - df["Date"]).dt.days
df["Ship Lag Clean (days)"] = np.where(df["Ship Lag Raw (days)"] >= 0, df["Ship Lag Raw (days)"], np.nan)
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
df["MonthNum"] = df["Date"].dt.month

st.sidebar.header("Filters")

min_d = df["Date"].min()
max_d = df["Date"].max()
if pd.isna(min_d) or pd.isna(max_d):
    st.error("Date column could not be parsed.")
    st.stop()

dr = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()), key="date_range")
if not isinstance(dr, tuple):
    dr = (dr, dr)
start = pd.to_datetime(dr[0])
end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

metric = st.sidebar.selectbox(
    "Metric ($ CAD)",
    ["Total Collected (CAD)", "Net Sales (CAD)", "Price (CAD)"],
    index=0,
    key="metric_pick"
)

exclude_negative_lag = st.sidebar.toggle("Exclude negative ship lag", value=True, key="toggle_excl_neg_lag")
top_n = st.sidebar.slider("Top N (countries)", 5, 30, 12, key="top_n")

countries = sorted([c for c in df["Country"].dropna().unique().tolist() if c])
channels = sorted([c for c in df["Channel"].dropna().unique().tolist() if c])

sel_countries = st.sidebar.multiselect("Countries", countries, default=[], key="sel_countries")
sel_channels = st.sidebar.multiselect("Channels", channels, default=[], key="sel_channels")

if "Product Type" in df.columns:
    prod_vals = sorted([x for x in df["Product Type"].dropna().unique().tolist() if str(x).strip()])
    sel_products = st.sidebar.multiselect("Product Types (optional)", prod_vals, default=[], key="sel_products")
else:
    sel_products = []

if "Customer Type" in df.columns:
    cust_vals = sorted([x for x in df["Customer Type"].dropna().unique().tolist() if str(x).strip()])
    sel_customers = st.sidebar.multiselect("Customer Types (optional)", cust_vals, default=[], key="sel_customers")
else:
    sel_customers = []

base = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
if sel_countries:
    base = base[base["Country"].isin(sel_countries)]
if sel_channels:
    base = base[base["Channel"].isin(sel_channels)]
if sel_products and "Product Type" in base.columns:
    base = base[base["Product Type"].isin(sel_products)]
if sel_customers and "Customer Type" in base.columns:
    base = base[base["Customer Type"].isin(sel_customers)]

cities = sorted([c for c in base["City"].dropna().unique().tolist() if c])
sel_cities = st.sidebar.multiselect("Cities (optional)", cities, default=[], key="sel_cities")

f = base.copy()
if sel_cities:
    f = f[f["City"].isin(sel_cities)]

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

lag_col = "Ship Lag Clean (days)" if exclude_negative_lag else "Ship Lag Raw (days)"

orders = int(len(f))
total = float(f[metric].sum())
aov = float(f[metric].mean())
median_val = float(f[metric].median())

country_totals = f.groupby("Country")[metric].sum().sort_values(ascending=False)
channel_totals = f.groupby("Channel")[metric].sum().sort_values(ascending=False)

top_country = country_totals.index[0] if len(country_totals) else "-"
top_channel = channel_totals.index[0] if len(channel_totals) else "-"

cons_rate = float((f["Consignment? (Y/N)"].astype(str).str.upper().eq("Y").mean()) * 100) if "Consignment? (Y/N)" in f.columns else np.nan
avg_lag = float(np.nanmean(f[lag_col].values)) if f[lag_col].notna().any() else np.nan

r1 = st.columns(4)
r1[0].metric("Orders", f"{orders:,}")
r1[1].metric("Total", cad(total, 0))
r1[2].metric("Avg Order", cad(aov, 0))
r1[3].metric("Median", cad(median_val, 0))

r2 = st.columns(4)
r2[0].metric("Top Country", top_country if top_country else "-")
r2[1].metric("Top Channel", top_channel if top_channel else "-")
r2[2].metric("Consignment", f"{cons_rate:.1f}%" if np.isfinite(cons_rate) else "-")
r2[3].metric("Avg Ship Lag", f"{avg_lag:.1f} days" if np.isfinite(avg_lag) else "-")

tabs = st.tabs([
    "Overview",
    "Price Drivers",
    "Product Mix",
    "Customer Segments",
    "Geography & Channels",
    "Time",
    "Inventory Timing",
    "Ownership",
    "Seasonality",
    "Compliance",
    "Stats",
    "Data"
])

with tabs[0]:
    st.subheader("Insights")
    share_top = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
    lines = []
    if np.isfinite(share_top):
        lines.append(f"- {top_country} drives about {share_top*100:.1f}% of {metric}.")
    lines.append(f"- Top channel by {metric} is {top_channel}.")
    if np.isfinite(cons_rate):
        lines.append(f"- Consignment is {cons_rate:.1f}% of orders.")
    st.markdown("\n".join(lines) if lines else "-")

    st.subheader("Recommendations")
    recs = [
        "- Protect the top markets first, then scale the next tier using the best-performing channels.",
        "- Keep pricing rules consistent across grade/finish and reduce unnecessary discounting on premium items.",
        "- Treat shipping lag as an operations KPI; fix the worst country/city hotspots first.",
        "- Use lead sources and segment behavior to decide where to invest marketing and sales effort."
    ]
    st.markdown("\n".join(recs))

with tabs[1]:
    st.subheader("Price Drivers")

    grade_col = pick_col(f, ["Grade"])
    finish_col = pick_col(f, ["Finish"])
    color_col = pick_col(f, ["Color Count (#)", "Color Count"])

    if grade_col and "Price (CAD)" in f.columns:
        gdf = f.dropna(subset=[grade_col, "Price (CAD)"]).copy()
        fig = px.box(gdf, x=grade_col, y="Price (CAD)", points="outliers", title="Price Distribution by Grade ($ CAD)")
        fig.update_traces(hovertemplate="Grade: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "price_grade_box.html")
        st.markdown("**Insight:** Price ranges differ by grade; some grades consistently sell higher.")
        st.markdown("**Recommendation:** Keep premium grades premium and use discounts carefully on high-grade items.")
    else:
        st.info("Grade and/or Price (CAD) not available for this view.")

    if finish_col and "Price (CAD)" in f.columns:
        fdf = f.dropna(subset=[finish_col, "Price (CAD)"]).copy()
        top_fin = fdf[finish_col].value_counts().head(10).index.tolist()
        fdf = fdf[fdf[finish_col].isin(top_fin)]
        fig2 = px.box(fdf, x=finish_col, y="Price (CAD)", points="outliers", title="Price Distribution by Finish (Top 10) ($ CAD)")
        fig2.update_traces(hovertemplate="Finish: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "price_finish_box.html")
        st.markdown("**Insight:** Some finishes are priced higher and show bigger price spreads.")
        st.markdown("**Recommendation:** Standardize finish naming and build finish-based pricing guidelines.")
    else:
        st.info("Finish and/or Price (CAD) not available for this view.")

    if color_col and "Price (CAD)" in f.columns:
        cdf = f.dropna(subset=[color_col, "Price (CAD)"]).copy()
        fig3 = px.scatter(cdf, x=color_col, y="Price (CAD)", color=finish_col if finish_col else None, title="Price vs Colour Count ($ CAD)")
        fig3.update_traces(hovertemplate="Colours: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig3 = fig_tight(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "price_colour_scatter.html")
        st.markdown("**Insight:** This shows whether more colours generally align with higher prices.")
        st.markdown("**Recommendation:** If higher colour count supports higher prices, reflect it in product descriptions and pricing tiers.")
    else:
        st.info("Colour Count and/or Price (CAD) not available for this view.")

with tabs[2]:
    st.subheader("Product Mix")

    prod_col = pick_col(f, ["Product Type", "Product", "Item Type"])
    if prod_col:
        prod_tot = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
        top_prod = prod_tot.head(15)
        fig = px.bar(top_prod, x=prod_col, y="value", title=f"Top Product Types by {metric} ($ CAD)")
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "product_mix_top_types.html")
        st.markdown("**Insight:** A small set of product types drives most of the value.")
        st.markdown("**Recommendation:** Keep top product types consistently available and feature them in top channels.")
    else:
        st.info("Product Type column not found.")

    species_col = pick_col(f, ["Species"])
    if species_col:
        sp = f.groupby(species_col)[metric].sum().sort_values(ascending=False).head(15).reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(sp, x=species_col, y="value", title=f"Top Species by {metric} ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "product_mix_species.html")
        st.markdown("**Insight:** Certain species dominate sales value.")
        st.markdown("**Recommendation:** Build content and bundles around top species and test pricing elasticity with controlled discounting.")
    else:
        st.info("Species column not found.")

with tabs[3]:
    st.subheader("Customer Segments")

    cust_col = pick_col(f, ["Customer Type", "Customer Segment", "Segment"])
    if cust_col:
        seg = f.groupby(cust_col)[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
        fig = px.bar(seg, x=cust_col, y="value", title=f"{metric} by Customer Type ($ CAD)")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "segments_value.html")
        st.markdown("**Insight:** Customer types behave differently in total value and average order size.")
        st.markdown("**Recommendation:** Use separate offers and channel strategies for wholesale vs retail (or whichever segments exist).")

        if "Channel" in f.columns:
            mix = f.groupby([cust_col, "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            mix["seg_total"] = mix.groupby(cust_col)["value"].transform("sum")
            mix["share"] = mix["value"] / mix["seg_total"]
            fig2 = px.bar(mix, x=cust_col, y="share", color="Channel", barmode="stack", title="Channel Mix by Customer Type (Share)")
            fig2.update_layout(yaxis_tickformat=".0%")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "segments_channel_mix.html")
            st.markdown("**Insight:** Different customer types tend to prefer different channels.")
            st.markdown("**Recommendation:** Put your strongest channels behind the segments that already convert well there.")
    else:
        st.info("Customer Type column not found.")

with tabs[4]:
    st.subheader("Geography & Channels")

    country_totals2 = f.groupby("Country")[metric].sum().sort_values(ascending=False)
    agg = country_totals2.reset_index().rename(columns={metric: "value"})
    agg["share"] = agg["value"] / agg["value"].sum()

    fig = px.choropleth(
        agg,
        locations="Country",
        locationmode="country names",
        color="value",
        hover_name="Country",
        custom_data=["share"],
        projection="natural earth",
        title=f"World Map - {metric} ($ CAD)"
    )
    fig.update_traces(hovertemplate="<b>%{location}</b><br>Value: %{z:$,.0f} CAD<br>Share: %{customdata[0]:.1%}<extra></extra>")
    fig = fig_tight(fig)
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "geo_world_map.html")
    st.markdown("**Insight:** Revenue is concentrated in a small number of countries.")
    st.markdown("**Recommendation:** Protect the top markets first, then scale the next tier using the best channels in each market.")

    colA, colB = st.columns(2)

    with colA:
        top_c = country_totals2.head(top_n).reset_index().rename(columns={metric: "value"})
        fig1 = px.bar(top_c, x="Country", y="value", title=f"Top {top_n} Countries by {metric} ($ CAD)")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig1 = fig_tight(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        download_html(fig1, "geo_top_countries.html")
        st.markdown("**Insight:** The top countries generate most of the total value.")
        st.markdown("**Recommendation:** Use market tiers (Anchor / Growth / Test) and allocate inventory and marketing by tier.")

    with colB:
        ch_tot = f.groupby("Channel")[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(ch_tot, x="Channel", y="value", title=f"{metric} by Channel ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "geo_channel_bar.html")
        st.markdown("**Insight:** Some channels clearly outperform others on total value.")
        st.markdown("**Recommendation:** Put your best products and campaigns into the strongest channel(s) first.")

    st.subheader("Country × Channel Heatmap (Top countries)")
    top_idx = country_totals2.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
    fig3 = heatmap_from_pivot(pv, f"Heatmap: {metric} ($ CAD)", "$ CAD")
    st.plotly_chart(fig3, use_container_width=True)
    download_html(fig3, "geo_country_channel_heatmap.html")
    st.markdown("**Insight:** The winning channel is not the same in every country.")
    st.markdown("**Recommendation:** Choose 1–2 winning channels per top country and build a country-specific playbook.")

    if "Lead Source" in f.columns and f["Lead Source"].notna().any():
        st.subheader("Lead Source")
        ls = f.dropna(subset=["Lead Source"]).copy()
        col1, col2 = st.columns(2)

        with col1:
            ls_tot = ls.groupby("Lead Source")[metric].sum().sort_values(ascending=False).head(12).reset_index().rename(columns={metric: "value"})
            fig_ls = px.bar(ls_tot, x="Lead Source", y="value", title=f"{metric} by Lead Source (Top 12) ($ CAD)")
            fig_ls.update_layout(xaxis={"categoryorder": "total descending"})
            fig_ls.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig_ls = fig_tight(fig_ls)
            st.plotly_chart(fig_ls, use_container_width=True)
            download_html(fig_ls, "geo_lead_source_value.html")

        with col2:
            top_sources = ls_tot["Lead Source"].tolist()
            mix = ls[ls["Lead Source"].isin(top_sources)].groupby(["Lead Source", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            mix["total"] = mix.groupby("Lead Source")["value"].transform("sum")
            mix["share"] = mix["value"] / mix["total"]
            fig_m = px.bar(mix, x="Lead Source", y="share", color="Channel", barmode="stack", title="Channel Mix inside Lead Sources (Share)")
            fig_m.update_layout(yaxis_tickformat=".0%")
            fig_m = fig_tight(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)
            download_html(fig_m, "geo_lead_source_channel_mix.html")

        st.markdown("**Insight:** A few lead sources drive most value and have different channel patterns.")
        st.markdown("**Recommendation:** Invest more in the top lead sources and optimize the best channel for each source.")

    st.subheader("Top Markets Table")
    tbl = agg.sort_values("value", ascending=False).head(20).copy()
    tbl["Value ($ CAD)"] = tbl["value"].round(0).map(lambda v: f"${v:,.0f} CAD")
    tbl["Share (%)"] = (tbl["share"] * 100).round(2).map(lambda v: f"{v:.2f}%")
    show_tbl = rank_df(tbl[["Country", "Value ($ CAD)", "Share (%)"]])
    st.dataframe(show_tbl.set_index("#"), use_container_width=True)

with tabs[5]:
    st.subheader("Time Trends + Shipping Lag")

    st.markdown("### Monthly Trend")
    ts_df = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "value"})
    fig = px.line(ts_df, x="Month", y="value", title=f"Monthly {metric} ($ CAD)")
    fig.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    fig = fig_tight(fig)
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "time_monthly_trend.html")
    if len(ts_df) >= 2:
        last_val = float(ts_df["value"].iloc[-1])
        prev_val = float(ts_df["value"].iloc[-2])
        mom = (last_val / prev_val - 1) if prev_val else np.nan
        st.markdown(f"**Insight:** Latest month changed by **{mom*100:.1f}%** vs previous month." if np.isfinite(mom) else "**Insight:** Month-over-month change cannot be computed.")
    else:
        st.markdown("**Insight:** Not enough months to compute changes.")
    st.markdown("**Recommendation:** When the trend shifts, drill into countries/channels/products to identify the cause.")

    st.markdown("### Monthly Trend by Channel (Top 6)")
    ch_tot = f.groupby("Channel")[metric].sum().sort_values(ascending=False)
    top6 = ch_tot.head(6).index.tolist()
    by_ch = f[f["Channel"].isin(top6)].groupby(["Month", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
    figc = px.line(by_ch, x="Month", y="value", color="Channel", title=f"Monthly {metric} by Channel (Top 6) ($ CAD)")
    figc.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    figc = fig_tight(figc)
    st.plotly_chart(figc, use_container_width=True)
    download_html(figc, "time_monthly_by_channel_top6.html")
    st.markdown("**Insight:** Different channels contribute differently over time.")
    st.markdown("**Recommendation:** Use this to pinpoint which channel is driving growth or decline each month.")

    st.divider()
    st.subheader("Shipping Lag")

    lag_df = f.dropna(subset=[lag_col]).copy()
    if lag_df.empty:
        st.info("No usable shipping lag values after filters.")
    else:
        lag_df["Ship Lag (days)"] = lag_df[lag_col].astype(float)
        col1, col2 = st.columns(2)

        with col1:
            by_country = lag_df.groupby("Country")["Ship Lag (days)"].mean().sort_values(ascending=False).head(20).reset_index()
            fig1 = px.bar(by_country, x="Country", y="Ship Lag (days)", title="Avg Ship Lag by Country (days)")
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            fig1 = fig_tight(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            download_html(fig1, "ship_lag_by_country.html")
            st.markdown("**Insight:** Some countries consistently ship slower than others.")
            st.markdown("**Recommendation:** Set SLAs by country and adjust carrier/fulfillment strategy for slow destinations.")

            pick = st.selectbox(
                "Country → City drilldown",
                sorted(lag_df["Country"].dropna().unique().tolist()),
                key="ship_country_drill"
            )
            by_city = lag_df[lag_df["Country"] == pick].groupby("City")["Ship Lag (days)"].mean().sort_values(ascending=False).head(15).reset_index()
            fig2 = px.bar(by_city, x="City", y="Ship Lag (days)", title=f"Avg Ship Lag by City in {pick} (Top 15)")
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "ship_lag_by_city.html")
            st.markdown("**Insight:** Within a country, delays often concentrate in a few cities.")
            st.markdown("**Recommendation:** Fix the worst cities first for the fastest improvement.")

        with col2:
            min_orders = st.slider("Minimum orders per Country+City", 2, 15, 5, key="ship_min_orders")
            cc = (lag_df.groupby(["Country", "City"]).agg(
                orders=("Sale ID", "count"),
                avg_lag=("Ship Lag (days)", "mean"),
                med_lag=("Ship Lag (days)", "median"),
                total_value=(metric, "sum")
            ).reset_index())
            cc = cc[cc["orders"] >= min_orders].sort_values(["avg_lag", "orders"], ascending=[False, False]).head(25).copy()
            cc["total_value"] = cc["total_value"].round(0).map(lambda v: f"${v:,.0f} CAD")
            cc["avg_lag"] = cc["avg_lag"].round(1)
            cc["med_lag"] = cc["med_lag"].round(1)
            cc = rank_df(cc).rename(columns={"total_value": f"Total ({metric})"})
            st.dataframe(cc.set_index("#")[["Country", "City", "orders", "avg_lag", "med_lag", f"Total ({metric})"]], use_container_width=True)
            st.markdown("**Insight:** These are the biggest delay hotspots with enough volume to trust.")
            st.markdown("**Recommendation:** Prioritize hotspots with both high delay and meaningful order volume.")

            samp = lag_df.copy()
            if len(samp) > 2500:
                samp = samp.sample(2500, random_state=7)
            fig4 = px.scatter(samp, x="Ship Lag (days)", y=metric, color="Channel", title=f"Ship Lag vs {metric} ($ CAD)", hover_data=["Country", "City"])
            fig4.update_traces(hovertemplate="Lag: %{x:.0f} days<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig4 = fig_tight(fig4)
            st.plotly_chart(fig4, use_container_width=True)
            download_html(fig4, "ship_lag_scatter.html")
            st.markdown("**Insight:** This checks whether high-value orders ship faster or slower (often weak).")
            st.markdown("**Recommendation:** Improve lag for customer experience even if revenue impact is small.")

with tabs[6]:
    st.subheader("Inventory Timing")

    if "Product Type" in f.columns:
        top_prod = f.groupby("Product Type")[metric].sum().sort_values(ascending=False).head(12).index.tolist()
        inv = f[f["Product Type"].isin(top_prod)].copy()
        pv = inv.pivot_table(values=metric, index="Month", columns="Product Type", aggfunc="sum", fill_value=0)
        fig = heatmap_from_pivot(pv, f"{metric} Heatmap by Month × Product Type ($ CAD)", "$ CAD")
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "inventory_month_product_heatmap.html")
        st.markdown("**Insight:** This shows when certain product types spike in value across months.")
        st.markdown("**Recommendation:** Use this to plan inventory intake and promotions ahead of expected peaks.")

        orders_m = inv.groupby(["Month", "Product Type"])["Sale ID"].count().reset_index().rename(columns={"Sale ID": "orders"})
        fig2 = px.line(orders_m, x="Month", y="orders", color="Product Type", title="Monthly Orders by Product Type (Top 12)")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "inventory_monthly_orders_by_product.html")
        st.markdown("**Insight:** Order volume trends can differ from revenue trends.")
        st.markdown("**Recommendation:** Plan stock using both volume and value so you do not overstock low-value items.")
    else:
        st.info("Product Type column not found.")

with tabs[7]:
    st.subheader("Ownership")

    own_col = pick_col(f, ["Consignment? (Y/N)", "Consigned? (Y/N)", "Ownership"])
    if own_col:
        odf = f.copy()
        odf["Ownership"] = odf[own_col].astype(str).str.upper().replace({"Y": "Consigned", "N": "Owned"})
        own_tot = odf.groupby("Ownership")[metric].sum().reset_index().rename(columns={metric: "value"})
        fig = px.bar(own_tot, x="Ownership", y="value", title=f"{metric} by Ownership ($ CAD)")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "ownership_value.html")
        st.markdown("**Insight:** Ownership mix affects how value is generated and risk is managed.")
        st.markdown("**Recommendation:** Use consignment to reduce cash risk and owned inventory to improve control and availability.")

        if "Price (CAD)" in odf.columns:
            bx = odf.dropna(subset=["Ownership", "Price (CAD)"])
            fig2 = px.box(bx, x="Ownership", y="Price (CAD)", points="outliers", title="Price Distribution by Ownership ($ CAD)")
            fig2.update_traces(hovertemplate="%{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "ownership_price_box.html")
            st.markdown("**Insight:** Typical prices can differ between consigned and owned items.")
            st.markdown("**Recommendation:** If consigned items skew premium, present them as curated/high-end with clear lead-time expectations.")
    else:
        st.info("Ownership/consignment column not found.")

with tabs[8]:
    st.subheader("Seasonality")

    m = f.groupby("MonthNum")[metric].sum().reset_index().rename(columns={metric: "value"})
    if not m.empty:
        m["MonthNum"] = m["MonthNum"].astype(int)
        fig = px.bar(m.sort_values("MonthNum"), x="MonthNum", y="value", title=f"Seasonality Pattern (Month of Year) • {metric} ($ CAD)")
        fig.update_traces(hovertemplate="Month: %{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "seasonality_month_of_year.html")
        st.markdown("**Insight:** Some months are consistently stronger across the year.")
        st.markdown("**Recommendation:** Plan inventory intake and marketing pushes ahead of high months; use weak months for campaigns and pipeline building.")
    else:
        st.info("Not enough date data to compute seasonality.")

    top_ctry = country_totals.head(6).index.tolist()
    if top_ctry:
        sub = f[f["Country"].isin(top_ctry)].groupby(["Month", "Country"])[metric].sum().reset_index().rename(columns={metric: "value"})
        fig2 = px.line(sub, x="Month", y="value", color="Country", title=f"Monthly {metric} by Country (Top 6) ($ CAD)")
        fig2.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "seasonality_monthly_by_country_top6.html")
        st.markdown("**Insight:** Peaks and dips vary by market, not just globally.")
        st.markdown("**Recommendation:** Use market-specific calendars for campaigns and inventory allocation.")
    else:
        st.info("No country data available for this view.")

with tabs[9]:
    st.subheader("Compliance")

    coa_col = pick_col(f, ["COA #", "COA", "COA Provided", "COA Provided? (Y/N)", "Certificate of Authenticity"])
    exp_col = pick_col(f, ["Export Permit (PDF link)", "Export Permit", "Export", "Export? (Y/N)", "Export Flag", "Exported"])

    if not coa_col and not exp_col:
        st.info("No compliance columns found (COA / Export).")
    else:
        colA, colB = st.columns(2)

        with colA:
            if coa_col:
                cdf = f.copy()
                if "Y/N" in str(coa_col).upper():
                    cdf["COA Status"] = cdf[coa_col].astype(str).str.upper().replace({"Y": "COA Present", "N": "COA Missing"})
                else:
                    cdf["COA Status"] = np.where(cdf[coa_col].notna(), "COA Present", "COA Missing")
                coa = cdf.groupby("COA Status")[metric].sum().reset_index().rename(columns={metric: "value"})
                fig = px.bar(coa, x="COA Status", y="value", title=f"{metric} by COA Status ($ CAD)")
                fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
                fig = fig_tight(fig)
                st.plotly_chart(fig, use_container_width=True)
                download_html(fig, "compliance_coa.html")
                st.markdown("**Insight:** Missing COA can still exist in meaningful value flow.")
                st.markdown("**Recommendation:** Enforce COA completion for high-value items and export-bound orders.")
            else:
                st.info("COA column not found.")

        with colB:
            if exp_col:
                edf = f.copy()
                if "Y/N" in str(exp_col).upper():
                    edf["Export Status"] = edf[exp_col].astype(str).str.upper().replace({"Y": "Export", "N": "Domestic"})
                else:
                    edf["Export Status"] = np.where(edf[exp_col].notna(), "Export", "Domestic/Unknown")
                exp = edf.groupby(["Export Status", "Country"])[metric].sum().reset_index().rename(columns={metric: "value"})
                top_ctry = country_totals.head(12).index
                exp = exp[exp["Country"].isin(top_ctry)]
                fig2 = px.bar(exp, x="Country", y="value", color="Export Status", barmode="group", title=f"{metric} by Country and Export Status (Top Countries)")
                fig2.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
                fig2 = fig_tight(fig2)
                st.plotly_chart(fig2, use_container_width=True)
                download_html(fig2, "compliance_export.html")
                st.markdown("**Insight:** Export-related value is concentrated in certain markets.")
                st.markdown("**Recommendation:** Tighten documentation controls in top export markets to reduce compliance risk.")
            else:
                st.info("Export column not found.")

with tabs[10]:
    st.subheader("Stats")

    st.markdown("### 1) Are typical order values different across channels?")
    if sps is None:
        st.write("p-value: -")
        st.markdown("**Insight:** SciPy is not installed, so tests are disabled (charts still work).")
        st.markdown("**Recommendation:** Add `scipy` to requirements.txt if you want p-values on Streamlit Cloud.")
    else:
        grp = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
        if len(grp) >= 2:
            _, p = sps.kruskal(*grp.tolist())
            st.write(f"p-value: **{p_fmt(p)}**")
            st.markdown("**Insight:** If p < 0.05, channels likely have different typical order values.")
            st.markdown("**Recommendation:** Focus on channels that combine strong median value and strong total volume.")
        else:
            st.write("p-value: -")
            st.markdown("**Insight:** Not enough channel groups in the current filter.")
            st.markdown("**Recommendation:** Expand the date range or remove filters to increase sample size.")

    med = f.groupby("Channel")[metric].median().sort_values(ascending=False).reset_index().rename(columns={metric: "Median"})
    med["Median"] = med["Median"].round(0).map(lambda v: f"${v:,.0f} CAD")
    st.dataframe(med, use_container_width=True)

    st.markdown("### 2) Quick numeric drivers (correlation)")
    driver_candidates = ["Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Color Count (#)", "length", "width", "weight", lag_col]
    drivers = [c for c in driver_candidates if c in f.columns]
    rows = []
    for c in drivers:
        x = f[c]
        y = f[metric]
        ok = x.notna() & y.notna()
        if ok.sum() >= 30:
            if sps is not None:
                r, pv = sps.spearmanr(x[ok], y[ok])
                rows.append((c, float(r), p_fmt(pv), int(ok.sum())))
            else:
                r = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")
                rows.append((c, float(r), "-", int(ok.sum())))
    if rows:
        out = pd.DataFrame(rows, columns=["Variable", "Spearman r", "p-value", "n"])
        out["abs_r"] = out["Spearman r"].abs()
        out = out.sort_values("abs_r", ascending=False).drop(columns=["abs_r"]).head(10).reset_index(drop=True)
        out["Spearman r"] = out["Spearman r"].round(3)
        out = rank_df(out)
        st.dataframe(out.set_index("#"), use_container_width=True)
        st.markdown("**Insight:** Bigger |r| means a stronger relationship with the chosen metric.")
        st.markdown("**Recommendation:** Use the top 2–3 drivers as dashboard filters/KPIs and ignore weak drivers.")
    else:
        st.write("-")
        st.markdown("**Insight:** Not enough numeric data under current filters to compute relationships.")
        st.markdown("**Recommendation:** Expand date range or remove filters to increase sample size.")

with tabs[11]:
    st.subheader("Data")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Top Countries")
        t = country_totals.reset_index().rename(columns={metric: "Total ($ CAD)"}).head(25)
        t["Total ($ CAD)"] = t["Total ($ CAD)"].round(0).map(lambda v: f"${v:,.0f} CAD")
        st.dataframe(rank_df(t).set_index("#"), use_container_width=True)

        st.markdown("### Top Cities (Country + City)")
        cct = f.groupby(["Country", "City"])[metric].sum().sort_values(ascending=False).head(25).reset_index().rename(columns={metric: "Total ($ CAD)"})
        cct["Total ($ CAD)"] = cct["Total ($ CAD)"].round(0).map(lambda v: f"${v:,.0f} CAD")
        st.dataframe(rank_df(cct).set_index("#"), use_container_width=True)

    with colB:
        st.markdown("### Country × Channel KPI")
        kpi = f.groupby(["Country", "Channel"]).agg(
            orders=("Sale ID", "count"),
            total=(metric, "sum"),
            avg=(metric, "mean"),
            median=(metric, "median"),
            avg_ship_lag=(lag_col, "mean"),
            avg_discount_rate=("Discount Rate", "mean")
        ).reset_index()

        kpi["total"] = kpi["total"].round(0).map(lambda v: f"${v:,.0f} CAD")
        kpi["avg"] = kpi["avg"].round(0).map(lambda v: f"${v:,.0f} CAD")
        kpi["median"] = kpi["median"].round(0).map(lambda v: f"${v:,.0f} CAD")
        kpi["avg_ship_lag"] = kpi["avg_ship_lag"].round(1)
        kpi["avg_discount_rate"] = (kpi["avg_discount_rate"] * 100).round(1)

        kpi = kpi.sort_values("orders", ascending=False).head(50)
        kpi = rank_df(kpi).rename(columns={
            "total": "Total ($ CAD)",
            "avg": "Avg ($ CAD)",
            "median": "Median ($ CAD)",
            "avg_ship_lag": "Avg Ship Lag (days)",
            "avg_discount_rate": "Avg Discount (%)"
        })
        st.dataframe(kpi.set_index("#"), use_container_width=True)

        st.download_button(
            "Download filtered data (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
            key="dl_filtered_csv"
        )

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
