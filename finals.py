import pathlib
from uuid import uuid4
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

try:
    from scipy import stats as sps
except Exception:
    sps = None

st.set_page_config(page_title="FAANG-Level Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 0.86rem !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1560px; }
div[data-testid="column"] { padding-left: 0.35rem; padding-right: 0.35rem; }
[data-testid="metric-container"] { padding: 0.7rem 0.85rem; border-radius: 14px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.12rem !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
    line-height: 1.15 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
.js-plotly-plot .plot-container { width: 100% !important; }
[data-testid="stDataFrame"] { width: 100% !important; }
hr { margin: 0.65rem 0 0.65rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("Lucentara & Dinosty Fossils • Full Team Dashboard")
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

def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out

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

def safe_series(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else pd.Series([np.nan] * len(df))

def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mu = np.nanmean(s.values) if s.notna().any() else np.nan
    sd = np.nanstd(s.values) if s.notna().any() else np.nan
    if not np.isfinite(sd) or sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd

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

compare_prev = st.sidebar.toggle("Compare vs previous period", value=True, key="compare_prev")
exclude_negative_lag = st.sidebar.toggle("Exclude negative ship lag", value=True, key="toggle_excl_neg_lag")
top_n = st.sidebar.slider("Top N (countries)", 5, 30, 12, key="top_n")

search_text = st.sidebar.text_input("Quick search (Country/City/Product)", value="", key="quick_search").strip().lower()

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

if search_text:
    ctry_mask = f["Country"].fillna("").astype(str).str.lower().str.contains(search_text)
    city_mask = f["City"].fillna("").astype(str).str.lower().str.contains(search_text)
    prod_mask = f["Product Type"].fillna("").astype(str).str.lower().str.contains(search_text) if "Product Type" in f.columns else False
    f = f[ctry_mask | city_mask | prod_mask]

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

prev_f = None
delta_total = delta_orders = delta_aov = delta_median = np.nan
drivers_country = drivers_channel = None

if compare_prev:
    span_days = (end.normalize() - start.normalize()).days + 1
    prev_end = start - pd.Timedelta(seconds=1)
    prev_start = start - pd.Timedelta(days=span_days)
    prev = df[(df["Date"] >= prev_start) & (df["Date"] <= prev_end)].copy()
    if sel_countries:
        prev = prev[prev["Country"].isin(sel_countries)]
    if sel_channels:
        prev = prev[prev["Channel"].isin(sel_channels)]
    if sel_products and "Product Type" in prev.columns:
        prev = prev[prev["Product Type"].isin(sel_products)]
    if sel_customers and "Customer Type" in prev.columns:
        prev = prev[prev["Customer Type"].isin(sel_customers)]
    if sel_cities:
        prev = prev[prev["City"].isin(sel_cities)]
    if search_text:
        ctry_mask2 = prev["Country"].fillna("").astype(str).str.lower().str.contains(search_text)
        city_mask2 = prev["City"].fillna("").astype(str).str.lower().str.contains(search_text)
        prod_mask2 = prev["Product Type"].fillna("").astype(str).str.lower().str.contains(search_text) if "Product Type" in prev.columns else False
        prev = prev[ctry_mask2 | city_mask2 | prod_mask2]
    prev_f = prev if not prev.empty else None

    if prev_f is not None:
        prev_total = float(prev_f[metric].sum())
        prev_orders = int(len(prev_f))
        prev_aov = float(prev_f[metric].mean())
        prev_median = float(prev_f[metric].median())

        delta_total = (total / prev_total - 1) if prev_total else np.nan
        delta_orders = (orders / prev_orders - 1) if prev_orders else np.nan
        delta_aov = (aov / prev_aov - 1) if prev_aov else np.nan
        delta_median = (median_val / prev_median - 1) if prev_median else np.nan

        cur_c = f.groupby("Country")[metric].sum()
        prv_c = prev_f.groupby("Country")[metric].sum()
        diff_c = (cur_c - prv_c).dropna().sort_values(ascending=False)
        drivers_country = diff_c.reset_index().rename(columns={0: "delta"})

        cur_ch = f.groupby("Channel")[metric].sum()
        prv_ch = prev_f.groupby("Channel")[metric].sum()
        diff_ch = (cur_ch - prv_ch).dropna().sort_values(ascending=False)
        drivers_channel = diff_ch.reset_index().rename(columns={0: "delta"})

def delta_badge(d):
    if d is None or (isinstance(d, float) and not np.isfinite(d)):
        return "-"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.1f}%"

r1 = st.columns(4)
r1[0].metric("Orders", f"{orders:,}", delta=delta_badge(delta_orders) if compare_prev else None)
r1[1].metric("Total", cad(total, 0), delta=delta_badge(delta_total) if compare_prev else None)
r1[2].metric("Avg Order", cad(aov, 0), delta=delta_badge(delta_aov) if compare_prev else None)
r1[3].metric("Median", cad(median_val, 0), delta=delta_badge(delta_median) if compare_prev else None)

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
    st.subheader("Executive Insights")
    bullets = []
    share_top = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
    if np.isfinite(share_top):
        bullets.append(f"- {top_country} drives about {share_top*100:.1f}% of {metric}.")
    if len(channel_totals) >= 2:
        top_share_ch = float(channel_totals.iloc[0] / channel_totals.sum()) if channel_totals.sum() else np.nan
        if np.isfinite(top_share_ch):
            bullets.append(f"- {top_channel} is the top channel and contributes about {top_share_ch*100:.1f}% of {metric}.")
    if np.isfinite(cons_rate):
        bullets.append(f"- Consignment is {cons_rate:.1f}% of orders.")
    if compare_prev and prev_f is not None and np.isfinite(delta_total):
        direction = "up" if delta_total >= 0 else "down"
        bullets.append(f"- Period-over-period total is {direction} {abs(delta_total*100):.1f}% for {metric}.")
    st.markdown("\n".join(bullets) if bullets else "-")

    st.subheader("Opportunity Radar")
    lag_df = f.dropna(subset=[lag_col]).copy()
    if not lag_df.empty:
        lag_df["Ship Lag (days)"] = pd.to_numeric(lag_df[lag_col], errors="coerce")
        hot = (lag_df.groupby(["Country", "City"]).agg(
            orders=("Sale ID", "count"),
            avg_lag=("Ship Lag (days)", "mean"),
            total_value=(metric, "sum")
        ).reset_index())
        hot = hot[hot["orders"] >= 3].copy()
        if not hot.empty:
            hot["score"] = zscore(hot["total_value"]) + zscore(hot["avg_lag"]) + zscore(hot["orders"])
            hot = hot.sort_values("score", ascending=False).head(15).copy()
            hot["Total ($ CAD)"] = hot["total_value"].round(0).map(lambda v: f"${v:,.0f} CAD")
            hot["Avg Lag (days)"] = hot["avg_lag"].round(1)
            hot = hot[["Country", "City", "orders", "Avg Lag (days)", "Total ($ CAD)", "score"]].copy()
            hot = rank_df(hot.drop(columns=["score"]))
            st.dataframe(hot.set_index("#"), use_container_width=True)
            st.markdown("**Insight:** These are the highest-impact shipping hotspots (value + delay + volume).")
            st.markdown("**Recommendation:** Fix the top 3 hotspots first (carrier, dispatch, routing, packaging SLA).")
        else:
            st.write("-")
    else:
        st.write("-")

    if compare_prev and drivers_country is not None and not drivers_country.empty:
        st.subheader("What drove the change")
        c = drivers_country.rename(columns={"delta": "Delta ($ CAD)"})
        c["Delta ($ CAD)"] = c["Delta ($ CAD)"].fillna(0.0)
        c = c.sort_values("Delta ($ CAD)", ascending=False).head(12).copy()
        fig = px.bar(c, x="Country", y="Delta ($ CAD)", title="Top Country Contributors (Current vs Previous)", text_auto=".2s")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Delta: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "overview_country_contributors.html")
        st.markdown("**Insight:** These countries explain most of the period-over-period movement.")
        st.markdown("**Recommendation:** Protect winners (double down where delta is positive) and investigate declines (stockouts, channel shift, ops delays).")

with tabs[1]:
    st.subheader("Price Drivers")
    grade_col = pick_col(f, ["Grade"])
    finish_col = pick_col(f, ["Finish"])
    color_col = pick_col(f, ["Color Count (#)", "Color Count"])

    left, right = st.columns(2)

    with left:
        if grade_col and "Price (CAD)" in f.columns:
            gdf = f.dropna(subset=[grade_col, "Price (CAD)"]).copy()
            top_grades = gdf[grade_col].value_counts().head(12).index.tolist()
            gdf = gdf[gdf[grade_col].isin(top_grades)]
            fig = px.box(gdf, x=grade_col, y="Price (CAD)", points="outliers", title="Price Distribution by Grade (Top 12) ($ CAD)")
            fig.update_traces(hovertemplate="Grade: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
            fig = fig_tight(fig)
            st.plotly_chart(fig, use_container_width=True)
            download_html(fig, "price_grade_box.html")
            med_by = gdf.groupby(grade_col)["Price (CAD)"].median().sort_values(ascending=False)
            best = med_by.index[0] if len(med_by) else "-"
            st.markdown(f"**Insight:** Grade impacts typical price; top median grade here is {best}.")
            st.markdown("**Recommendation:** Keep premium grades premium, limit discounting, and align inventory to high-median grades.")
        else:
            st.info("Grade and/or Price (CAD) not available for this view.")

    with right:
        if finish_col and "Price (CAD)" in f.columns:
            fdf = f.dropna(subset=[finish_col, "Price (CAD)"]).copy()
            top_fin = fdf[finish_col].value_counts().head(10).index.tolist()
            fdf = fdf[fdf[finish_col].isin(top_fin)]
            fig2 = px.violin(fdf, x=finish_col, y="Price (CAD)", box=True, points="outliers", title="Price by Finish (Top 10) ($ CAD)")
            fig2.update_traces(hovertemplate="Finish: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "price_finish_violin.html")
            st.markdown("**Insight:** Finish type shows different pricing bands and variability.")
            st.markdown("**Recommendation:** Standardize finish naming and add finish-based price guardrails to reduce inconsistent pricing.")
        else:
            st.info("Finish and/or Price (CAD) not available for this view.")

    if color_col and "Price (CAD)" in f.columns:
        cdf = f.dropna(subset=[color_col, "Price (CAD)"]).copy()
        if len(cdf) > 5000:
            cdf = cdf.sample(5000, random_state=11)
        fig3 = px.scatter(
            cdf,
            x=color_col,
            y="Price (CAD)",
            color=finish_col if finish_col else None,
            trendline=None,
            title="Price vs Colour Count ($ CAD)"
        )
        fig3.update_traces(hovertemplate="Colours: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig3 = fig_tight(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "price_colour_scatter.html")
        corr = pd.to_numeric(cdf[color_col], errors="coerce").corr(pd.to_numeric(cdf["Price (CAD)"], errors="coerce"), method="spearman")
        if np.isfinite(corr):
            st.markdown(f"**Insight:** Colour count vs price correlation (Spearman) is about {corr:.2f} in this view.")
        else:
            st.markdown("**Insight:** Colour count relationship is unclear due to missing/non-numeric values.")
        st.markdown("**Recommendation:** If colour count is a consistent driver, reflect it in product storytelling and tiered pricing.")
    else:
        st.info("Colour Count and/or Price (CAD) not available for this view.")

with tabs[2]:
    st.subheader("Product Mix")
    prod_col = pick_col(f, ["Product Type", "Product", "Item Type"])
    species_col = pick_col(f, ["Species"])

    colA, colB = st.columns(2)

    with colA:
        if prod_col:
            prod_tot = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
            top_prod = prod_tot.head(15).copy()
            fig = px.bar(top_prod, x=prod_col, y="value", title=f"Top Product Types by {metric} ($ CAD)")
            fig.update_layout(xaxis={"categoryorder": "total descending"})
            fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig = fig_tight(fig)
            st.plotly_chart(fig, use_container_width=True)
            download_html(fig, "product_mix_top_types.html")
            top_name = top_prod.iloc[0][prod_col] if len(top_prod) else "-"
            st.markdown(f"**Insight:** Product mix is concentrated; #1 type here is {top_name}.")
            st.markdown("**Recommendation:** Protect availability for top types, and test bundling/upsells on mid-tier types.")
        else:
            st.info("Product Type column not found.")

    with colB:
        if prod_col:
            mix = f.groupby([prod_col, "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            mix_tot = mix.groupby(prod_col)["value"].transform("sum")
            mix["share"] = np.where(mix_tot > 0, mix["value"] / mix_tot, np.nan)
            top_types = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).head(10).index.tolist()
            mix = mix[mix[prod_col].isin(top_types)]
            figm = px.bar(mix, x=prod_col, y="share", color="Channel", barmode="stack", title="Channel Mix by Product Type (Top 10)")
            figm.update_layout(yaxis_tickformat=".0%")
            figm = fig_tight(figm)
            st.plotly_chart(figm, use_container_width=True)
            download_html(figm, "product_mix_channel_share.html")
            st.markdown("**Insight:** Different product types sell through different channels.")
            st.markdown("**Recommendation:** Match each product type to its strongest channel instead of using one channel plan for all products.")
        else:
            st.info("Product Type column not found.")

    if species_col:
        sp = f.groupby(species_col)[metric].sum().sort_values(ascending=False).head(15).reset_index().rename(columns={metric: "value"})
        fig2 = px.treemap(sp, path=[species_col], values="value", title=f"Species Value Treemap ({metric})")
        fig2.update_traces(hovertemplate="%{label}<br>Value: %{value:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "product_mix_species_treemap.html")
        st.markdown("**Insight:** Treemap highlights which species dominate value at a glance.")
        st.markdown("**Recommendation:** Focus content and catalog depth on the largest species blocks; run small tests on smaller blocks.")
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

        seg2 = f.groupby(cust_col).agg(
            orders=("Sale ID", "count"),
            avg_value=(metric, "mean"),
            median_value=(metric, "median"),
            avg_discount=("Discount Rate", "mean")
        ).reset_index()
        seg2["avg_value"] = seg2["avg_value"].round(0).map(lambda v: f"${v:,.0f} CAD")
        seg2["median_value"] = seg2["median_value"].round(0).map(lambda v: f"${v:,.0f} CAD")
        seg2["avg_discount"] = (seg2["avg_discount"] * 100).round(1).map(lambda v: f"{v:.1f}%")
        seg2 = seg2.sort_values("orders", ascending=False)
        st.dataframe(seg2, use_container_width=True)

        st.markdown("**Insight:** Customer types differ in value, typical order size, and discount behavior.")
        st.markdown("**Recommendation:** Create segment-specific offers (pricing/discount limits), and route each segment to its best channel.")
    else:
        st.info("Customer Type column not found.")

with tabs[4]:
    st.subheader("Geography & Channels")

    agg = country_totals.reset_index().rename(columns={metric: "value"})
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
    st.markdown("**Insight:** Value is concentrated in a few countries (darker shading).")
    st.markdown("**Recommendation:** Defend top markets first (inventory + service level), then expand using market-specific channel strategies.")

    colA, colB = st.columns(2)

    with colA:
        top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "value"})
        fig1 = px.bar(top_c, x="Country", y="value", title=f"Top {top_n} Countries by {metric} ($ CAD)")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig1 = fig_tight(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        download_html(fig1, "geo_top_countries.html")
        st.markdown("**Insight:** The top countries generate most of the total value.")
        st.markdown("**Recommendation:** Use tiering (Anchor / Growth / Test) and allocate budget + inventory accordingly.")

    with colB:
        ch_tot = channel_totals.reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(ch_tot, x="Channel", y="value", title=f"{metric} by Channel ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "geo_channel_bar.html")
        st.markdown("**Insight:** Channels differ a lot in contribution.")
        st.markdown("**Recommendation:** Put best inventory and campaigns into the top channels; keep weaker channels for targeted experiments.")

    st.subheader("Country × Channel Heatmap (Top countries)")
    top_idx = country_totals.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
    fig3 = heatmap_from_pivot(pv, f"Heatmap: {metric} ($ CAD)", "$ CAD")
    st.plotly_chart(fig3, use_container_width=True)
    download_html(fig3, "geo_country_channel_heatmap.html")
    st.markdown("**Insight:** The best channel is not the same in every country (hot cells show concentration).")
    st.markdown("**Recommendation:** Pick 1–2 winning channels per top country and build a market-specific playbook.")

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

        st.markdown("**Insight:** A few lead sources drive most value and show different channel patterns.")
        st.markdown("**Recommendation:** Invest more in the top lead sources and optimize the best channel for each source.")

    st.subheader("Top Markets Table")
    tbl = agg.sort_values("value", ascending=False).head(20).copy()
    tbl["Value ($ CAD)"] = tbl["value"].round(0).map(lambda v: f"${v:,.0f} CAD")
    tbl["Share (%)"] = (tbl["share"] * 100).round(2).map(lambda v: f"{v:.2f}%")
    st.dataframe(rank_df(tbl[["Country", "Value ($ CAD)", "Share (%)"]]).set_index("#"), use_container_width=True)

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
        st.markdown(f"**Insight:** Latest month changed by {mom*100:.1f}% vs previous month." if np.isfinite(mom) else "**Insight:** Month-over-month cannot be computed.")
    else:
        st.markdown("**Insight:** Not enough months to compute changes.")
    st.markdown("**Recommendation:** When the line moves, drill into channels/countries to identify the driver.")

    st.markdown("### Monthly Trend by Channel (Top 6)")
    ch_tot = f.groupby("Channel")[metric].sum().sort_values(ascending=False)
    top6 = ch_tot.head(6).index.tolist()
    by_ch = f[f["Channel"].isin(top6)].groupby(["Month", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
    figc = px.line(by_ch, x="Month", y="value", color="Channel", title=f"Monthly {metric} by Channel (Top 6) ($ CAD)")
    figc.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    figc = fig_tight(figc)
    st.plotly_chart(figc, use_container_width=True)
    download_html(figc, "time_monthly_by_channel_top6.html")
    st.markdown("**Insight:** Channels contribute differently over time.")
    st.markdown("**Recommendation:** Use this to pinpoint which channel caused the monthly spike/dip.")

    if compare_prev and drivers_channel is not None and not drivers_channel.empty:
        dch = drivers_channel.rename(columns={"delta": "Delta ($ CAD)"})
        dch["Delta ($ CAD)"] = dch["Delta ($ CAD)"].fillna(0.0)
        dch = dch.sort_values("Delta ($ CAD)", ascending=False).head(10).copy()
        figd = px.bar(dch, x="Channel", y="Delta ($ CAD)", title="Channel Contributors (Current vs Previous)", text_auto=".2s")
        figd.update_traces(hovertemplate="<b>%{x}</b><br>Delta: %{y:$,.0f} CAD<extra></extra>")
        figd = fig_tight(figd)
        st.plotly_chart(figd, use_container_width=True)
        download_html(figd, "time_channel_contributors.html")
        st.markdown("**Insight:** These channels explain most of the period-over-period movement.")
        st.markdown("**Recommendation:** Reinforce the winners and investigate declines (campaign changes, channel capacity, inventory fit).")

    st.divider()
    st.subheader("Shipping Lag")
    lag_df = f.dropna(subset=[lag_col]).copy()

    if lag_df.empty:
        st.info("No usable shipping lag values after filters.")
    else:
        lag_df["Ship Lag (days)"] = pd.to_numeric(lag_df[lag_col], errors="coerce")
        col1, col2 = st.columns(2)

        with col1:
            by_country = lag_df.groupby("Country")["Ship Lag (days)"].mean().sort_values(ascending=False).head(20).reset_index()
            fig1 = px.bar(by_country, x="Country", y="Ship Lag (days)", title="Avg Ship Lag by Country (days)")
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            fig1 = fig_tight(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            download_html(fig1, "ship_lag_by_country.html")
            st.markdown("**Insight:** Some countries consistently ship slower than others.")
            st.markdown("**Recommendation:** Set country-level SLAs and adjust carriers/fulfillment where lag is highest.")

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
            st.markdown("**Insight:** Delays often concentrate in a handful of cities.")
            st.markdown("**Recommendation:** Fix the worst cities first for the fastest impact.")

        with col2:
            min_orders = st.slider("Minimum orders per Country+City", 2, 15, 5, key="ship_min_orders")
            cc = (lag_df.groupby(["Country", "City"]).agg(
                orders=("Sale ID", "count"),
                avg_lag=("Ship Lag (days)", "mean"),
                med_lag=("Ship Lag (days)", "median"),
                total_value=(metric, "sum")
            ).reset_index())
            cc = cc[cc["orders"] >= min_orders].sort_values(["avg_lag", "orders"], ascending=[False, False]).head(25).copy()
            cc["total_value_fmt"] = cc["total_value"].round(0).map(lambda v: f"${v:,.0f} CAD")
            cc["avg_lag"] = cc["avg_lag"].round(1)
            cc["med_lag"] = cc["med_lag"].round(1)
            cc = rank_df(cc.rename(columns={"total_value_fmt": f"Total ({metric})"}))
            st.dataframe(cc.set_index("#")[["Country", "City", "orders", "avg_lag", "med_lag", f"Total ({metric})"]], use_container_width=True)
            st.markdown("**Insight:** These are the highest-delay hotspots with enough volume to trust.")
            st.markdown("**Recommendation:** Prioritize hotspots with high lag and high value to improve customer experience fast.")

            samp = lag_df.copy()
            if len(samp) > 3500:
                samp = samp.sample(3500, random_state=7)
            fig4 = px.scatter(
                samp,
                x="Ship Lag (days)",
                y=metric,
                color="Channel",
                title=f"Ship Lag vs {metric} ($ CAD)",
                hover_data=["Country", "City"]
            )
            fig4.update_traces(hovertemplate="Lag: %{x:.0f} days<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig4 = fig_tight(fig4)
            st.plotly_chart(fig4, use_container_width=True)
            download_html(fig4, "ship_lag_scatter.html")
            st.markdown("**Insight:** This checks if high-value orders ship faster or slower (often weak).")
            st.markdown("**Recommendation:** Improve lag primarily for customer experience and repeat purchase, not only revenue.")

with tabs[6]:
    st.subheader("Inventory Timing")
    prod_col = pick_col(f, ["Product Type", "Product", "Item Type"])
    if prod_col:
        top_prod = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).head(12).index.tolist()
        inv = f[f[prod_col].isin(top_prod)].copy()
        pv = inv.pivot_table(values=metric, index="Month", columns=prod_col, aggfunc="sum", fill_value=0)
        fig = heatmap_from_pivot(pv, f"{metric} Heatmap by Month × Product Type ($ CAD)", "$ CAD")
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "inventory_month_product_heatmap.html")
        st.markdown("**Insight:** Some product types spike in specific months.")
        st.markdown("**Recommendation:** Pull inventory intake forward before spike months; avoid overstocking low-value periods.")

        vol = inv.groupby(["Month", prod_col])["Sale ID"].count().reset_index().rename(columns={"Sale ID": "orders"})
        fig2 = px.line(vol, x="Month", y="orders", color=prod_col, title="Monthly Orders by Product Type (Top 12)")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "inventory_monthly_orders_by_product.html")
        st.markdown("**Insight:** Volume trends can differ from revenue trends.")
        st.markdown("**Recommendation:** Plan inventory using both order volume and value.")
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
        st.markdown("**Insight:** Ownership mix shapes value and risk management.")
        st.markdown("**Recommendation:** Use consignment to reduce cash risk; use owned inventory to improve availability and control.")
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
        st.markdown("**Recommendation:** Plan marketing and inventory before high months; use weak months for promotions and pipeline building.")
    else:
        st.write("-")

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
                st.markdown("**Insight:** Missing COA can still appear in meaningful value flow.")
                st.markdown("**Recommendation:** Enforce COA completion for high-value items and export-bound orders.")
            else:
                st.write("-")

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
                st.markdown("**Insight:** Export-related value is concentrated in specific markets.")
                st.markdown("**Recommendation:** Tighten documentation controls in top export markets to reduce compliance risk.")
            else:
                st.write("-")

with tabs[10]:
    st.subheader("Stats")
    st.markdown("### 1) Do channels differ in typical order value?")
    if sps is None:
        st.write("p-value: -")
        st.markdown("**Insight:** Statistical tests are disabled because SciPy is missing.")
        st.markdown("**Recommendation:** Add `scipy` to requirements.txt to enable p-values.")
    else:
        grp = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
        if len(grp) >= 2:
            _, p = sps.kruskal(*grp.tolist())
            st.write(f"p-value: **{p_fmt(p)}**")
            st.markdown("**Insight:** If p < 0.05, channels likely have different typical order values.")
            st.markdown("**Recommendation:** Prioritize channels with strong median value and strong total volume.")
        else:
            st.write("p-value: -")
            st.markdown("**Insight:** Not enough channel groups under current filters.")
            st.markdown("**Recommendation:** Expand date range or remove filters to increase sample size.")

    med = f.groupby("Channel")[metric].median().sort_values(ascending=False).reset_index().rename(columns={metric: "Median"})
    med["Median"] = med["Median"].round(0).map(lambda v: f"${v:,.0f} CAD")
    st.dataframe(med, use_container_width=True)

    st.markdown("### 2) Quick numeric drivers (correlation)")
    driver_candidates = ["Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Color Count (#)", "length", "width", "weight", lag_col]
    drivers = [c for c in driver_candidates if c in f.columns]
    rows = []
    for c in drivers:
        x = pd.to_numeric(f[c], errors="coerce")
        y = pd.to_numeric(f[metric], errors="coerce")
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
        st.markdown("**Recommendation:** Use the top 2–3 drivers as filters/KPIs and ignore weak drivers.")
    else:
        st.write("-")

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

        kpi = kpi.sort_values("orders", ascending=False).head(60)
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

        dq = pd.DataFrame([
            {"Check": "Rows in filtered view", "Value": f"{len(f):,}"},
            {"Check": "Missing Country", "Value": f"{int(f['Country'].isna().sum()):,}"},
            {"Check": "Missing City", "Value": f"{int(f['City'].isna().sum()):,}"},
            {"Check": "Missing Channel", "Value": f"{int(f['Channel'].isna().sum()):,}"},
            {"Check": "Missing Price (CAD)", "Value": f"{int(safe_series(f, 'Price (CAD)').isna().sum()):,}"},
            {"Check": "Missing Shipped Date", "Value": f"{int(safe_series(f, 'Shipped Date').isna().sum()):,}"},
        ])
        st.markdown("### Data Quality Snapshot")
        st.dataframe(dq, use_container_width=True)

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
