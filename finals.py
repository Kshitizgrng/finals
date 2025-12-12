import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

st.set_page_config(page_title="Week 10 - Full Analytics Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 0.90rem !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }
div[data-testid="column"] { padding-left: 0.40rem; padding-right: 0.40rem; }
[data-testid="metric-container"] { padding: 0.75rem 0.9rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
    line-height: 1.15 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
.js-plotly-plot .plot-container { width: 100% !important; }
[data-testid="stDataFrame"] { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.title("Lucentara & Dinosty Fossils - Week 10 Analytics")
st.caption("All groups in one place: Price drivers • Product mix • Customer segments • Geography × Channels • Inventory timing • Ownership • Seasonality • Compliance • Stats • $ CAD")

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

def download_html(fig: go.Figure, filename: str):
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button(
        label="Download HTML",
        data=html,
        file_name=filename,
        mime="text/html",
        key=f"dl_{filename}"
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

def fig_tight(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
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

text_cols = ["Country", "City", "Channel", "Customer Type", "Product Type", "Lead Source", "Consignment? (Y/N)"]
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
df["Total Collected (CAD)"] = (df["Net Sales (CAD)"] + df["Shipping (CAD)"].fillna(0) + df["Taxes Collected (CAD)"].fillna(0)).clip(lower=0)
df["Discount Rate"] = np.where(df["Price (CAD)"] > 0, df["Discount (CAD)"] / df["Price (CAD)"], np.nan)
df["Ship Lag Raw (days)"] = (df["Shipped Date"] - df["Date"]).dt.days
df["Ship Lag Clean (days)"] = np.where(df["Ship Lag Raw (days)"] >= 0, df["Ship Lag Raw (days)"], np.nan)
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

st.sidebar.header("Filters")
min_d = df["Date"].min()
max_d = df["Date"].max()
if pd.isna(min_d) or pd.isna(max_d):
    st.error("Date column could not be parsed.")
    st.stop()

dr = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
if not isinstance(dr, tuple):
    dr = (dr, dr)
start = pd.to_datetime(dr[0])
end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

metric = st.sidebar.selectbox("Metric ($ CAD)", ["Total Collected (CAD)", "Net Sales (CAD)", "Price (CAD)"], index=0)
exclude_negative_lag = st.sidebar.toggle("Exclude negative ship lag", value=True)
top_n = st.sidebar.slider("Top N (countries)", 5, 30, 12)

countries = sorted([c for c in df["Country"].dropna().unique().tolist() if c])
channels = sorted([c for c in df["Channel"].dropna().unique().tolist() if c])
sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])

base = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
if sel_countries:
    base = base[base["Country"].isin(sel_countries)]
if sel_channels:
    base = base[base["Channel"].isin(sel_channels)]

cities = sorted([c for c in base["City"].dropna().unique().tolist() if c])
sel_cities = st.sidebar.multiselect("Cities (optional)", cities, default=[])

f = base.copy()
if sel_cities:
    f = f[f["City"].isin(sel_cities)]

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

lag_col = "Ship Lag Clean (days)" if exclude_negative_lag else "Ship Lag Raw (days)"

total = float(f[metric].sum())
orders = int(len(f))
aov = float(f[metric].mean())
median_val = float(f[metric].median())

country_totals = f.groupby("Country")[metric].sum().sort_values(ascending=False)
channel_totals = f.groupby("Channel")[metric].sum().sort_values(ascending=False)

top_country = country_totals.index[0] if len(country_totals) else "-"
top_channel = channel_totals.index[0] if len(channel_totals) else "-"

cons_rate = float((f["Consignment? (Y/N)"].astype(str).str.upper().eq("Y").mean()) * 100) if "Consignment? (Y/N)" in f.columns else np.nan
neg_lag_rows = int((f["Ship Lag Raw (days)"] < 0).sum())
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
    "Geography × Channels",
    "Inventory Timing",
    "Ownership",
    "Seasonality",
    "Compliance",
    "Stats",
    "Data"
])

with tabs[0]:
    st.subheader("High-level insights")
    share_top = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
    bullets = []
    if np.isfinite(share_top):
        bullets.append(f"- {top_country} is the biggest market and drives about {share_top*100:.1f}% of {metric}.")
    bullets.append(f"- The top channel by {metric} is {top_channel}.")
    if np.isfinite(cons_rate):
        bullets.append(f"- Consignment is {cons_rate:.1f}% of orders.")
    if neg_lag_rows > 0:
        bullets.append(f"- There are {neg_lag_rows} rows where shipped date is before sale date (treated as missing in lag charts).")
    st.markdown("\n".join(bullets) if bullets else "-")

    st.subheader("Recommendations")
    recs = []
    if np.isfinite(share_top) and share_top >= 0.5:
        recs.append("Focus on protecting and growing the anchor market first, then scale the next 2–3 countries.")
    else:
        recs.append("Group countries into anchor, growth, and test tiers and set different expectations per tier.")
    if neg_lag_rows > 0:
        recs.append("Clean impossible shipping dates so operational KPIs are trustworthy.")
    recs.append("Use the theme tabs to go deeper on drivers, mix, segments, geography, timing, ownership, and compliance.")
    st.markdown("\n".join(["- " + r for r in recs]))

with tabs[1]:
    st.subheader("Price drivers (Grade • Colour count • Finish)")

    price_col = "Price (CAD)" if "Price (CAD)" in f.columns else metric
    grade_col = pick_col(f, ["Grade"])
    color_col = pick_col(f, ["Color Count (#)", "Color Count"])
    finish_col = pick_col(f, ["Finish"])

    if grade_col and price_col in f.columns:
        gdf = f.dropna(subset=[grade_col, price_col]).copy()
        fig = px.box(gdf, x=grade_col, y=price_col, points="outliers",
                     title="Price distribution by grade ($ CAD)")
        fig.update_traces(hovertemplate="Grade: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "price_by_grade_box.html")
        st.markdown("Insight: Higher grades tend to show higher typical prices, with some very high outliers.")
        st.markdown("Recommendation: Keep high-grade items positioned as premium and avoid discounting them too heavily.")
    else:
        st.info("No grade column found in the dataset.")

    if color_col and price_col in f.columns:
        cdf = f.dropna(subset=[color_col, price_col]).copy()
        if finish_col and finish_col in cdf.columns:
            fig2 = px.scatter(cdf, x=color_col, y=price_col, color=finish_col,
                              title="Price vs colour count by finish",
                              trendline="ols")
        else:
            fig2 = px.scatter(cdf, x=color_col, y=price_col,
                              title="Price vs colour count",
                              trendline="ols")
        fig2.update_traces(hovertemplate="Colour count: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "price_vs_colour_count.html")
        st.markdown("Insight: The trendline shows whether adding more colours tends to increase or decrease price.")
        st.markdown("Recommendation: Use this to decide whether higher colour counts justify a price premium.")
    else:
        st.info("No colour count column found in the dataset.")

with tabs[2]:
    st.subheader("Product mix (what we sell)")

    prod_col = pick_col(f, ["Product Type", "Product", "Item Type"])
    if prod_col:
        prod_tot = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).reset_index()
        top_prod = prod_tot.head(15)
        fig = px.bar(top_prod, x=prod_col, y=metric, title=f"Top product types by {metric} ($ CAD)")
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "product_mix_bar.html")
        st.markdown("Insight: A small number of product types drive most of the value.")
        st.markdown("Recommendation: Keep the top types well stocked and featured, and treat small types as experimental.")

        if "Channel" in f.columns:
            pm = f.groupby([prod_col, "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            pm["product_total"] = pm.groupby(prod_col)["value"].transform("sum")
            pm["share"] = pm["value"] / pm["product_total"]
            top_for_mix = prod_tot.head(8)[prod_col]
            pm = pm[pm[prod_col].isin(top_for_mix)]
            fig2 = px.bar(pm, x=prod_col, y="share", color="Channel", barmode="stack",
                          title="Channel mix inside each top product type")
            fig2.update_layout(yaxis_tickformat=".0%")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "product_mix_channel_share.html")
            st.markdown("Insight: Some product types are mostly sold online, others via partners or specific channels.")
            st.markdown("Recommendation: Match the right channel strategy to each product family instead of treating them all the same.")
    else:
        st.info("No product type column found in the dataset.")

with tabs[3]:
    st.subheader("Customer segments")

    cust_col = pick_col(f, ["Customer Type", "Customer Segment", "Segment"])
    if cust_col:
        seg_tot = f.groupby(cust_col)[metric].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(seg_tot, x=cust_col, y=metric, title=f"{metric} by customer segment ($ CAD)")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "segment_bar.html")
        st.markdown("Insight: One or two segments usually account for most of the value (often wholesale vs retail).")
        st.markdown("Recommendation: Build separate plans for high-value and low-value segments.")

        if "Channel" in f.columns:
            seg_mix = f.groupby([cust_col, "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            seg_mix["seg_total"] = seg_mix.groupby(cust_col)["value"].transform("sum")
            seg_mix["share"] = seg_mix["value"] / seg_mix["seg_total"]
            fig2 = px.bar(seg_mix, x=cust_col, y="share", color="Channel", barmode="stack",
                          title="Channel mix by customer segment")
            fig2.update_layout(yaxis_tickformat=".0%")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "segment_channel_mix.html")
            st.markdown("Insight: Different segments prefer different channels.")
            st.markdown("Recommendation: Use this to decide which channels to emphasize for each customer type.")
    else:
        st.info("No customer segment column found in the dataset.")

with tabs[4]:
    st.subheader("Geography × Channels")

    agg = country_totals.reset_index().rename(columns={metric: "value"})
    agg["share"] = agg["value"] / agg["value"].sum()

    fig = px.choropleth(
        agg,
        locations="Country",
        locationmode="country names",
        color="value",
        hover_name="Country",
        custom_data=["share"],
        projection="natural earth"
    )
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>Value: %{z:$,.0f} CAD<br>Share: %{customdata[0]:.1%}<extra></extra>"
    )
    fig = fig_tight(fig)
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "world_map.html")
    st.markdown("Insight: Revenue is concentrated in a few countries (darker areas on the map).")
    st.markdown("Recommendation: Focus spend and inventory on the top markets first, then expand to the next tier.")

    colA, colB = st.columns(2)

    with colA:
        st.subheader(f"Top countries by {metric}")
        top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "value"})
        fig1 = px.bar(top_c, x="Country", y="value", title=f"Top {top_n} countries ({metric})")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig1 = fig_tight(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        download_html(fig1, "top_countries.html")
        st.markdown("Insight: A handful of countries drive most of total value.")
        st.markdown("Recommendation: Treat smaller countries as experiments and scale only the ones that show repeatable demand.")

    with colB:
        st.subheader(f"{metric} by channel")
        ch = channel_totals.reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(ch, x="Channel", y="value", title=f"{metric} by channel ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "channel_bar.html")
        st.markdown("Insight: Some channels clearly outperform others on value.")
        st.markdown("Recommendation: Put your best products and marketing into the strongest channels first.")

    st.subheader("Country × Channel heatmap (top countries)")

    top_idx = country_totals.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    if not df_top.empty:
        pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
        fig3 = heatmap_from_pivot(pv, f"Heatmap: {metric} ($ CAD)", "$ CAD")
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "country_channel_heatmap.html")
        st.markdown("Insight: The best channel is not the same in every country.")
        st.markdown("Recommendation: Choose one or two winning channels per country instead of using the same mix everywhere.")
    else:
        st.info("No data available for the selected top countries.")

with tabs[5]:
    st.subheader("Inventory timing (shipping lag)")

    lag_df = f.dropna(subset=[lag_col]).copy()
    if lag_df.empty:
        st.info("No usable shipping lag values after filters.")
    else:
        lag_df["Ship Lag (days)"] = lag_df[lag_col].astype(float)
        col1, col2 = st.columns(2)

        with col1:
            by_country = (lag_df.groupby("Country")["Ship Lag (days)"]
                          .mean().sort_values(ascending=False).head(20).reset_index())
            fig1 = px.bar(by_country, x="Country", y="Ship Lag (days)", title="Avg ship lag by country (days)")
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            fig1 = fig_tight(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            download_html(fig1, "ship_lag_by_country.html")
            st.markdown("Insight: Some countries consistently ship slower than others.")
            st.markdown("Recommendation: Set country-level SLAs and adjust carrier or fulfillment strategy for the slowest destinations.")

            pick = st.selectbox("Country → city drilldown", sorted(lag_df["Country"].unique().tolist()))
            by_city = (lag_df[lag_df["Country"] == pick]
                       .groupby("City")["Ship Lag (days)"].mean()
                       .sort_values(ascending=False).head(15).reset_index())
            fig2 = px.bar(by_city, x="City", y="Ship Lag (days)", title=f"Avg ship lag by city in {pick} (top 15)")
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "ship_lag_by_city.html")
            st.markdown("Insight: Within a country, delays are usually concentrated in a few cities.")
            st.markdown("Recommendation: Fix the worst cities first instead of changing the whole country plan.")

        with col2:
            min_orders = st.slider("Minimum orders per country+city", 2, 15, 5)

            cc = (lag_df.groupby(["Country", "City"]).agg(
                orders=("Sale ID", "count"),
                avg_lag=("Ship Lag (days)", "mean"),
                med_lag=("Ship Lag (days)", "median"),
                total_metric=(metric, "sum")
            ).reset_index())
            cc = cc[cc["orders"] >= min_orders].copy()
            cc = cc.sort_values(["avg_lag", "orders"], ascending=[False, False]).head(25)
            cc["total_metric"] = cc["total_metric"].round(0)
            cc["avg_lag"] = cc["avg_lag"].round(1)
            cc["med_lag"] = cc["med_lag"].round(1)
            cc = rank_df(cc).rename(columns={"total_metric": f"Total ({metric})"})
            st.dataframe(cc.set_index("#")[["Country", "City", "orders", "avg_lag", "med_lag", f"Total ({metric})"]], use_container_width=True)
            st.markdown("Insight: This table shows the biggest delay hotspots that also have enough volume to matter.")
            st.markdown("Recommendation: Prioritize hotspots with both high delay and meaningful order volume.")

            top_countries = (lag_df.groupby("Country")[metric].sum().sort_values(ascending=False).head(12).index)
            sub = lag_df[lag_df["Country"].isin(top_countries)].copy()
            top_cities = (sub.groupby("City")[metric].sum().sort_values(ascending=False).head(20).index)
            sub = sub[sub["City"].isin(top_cities)].copy()

            pv2 = sub.pivot_table(values="Ship Lag (days)", index="Country", columns="City", aggfunc="mean")
            fig3 = heatmap_from_pivot(pv2, "Avg ship lag heatmap (country × city)", "days")
            st.plotly_chart(fig3, use_container_width=True)
            download_html(fig3, "ship_lag_heatmap_country_city.html")
            st.markdown("Insight: This heatmap highlights where delays cluster across top countries and cities.")
            st.markdown("Recommendation: Choose two or three country–city routes to optimize first.")

with tabs[6]:
    st.subheader("Ownership (consigned vs owned)")

    own_col = pick_col(f, ["Consignment? (Y/N)", "Consigned? (Y/N)", "Ownership"])
    if own_col:
        odf = f.copy()
        odf["Ownership"] = odf[own_col].astype(str).str.upper().replace({"Y": "Consigned", "N": "Owned"})
        own_tot = odf.groupby("Ownership")[metric].sum().reset_index()
        fig = px.bar(own_tot, x="Ownership", y=metric, title=f"{metric} by ownership")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "ownership_bar.html")
        st.markdown("Insight: This shows how much value comes from consigned vs owned pieces.")
        st.markdown("Recommendation: Balance risk and margin by choosing the right mix of consigned inventory.")

        if "Price (CAD)" in odf.columns:
            odf2 = odf.dropna(subset=["Ownership", "Price (CAD)"])
            fig2 = px.box(odf2, x="Ownership", y="Price (CAD)", title="Price distribution by ownership")
            fig2.update_traces(hovertemplate="%{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "ownership_price_box.html")
            st.markdown("Insight: This compares typical price levels for consigned vs owned pieces.")
            st.markdown("Recommendation: If consigned items skew higher value, use them to attract premium customers while managing cash risk.")
    else:
        st.info("No consignment or ownership column found in the dataset.")

with tabs[7]:
    st.subheader("Seasonality")

    ts_df = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "value"})
    fig = px.line(ts_df, x="Month", y="value", title=f"Monthly {metric} ($ CAD)")
    fig.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    fig = fig_tight(fig)
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "monthly_trend.html")

    if len(ts_df) >= 2:
        last_val = float(ts_df["value"].iloc[-1])
        prev_val = float(ts_df["value"].iloc[-2])
        mom = (last_val / prev_val - 1) if prev_val else np.nan
        if np.isfinite(mom):
            st.markdown(f"Insight: Latest month changed by {mom*100:.1f}% vs previous month.")
        else:
            st.markdown("Insight: Monthly movement is visible, but the latest month cannot be compared cleanly to the previous one.")
    else:
        st.markdown("Insight: Not enough months in the filtered data to show a trend.")

    st.markdown("Recommendation: Use this chart to spot seasonal spikes and dips, then drill into products, channels, and countries driving the changes.")

    ch_tot = f.groupby("Channel")[metric].sum().sort_values(ascending=False)
    top6 = ch_tot.head(6).index.tolist()
    by_ch = f[f["Channel"].isin(top6)].groupby(["Month", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
    figc = px.line(by_ch, x="Month", y="value", color="Channel", title=f"Monthly {metric} by channel (top 6) ($ CAD)")
    figc.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    figc = fig_tight(figc)
    st.plotly_chart(figc, use_container_width=True)
    download_html(figc, "monthly_trend_by_channel_top6.html")

    if len(top6) > 0:
        st.markdown(f"Insight: The top channels in this date range are: {', '.join(top6)}.")
    st.markdown("Recommendation: When the overall line moves, use this chart to identify which channel is driving the change.")

with tabs[8]:
    st.subheader("Compliance (COA / Export)")

    coa_col = pick_col(f, ["COA", "COA Provided", "COA Provided? (Y/N)", "Certificate of Authenticity"])
    export_col = pick_col(f, ["Export", "Export? (Y/N)", "Export Flag", "Exported"])

    if coa_col or export_col:
        if coa_col:
            cdf = f.copy()
            cdf["COA Flag"] = cdf[coa_col].astype(str).str.upper().replace({"Y": "COA OK", "N": "Missing COA"})
            coa_tot = cdf.groupby("COA Flag")[metric].sum().reset_index()
            fig = px.bar(coa_tot, x="COA Flag", y=metric, title=f"{metric} by COA status")
            fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig = fig_tight(fig)
            st.plotly_chart(fig, use_container_width=True)
            download_html(fig, "coa_bar.html")
            st.markdown("Insight: This shows how much value flows through fully documented vs missing-COA orders.")
            st.markdown("Recommendation: Reduce missing-COA orders, especially in high-value markets.")

        if export_col and "Country" in f.columns:
            edf = f.copy()
            edf["Export Flag"] = edf[export_col].astype(str).str.upper().replace({"Y": "Export", "N": "Domestic"})
            exp = edf.groupby(["Export Flag", "Country"])[metric].sum().reset_index().rename(columns={metric: "value"})
            fig2 = px.bar(exp, x="Country", y="value", color="Export Flag", barmode="group",
                          title=f"{metric} by country and export flag")
            fig2.update_traces(hovertemplate="%{x} (%{legendgroup})<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "export_by_country.html")
            st.markdown("Insight: This compares domestic vs export value by country.")
            st.markdown("Recommendation: For high-export countries, double check documentation and compliance controls.")
    else:
        st.info("No obvious COA or export columns found in the dataset. Compliance analysis is not available for this file.")

with tabs[9]:
    st.subheader("Stats")

    st.markdown("1) Do channels differ on order value?")
    grp = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
    if len(grp) >= 2:
        _, p = stats.kruskal(*grp.tolist())
        st.write(f"p-value: {p_fmt(p)} → " + ("Yes, typical order values differ across channels." if p < 0.05 else "No strong evidence of difference."))
    else:
        st.write("Not enough data for this test with current filters.")

    st.markdown("Insight: This tells you whether the channel differences you see in charts are likely real.")
    st.markdown("Recommendation: If significant, prioritize channels with the best combination of median value and volume.")

    st.markdown("2) Is channel mix different by country?")
    top_for_test = country_totals.head(min(10, len(country_totals))).index
    tmp = f.copy()
    tmp["Country (top)"] = np.where(tmp["Country"].isin(top_for_test), tmp["Country"], "Other")
    ct = pd.crosstab(tmp["Country (top)"], tmp["Channel"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        _, p2, _, _ = stats.chi2_contingency(ct)
        st.write(f"p-value: {p_fmt(p2)} → " + ("Yes, channel mix differs by country." if p2 < 0.05 else "No strong evidence of different mixes."))
    else:
        st.write("Not enough data for this test with current filters.")

    st.markdown("Insight: This supports a market-specific channel strategy instead of a one-size-fits-all mix.")
    st.markdown("Recommendation: Use the geography and segment tabs to decide which channel to emphasize in each market.")

    st.markdown("3) Strongest numeric relationships (Spearman)")
    driver_candidates = [
        "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)",
        "Color Count (#)", "length", "width", "weight", lag_col
    ]
    drivers = [c for c in driver_candidates if c in f.columns]
    rows = []
    for c in drivers:
        x = f[c]
        y = f[metric]
        ok = x.notna() & y.notna()
        if ok.sum() >= 30:
            r, pv = stats.spearmanr(x[ok], y[ok])
            rows.append((c, float(r), float(pv), int(ok.sum())))
    if rows:
        out = pd.DataFrame(rows, columns=["variable", "spearman_r", "p_value", "n"])
        out["abs_r"] = out["spearman_r"].abs()
        out = out.sort_values("abs_r", ascending=False).drop(columns=["abs_r"]).head(10).reset_index(drop=True)
        out["spearman_r"] = out["spearman_r"].round(3)
        out["p_value"] = out["p_value"].apply(lambda v: "<0.0001" if float(v) < 1e-4 else f"{float(v):.4f}")
        out = rank_df(out)
        st.dataframe(out.set_index("#"), use_container_width=True)
    else:
        st.write("Not enough data for this test with current filters.")

    st.markdown("Insight: Bigger absolute correlation means a stronger relationship; positive means they rise together, negative means they move in opposite directions.")
    st.markdown("Recommendation: Use the top 2–3 numeric drivers as dashboard filters or KPIs and avoid overloading the dashboard.")

with tabs[10]:
    st.subheader("Clean tables + download")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("Top countries")
        t = country_totals.reset_index().rename(columns={metric: "Total ($ CAD)"}).head(25)
        t["Total ($ CAD)"] = t["Total ($ CAD)"].round(0)
        st.dataframe(rank_df(t).set_index("#"), use_container_width=True)

        st.markdown("Top cities (country + city)")
        cct = f.groupby(["Country", "City"])[metric].sum().sort_values(ascending=False).head(25).reset_index().rename(columns={metric: "Total ($ CAD)"})
        cct["Total ($ CAD)"] = cct["Total ($ CAD)"].round(0)
        st.dataframe(rank_df(cct).set_index("#"), use_container_width=True)

    with colB:
        st.markdown("Country × Channel KPI")
        kpi = f.groupby(["Country", "Channel"]).agg(
            orders=("Sale ID", "count"),
            total=(metric, "sum"),
            avg=(metric, "mean"),
            median=(metric, "median"),
            avg_ship_lag=(lag_col, "mean"),
            avg_discount_rate=("Discount Rate", "mean")
        ).reset_index()

        kpi["total"] = kpi["total"].round(0)
        kpi["avg"] = kpi["avg"].round(0)
        kpi["median"] = kpi["median"].round(0)
        kpi["avg_ship_lag"] = kpi["avg_ship_lag"].round(1)
        kpi["avg_discount_rate"] = (kpi["avg_discount_rate"] * 100).round(1)

        kpi = kpi.sort_values("total", ascending=False).head(40)
        kpi = rank_df(kpi).rename(columns={
            "total": "Total ($ CAD)",
            "avg": "Avg ($ CAD)",
            "median": "Median ($ CAD)",
            "avg_ship_lag": "Avg ship lag (days)",
            "avg_discount_rate": "Avg discount (%)"
        })
        st.dataframe(kpi.set_index("#"), use_container_width=True)

        st.download_button(
            "Download filtered data (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
