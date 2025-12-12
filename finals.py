import pathlib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ModuleNotFoundError:
    st.error("Missing dependency: plotly. Add `plotly` to requirements.txt and redeploy.")
    st.stop()

try:
    from scipy import stats as sps
except ModuleNotFoundError:
    sps = None

st.set_page_config(page_title="Week 10 - Full Analytics Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 0.90rem !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }
div[data-testid="column"] { padding-left: 0.40rem; padding-right: 0.40rem; }
[data-testid="metric-container"] { padding: 0.75rem 0.9rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
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

st.title("Lucentara & Dinosty Fossils - Week 10 Analytics Dashboard")
st.caption("Price Drivers • Product Mix • Customer Segments • Geography × Channels • Inventory Timing • Ownership • Seasonality • Compliance • Stats • $ CAD")

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

def pct(x, decimals=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x)*100:.{decimals}f}%"

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

def pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
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
    "Country", "City", "Channel", "Customer Type", "Product Type",
    "Lead Source", "Consignment? (Y/N)", "Species", "Grade", "Finish", "COA #", "Export Permit (PDF link)"
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
products = sorted([c for c in df["Product Type"].dropna().unique().tolist()]) if "Product Type" in df.columns else []
cust_types = sorted([c for c in df["Customer Type"].dropna().unique().tolist()]) if "Customer Type" in df.columns else []

sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])
sel_products = st.sidebar.multiselect("Product Types (optional)", products, default=[]) if products else []
sel_customers = st.sidebar.multiselect("Customer Types (optional)", cust_types, default=[]) if cust_types else []

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
sel_cities = st.sidebar.multiselect("Cities (optional)", cities, default=[])

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
    "Geography × Channels",
    "Time + Shipping",
    "Ownership",
    "Seasonality",
    "Compliance",
    "Stats",
    "Data"
])

with tabs[0]:
    st.subheader("Summary Insights")
    share_top = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
    lines = []
    if np.isfinite(share_top):
        lines.append(f"- {top_country} drives about {share_top*100:.1f}% of {metric}.")
    lines.append(f"- Top channel by {metric} is {top_channel}.")
    if np.isfinite(cons_rate):
        lines.append(f"- Consignment is {cons_rate:.1f}% of orders.")
    st.markdown("\n".join(lines) if lines else "-")

    st.subheader("Summary Recommendations")
    recs = [
        "- Protect and grow the top markets first, then scale the next tier.",
        "- Use different channel strategies per country (do not copy-paste one global mix).",
        "- Track shipping lag as an operations KPI and clean shipping date issues early.",
        "- Use the theme tabs to translate these patterns into business actions."
    ]
    st.markdown("\n".join(recs))

with tabs[1]:
    st.subheader("Price Drivers (Grade • Finish • Colour Count)")

    grade_col = pick_col(f, ["Grade"])
    finish_col = pick_col(f, ["Finish"])
    color_col = pick_col(f, ["Color Count (#)", "Color Count"])

    if grade_col:
        gdf = f.dropna(subset=[grade_col, "Price (CAD)"]).copy()
        fig = px.box(gdf, x=grade_col, y="Price (CAD)", points="outliers", title="Price distribution by Grade ($ CAD)")
        fig.update_traces(hovertemplate="Grade: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "price_by_grade.html")
        st.markdown("**Insight:** Some grades have higher typical prices and wider spreads.")
        st.markdown("**Recommendation:** Keep premium grades positioned as premium; avoid heavy discounting on high grades.")
    else:
        st.info("Grade column not found.")

    if color_col:
        cdf = f.dropna(subset=[color_col, "Price (CAD)"]).copy()
        fig2 = px.scatter(cdf, x=color_col, y="Price (CAD)", color=finish_col if finish_col else None,
                          title="Price vs Colour Count ($ CAD)")
        fig2.update_traces(hovertemplate="Colours: %{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "price_vs_colour_count.html")
        st.markdown("**Insight:** This shows whether more colours generally align with higher pricing.")
        st.markdown("**Recommendation:** If colour count is a strong driver, reflect it in pricing rules and product descriptions.")
    else:
        st.info("Colour Count column not found.")

with tabs[2]:
    st.subheader("Product Mix (Product Type / Species)")

    prod_col = pick_col(f, ["Product Type"])
    if prod_col:
        prod_tot = f.groupby(prod_col)[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
        top_prod = prod_tot.head(15)
        fig = px.bar(top_prod, x=prod_col, y="value", title=f"Top Product Types by {metric} ($ CAD)")
        fig.update_layout(xaxis={"categoryorder": "total descending"})
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "product_mix_top_types.html")
        st.markdown("**Insight:** A small number of product types drive most value.")
        st.markdown("**Recommendation:** Keep top product types consistently available and featured; treat minor types as experiments.")
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
        st.markdown("**Recommendation:** Double down marketing content around top species; ensure pricing is consistent with demand.")
    else:
        st.info("Species column not found.")

with tabs[3]:
    st.subheader("Customer Segments (Customer Type)")

    cust_col = pick_col(f, ["Customer Type"])
    if cust_col:
        seg = f.groupby(cust_col)[metric].sum().sort_values(ascending=False).reset_index().rename(columns={metric: "value"})
        fig = px.bar(seg, x=cust_col, y="value", title=f"{metric} by Customer Type ($ CAD)")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "customer_type_value.html")
        st.markdown("**Insight:** One or two customer types often generate the majority of value.")
        st.markdown("**Recommendation:** Create different offers and channel strategies for high-value vs low-value segments.")
    else:
        st.info("Customer Type column not found.")

with tabs[4]:
    st.subheader("Geography × Channels")

    agg = country_totals.reset_index().rename(columns={metric: "value"})
    agg["share"] = agg["value"] / agg["value"].sum()

    show_other = st.toggle("Add 'Other Countries' so shares sum to 100%", value=True)
    if show_other and len(agg) > 0:
        top_keep = agg.sort_values("value", ascending=False).head(15).copy()
        other_val = float(agg["value"].sum() - top_keep["value"].sum())
        if other_val > 0:
            other = pd.DataFrame([{"Country": "Other Countries", "value": other_val}])
            top_keep = pd.concat([top_keep[["Country", "value"]], other], ignore_index=True)
            top_keep["share"] = top_keep["value"] / top_keep["value"].sum()
            agg_show = top_keep.copy()
        else:
            agg_show = agg.copy()
    else:
        agg_show = agg.copy()

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
    st.markdown("**Insight:** Revenue is concentrated in a few top markets.")
    st.markdown("**Recommendation:** Protect the top markets first, then scale the next tier using the best-performing channels.")

    colA, colB = st.columns(2)

    with colA:
        top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "value"})
        fig1 = px.bar(top_c, x="Country", y="value", title=f"Top {top_n} Countries by {metric}")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig1 = fig_tight(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        download_html(fig1, "geo_top_countries.html")
        st.markdown("**Insight:** A small set of countries drives most of the value.")
        st.markdown("**Recommendation:** Use market tiers (Anchor / Growth / Test) and budget accordingly.")

    with colB:
        ch = channel_totals.reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(ch, x="Channel", y="value", title=f"{metric} by Channel ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "geo_channel_bar.html")
        st.markdown("**Insight:** Some channels generate much more value than others.")
        st.markdown("**Recommendation:** Put your best products and campaigns into the strongest channels first.")

    st.subheader("Country × Channel Heatmap (Top countries)")
    top_idx = country_totals.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
    fig3 = heatmap_from_pivot(pv, f"Heatmap: {metric} ($ CAD)", "$ CAD")
    st.plotly_chart(fig3, use_container_width=True)
    download_html(fig3, "geo_country_channel_heatmap.html")
    st.markdown("**Insight:** The best channel is not the same in every country.")
    st.markdown("**Recommendation:** Choose 1–2 winning channels per country instead of forcing one global mix.")

    if "Lead Source" in f.columns and f["Lead Source"].notna().any():
        st.subheader("Lead Source")
        ls = f.dropna(subset=["Lead Source"]).copy()
        col1, col2 = st.columns(2)

        with col1:
            ls_tot = ls.groupby("Lead Source")[metric].sum().sort_values(ascending=False).head(12).reset_index().rename(columns={metric: "value"})
            fig_ls = px.bar(ls_tot, x="Lead Source", y="value", title=f"{metric} by Lead Source (Top 12)")
            fig_ls.update_layout(xaxis={"categoryorder": "total descending"})
            fig_ls.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig_ls = fig_tight(fig_ls)
            st.plotly_chart(fig_ls, use_container_width=True)
            download_html(fig_ls, "geo_lead_source_top12.html")

        with col2:
            top_sources = ls_tot["Lead Source"].tolist()
            mix = ls[ls["Lead Source"].isin(top_sources)].groupby(["Lead Source", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
            mix["total"] = mix.groupby("Lead Source")["value"].transform("sum")
            mix["share"] = mix["value"] / mix["total"]
            fig_m = px.bar(mix, x="Lead Source", y="share", color="Channel", barmode="stack", title="Channel Mix inside Lead Sources (Top 12)")
            fig_m.update_layout(yaxis_tickformat=".0%")
            fig_m = fig_tight(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)
            download_html(fig_m, "geo_lead_source_channel_mix.html")

        st.markdown("**Insight:** A few lead sources drive most of the value, and they often have different channel mixes.")
        st.markdown("**Recommendation:** Allocate effort/budget to the highest-performing lead sources and optimize their strongest channels.")

    st.subheader("Top Markets Table")
    tbl = agg_show.sort_values("value", ascending=False).copy()
    tbl["Value ($ CAD)"] = tbl["value"].round(0).map(lambda v: f"${v:,.0f} CAD")
    tbl["Share (%)"] = (tbl["share"] * 100).round(2).map(lambda v: f"{v:.2f}%")
    show_tbl = tbl[["Country", "Value ($ CAD)", "Share (%)"]].copy()
    show_tbl = rank_df(show_tbl)
    st.dataframe(show_tbl.set_index("#"), use_container_width=True)

with tabs[5]:
    st.subheader("Time + Shipping")

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
        st.markdown("**Insight:** Not enough months to show trend changes.")
    st.markdown("**Recommendation:** When the trend shifts, drill into countries/channels/product types to identify the cause.")

    st.markdown("### Monthly Trend by Channel (Top 6)")
    ch_tot = f.groupby("Channel")[metric].sum().sort_values(ascending=False)
    top6 = ch_tot.head(6).index.tolist()
    by_ch = f[f["Channel"].isin(top6)].groupby(["Month", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
    figc = px.line(by_ch, x="Month", y="value", color="Channel", title=f"Monthly {metric} by Channel (Top 6) ($ CAD)")
    figc.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    figc = fig_tight(figc)
    st.plotly_chart(figc, use_container_width=True)
    download_html(figc, "time_monthly_trend_by_channel_top6.html")
    st.markdown("**Insight:** Different channels contribute differently over time.")
    st.markdown("**Recommendation:** Use this to identify which channel is driving growth or decline in specific months.")

    st.divider()
    st.subheader("Shipping Lag (Country + City)")

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
            st.markdown("**Insight:** Some countries have consistently slower shipping times.")
            st.markdown("**Recommendation:** Set country-level SLAs and adjust fulfillment strategy for the slowest destinations.")

            pick = st.selectbox("Country → City drilldown", sorted(lag_df["Country"].dropna().unique().tolist()))
            by_city = lag_df[lag_df["Country"] == pick].groupby("City")["Ship Lag (days)"].mean().sort_values(ascending=False).head(15).reset_index()
            fig2 = px.bar(by_city, x="City", y="Ship Lag (days)", title=f"Avg Ship Lag by City in {pick} (Top 15)")
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            fig2 = fig_tight(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "ship_lag_by_city.html")
            st.markdown("**Insight:** Within a country, delays are often concentrated in a few cities.")
            st.markdown("**Recommendation:** Fix the worst cities first to improve performance quickly.")

        with col2:
            min_orders = st.slider("Minimum orders per Country+City", 2, 15, 5)
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
            st.markdown("**Insight:** This lists the biggest delay hotspots with enough volume to trust.")
            st.markdown("**Recommendation:** Prioritize improvements where delays and volume are both high.")

            samp = lag_df.copy()
            if len(samp) > 2500:
                samp = samp.sample(2500, random_state=7)
            fig4 = px.scatter(samp, x="Ship Lag (days)", y=metric, color="Channel",
                              title=f"Ship Lag vs {metric} ($ CAD)", hover_data=["Country", "City"])
            fig4.update_traces(hovertemplate="Lag: %{x:.0f} days<br>Value: %{y:$,.0f} CAD<extra></extra>")
            fig4 = fig_tight(fig4)
            st.plotly_chart(fig4, use_container_width=True)
            download_html(fig4, "ship_lag_scatter.html")
            st.markdown("**Insight:** This checks whether higher value orders ship faster or slower (often weak).")
            st.markdown("**Recommendation:** Treat shipping lag as a customer experience KPI even if revenue impact is small.")

with tabs[6]:
    st.subheader("Ownership (Consigned vs Owned)")

    own_col = pick_col(f, ["Consignment? (Y/N)"])
    if own_col:
        odf = f.copy()
        odf["Ownership"] = odf[own_col].astype(str).str.upper().replace({"Y": "Consigned", "N": "Owned"})
        own_tot = odf.groupby("Ownership")[metric].sum().reset_index().rename(columns={metric: "value"})
        fig = px.bar(own_tot, x="Ownership", y="value", title=f"{metric} by Ownership ($ CAD)")
        fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
        fig = fig_tight(fig)
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "ownership_value.html")
        st.markdown("**Insight:** This shows how much value comes from consigned vs owned inventory.")
        st.markdown("**Recommendation:** Use ownership mix to balance cash risk and supply reliability.")

        box = odf.dropna(subset=["Ownership", "Price (CAD)"])
        fig2 = px.box(box, x="Ownership", y="Price (CAD)", points="outliers", title="Price distribution by Ownership ($ CAD)")
        fig2.update_traces(hovertemplate="%{x}<br>Price: %{y:$,.0f} CAD<extra></extra>")
        fig2 = fig_tight(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "ownership_price_box.html")
        st.markdown("**Insight:** Ownership may be linked to different typical price levels.")
        st.markdown("**Recommendation:** If consigned items are higher value, treat them as premium draws and manage lead time expectations.")
    else:
        st.info("Ownership/consignment column not found.")

with tabs[7]:
    st.subheader("Seasonality")

    ts_df = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "value"})
    fig = px.line(ts_df, x="Month", y="value", title=f"Monthly {metric} ($ CAD)")
    fig.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    fig = fig_tight(fig)
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "seasonality_monthly.html")
    st.markdown("**Insight:** This shows whether the business has seasonal peaks and dips.")
    st.markdown("**Recommendation:** Use peaks to plan inventory and staffing; use dips for promotions and pipeline building.")

with tabs[8]:
    st.subheader("Compliance (COA / Export)")

    coa_col = pick_col(f, ["COA #"])
    exp_col = pick_col(f, ["Export Permit (PDF link)"])

    if not coa_col and not exp_col:
        st.info("No compliance columns found (COA # / Export Permit link).")
    else:
        colA, colB = st.columns(2)

        with colA:
            if coa_col:
                cdf = f.copy()
                cdf["COA Status"] = np.where(cdf[coa_col].notna(), "COA Present", "COA Missing")
                coa = cdf.groupby("COA Status")[metric].sum().reset_index().rename(columns={metric: "value"})
                fig = px.bar(coa, x="COA Status", y="value", title=f"{metric} by COA Status ($ CAD)")
                fig.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
                fig = fig_tight(fig)
                st.plotly_chart(fig, use_container_width=True)
                download_html(fig, "compliance_coa.html")
                st.markdown("**Insight:** Missing COA appears in some portion of value flow.")
                st.markdown("**Recommendation:** Reduce missing COA on high-value orders to prevent downstream compliance risk.")
            else:
                st.info("COA # column not found.")

        with colB:
            if exp_col:
                edf = f.copy()
                edf["Export Permit"] = np.where(edf[exp_col].notna(), "Permit Present", "Permit Missing")
                exp = edf.groupby(["Export Permit", "Country"])[metric].sum().reset_index().rename(columns={metric: "value"})
                top_ctry = country_totals.head(12).index
                exp = exp[exp["Country"].isin(top_ctry)]
                fig2 = px.bar(exp, x="Country", y="value", color="Export Permit", barmode="group",
                              title=f"{metric} by Country and Export Permit (Top Countries)")
                fig2.update_traces(hovertemplate="%{x}<br>Value: %{y:$,.0f} CAD<extra></extra>")
                fig2 = fig_tight(fig2)
                st.plotly_chart(fig2, use_container_width=True)
                download_html(fig2, "compliance_export.html")
                st.markdown("**Insight:** Export documentation completeness varies by market.")
                st.markdown("**Recommendation:** For top export markets, enforce documentation checks before shipment.")
            else:
                st.info("Export Permit (PDF link) column not found.")

with tabs[9]:
    st.subheader("Stats (Short + Understandable)")

    st.markdown("### 1) Do channels differ on order value?")
    if sps is None:
        st.write("SciPy not installed, so statistical tests are disabled. Add `scipy` to requirements.txt to enable p-values.")
    else:
        grp = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
        if len(grp) >= 2:
            _, p = sps.kruskal(*grp.tolist())
            st.write(f"p-value: **{p_fmt(p)}** → " + ("Likely different typical values across channels." if p < 0.05 else "No strong evidence of a difference."))
        else:
            st.write("Not enough channel groups in the current filter.")

    med = f.groupby("Channel")[metric].median().sort_values(ascending=False).reset_index().rename(columns={metric: "Median"})
    med["Median"] = med["Median"].round(0).map(lambda v: f"${v:,.0f} CAD")
    st.dataframe(med, use_container_width=True)
    st.markdown("**Insight:** This compares typical (median) order value by channel.")
    st.markdown("**Recommendation:** Prioritize channels with strong median value AND strong total volume.")

    st.markdown("### 2) Is channel mix different by country?")
    top_for_test = country_totals.head(min(10, len(country_totals))).index
    tmp = f.copy()
    tmp["Country (top)"] = np.where(tmp["Country"].isin(top_for_test), tmp["Country"], "Other")
    ct = pd.crosstab(tmp["Country (top)"], tmp["Channel"])
    if sps is not None and ct.shape[0] >= 2 and ct.shape[1] >= 2:
        _, p2, _, _ = sps.chi2_contingency(ct)
        st.write(f"p-value: **{p_fmt(p2)}** → " + ("Mix differs by country." if p2 < 0.05 else "No strong evidence of different mixes."))
    else:
        st.write("Using a descriptive view (no test) based on current filters.")
    mix_show = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    st.dataframe(mix_show, use_container_width=True)
    st.markdown("**Insight:** Countries often have different channel profiles.")
    st.markdown("**Recommendation:** Build a country-specific channel plan instead of a single global approach.")

    st.markdown("### 3) Strongest numeric relationships")
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
        out["|r|"] = out["Spearman r"].abs()
        out = out.sort_values("|r|", ascending=False).drop(columns=["|r|"]).head(10).reset_index(drop=True)
        out["Spearman r"] = out["Spearman r"].round(3)
        out = rank_df(out)
        st.dataframe(out.set_index("#"), use_container_width=True)
        st.markdown("**Insight:** Bigger |r| means a stronger relationship with the chosen metric.")
        st.markdown("**Recommendation:** Use the top 2–3 drivers as dashboard filters or KPIs; avoid adding every numeric field.")
    else:
        st.write("Not enough numeric data to compute relationships with current filters.")

with tabs[10]:
    st.subheader("Data + Downloads")

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

        kpi = kpi.sort_values("orders", ascending=False).head(40)
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
            mime="text/csv"
        )

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
