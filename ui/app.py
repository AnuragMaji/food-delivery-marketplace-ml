"""Streamlit dashboard: the thing a recruiter/interviewer actually opens in a
browser. Reads live from Postgres and Redis — no separate data path from
what the rest of the pipeline writes.

Run locally:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import redis
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_MISSING = object()


def get_config(key: str, default=_MISSING) -> str:
    """Prefers Streamlit Cloud's secrets store when configured (production),
    falls back to .env/os.environ (local dev). st.secrets raises
    FileNotFoundError on any access at all when no secrets.toml exists
    (not just a missing key), so this must be a try/except, not a
    dict-style .get() with a default. Pass `default` for optional keys
    (e.g. GH_PAT) — omit it for required keys, which raise KeyError."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    if default is _MISSING:
        return os.environ[key]
    return os.environ.get(key, default)

REPO_ROOT = Path(__file__).parent.parent

GITHUB_REPO = "AnuragMaji/food-delivery-marketplace-ml"
GITHUB_WORKFLOW_FILE = "pipeline.yml"
GITHUB_BRANCH = "main"


def trigger_github_workflow() -> tuple[bool, str]:
    """Kicks off the same pipeline.yml GitHub Actions workflow the cron
    schedule uses (workflow_dispatch), rather than running the pipeline
    in-process here — this dashboard's host has no Redpanda broker and no
    kafka-python installed (ui/requirements.txt is deliberately slim), so
    the pipeline can only actually run inside that workflow's job."""
    token = get_config("GH_PAT", None)
    if not token:
        return False, "GH_PAT isn't configured in this app's secrets."
    try:
        resp = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"ref": GITHUB_BRANCH},
            timeout=10,
        )
    except requests.RequestException as e:
        return False, f"GitHub API call failed: {e}"
    if resp.status_code == 204:
        return True, ""
    return False, f"GitHub API returned {resp.status_code}: {resp.text[:300]}"

# Reference palette (dataviz skill) — sequential blue for magnitude, status
# colors reserved for the data-quality panel, never reused as series colors.
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
GRAY_MUTED = "#898781"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

st.set_page_config(page_title="Food Delivery Marketplace — Live Dashboard", layout="wide")

PLOTLY_LAYOUT = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, sans-serif", color=INK_PRIMARY, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)


@st.cache_resource
def get_pg_connection():
    return psycopg2.connect(get_config("POSTGRES_URL"))


@st.cache_resource
def get_redis_client():
    return redis.from_url(get_config("REDIS_URL"), decode_responses=True)


@st.cache_data(ttl=60)
def load_kpis() -> dict:
    conn = get_pg_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM fact_deliveries")
        total_deliveries = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM dim_users WHERE status = 'active'")
        active_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dim_merchants WHERE is_active")
        active_merchants = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dim_dashers WHERE status = 'active'")
        active_dashers = cur.fetchone()[0]

        cur.execute(
            """SELECT ROUND(100.0 * SUM(CASE WHEN is_late THEN 1 ELSE 0 END) / COUNT(*), 1)
               FROM fact_deliveries WHERE order_ts > now() - interval '30 days'"""
        )
        late_rate_30d = cur.fetchone()[0]

        cur.execute(
            """SELECT ROUND(AVG(ABS(fp.predicted_eta_min - fd.actual_delivery_min))::numeric, 2), COUNT(*)
               FROM fact_predictions fp JOIN fact_deliveries fd ON fd.delivery_id = fp.delivery_id
               WHERE fd.actual_delivery_min IS NOT NULL"""
        )
        mae, n_predictions = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM deliveries_quarantine")
        n_quarantined = cur.fetchone()[0]

    return {
        "total_deliveries": total_deliveries,
        "active_users": active_users,
        "active_merchants": active_merchants,
        "active_dashers": active_dashers,
        "late_rate_30d": float(late_rate_30d) if late_rate_30d is not None else None,
        "mae": float(mae) if mae is not None else None,
        "n_predictions": n_predictions,
        "n_quarantined": n_quarantined,
    }


@st.cache_data(ttl=60)
def load_growth_trend() -> pd.DataFrame:
    conn = get_pg_connection()
    query = """
        SELECT DATE_TRUNC('month', order_ts)::date AS month, COUNT(*) AS deliveries
        FROM fact_deliveries GROUP BY 1 ORDER BY 1
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_zone_demand(top_n: int = 15) -> pd.DataFrame:
    conn = get_pg_connection()
    query = f"""
        SELECT z.city, z.state, COUNT(*) AS deliveries
        FROM fact_deliveries fd JOIN dim_zones z ON z.zone_id = fd.zone_id
        WHERE fd.order_ts > now() - interval '30 days'
        GROUP BY z.city, z.state ORDER BY deliveries DESC LIMIT {top_n}
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    conn = get_pg_connection()
    query = """
        SELECT fp.predicted_eta_min, fd.actual_delivery_min, fp.scored_at
        FROM fact_predictions fp JOIN fact_deliveries fd ON fd.delivery_id = fp.delivery_id
        WHERE fd.actual_delivery_min IS NOT NULL
        ORDER BY fp.scored_at DESC LIMIT 2000
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_quarantine_breakdown() -> pd.DataFrame:
    conn = get_pg_connection()
    query = """
        SELECT reason, COUNT(*) AS n
        FROM deliveries_quarantine, jsonb_array_elements_text(failed_expectations) AS reason
        GROUP BY reason ORDER BY n DESC
    """
    return pd.read_sql(query, conn)


def load_live_metrics() -> dict:
    r = get_redis_client()
    return {
        "latest_batch_orders": r.get("metrics:latest_batch:total_orders"),
        "latest_batch_late_rate": r.get("metrics:latest_batch:late_rate_pct"),
        "latest_batch_avg_eta": r.get("metrics:latest_batch:avg_promised_eta_min"),
        "date_range_start": r.get("metrics:latest_batch:date_range_start"),
        "date_range_end": r.get("metrics:latest_batch:date_range_end"),
        "cumulative_orders_seen": r.get("metrics:orders_total"),
    }


@st.cache_data(ttl=60)
def load_data_overview() -> dict:
    conn = get_pg_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(order_ts)::date, MAX(order_ts)::date FROM fact_deliveries")
        min_date, max_date = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM dim_users")
        n_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dim_merchants")
        n_merchants = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dim_dashers")
        n_dashers = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT state) FROM dim_zones")
        n_zones, n_states = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM dim_promotions")
        n_promotions = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM fact_weather_daily")
        n_weather = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dim_external_events")
        n_events = cur.fetchone()[0]
    return {
        "min_date": min_date, "max_date": max_date, "days_span": (max_date - min_date).days,
        "n_users": n_users, "n_merchants": n_merchants, "n_dashers": n_dashers,
        "n_zones": n_zones, "n_states": n_states, "n_promotions": n_promotions,
        "n_weather": n_weather, "n_events": n_events,
    }


@st.cache_data(ttl=300)
def load_distribution_sample(sample_pct: float = 1.0) -> pd.DataFrame:
    """A ~1%-of-pages TABLESAMPLE — plenty for a distribution shape, far
    cheaper than scanning all 5M+ rows for a histogram."""
    conn = get_pg_connection()
    query = f"""
        SELECT subtotal, actual_delivery_min, item_count
        FROM fact_deliveries TABLESAMPLE SYSTEM ({sample_pct})
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_accuracy_over_time() -> pd.DataFrame:
    conn = get_pg_connection()
    query = """
        SELECT DATE_TRUNC('day', fp.scored_at)::date AS day,
               AVG(fp.predicted_eta_min - fd.actual_delivery_min) AS mean_residual,
               AVG(ABS(fp.predicted_eta_min - fd.actual_delivery_min)) AS mae,
               COUNT(*) AS n
        FROM fact_predictions fp JOIN fact_deliveries fd ON fd.delivery_id = fp.delivery_id
        WHERE fd.actual_delivery_min IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_error_by_segment(segment_key: str) -> pd.DataFrame:
    """segment_key: a field name inside fact_predictions.features_snapshot,
    e.g. 'weather_condition' or 'density_tier' — this works because the API
    stores the exact features it used for every prediction, so monitoring
    can slice error by any of them without re-deriving anything."""
    conn = get_pg_connection()
    query = f"""
        SELECT fp.features_snapshot->>'{segment_key}' AS segment,
               AVG(ABS(fp.predicted_eta_min - fd.actual_delivery_min)) AS mae,
               COUNT(*) AS n
        FROM fact_predictions fp JOIN fact_deliveries fd ON fd.delivery_id = fp.delivery_id
        WHERE fd.actual_delivery_min IS NOT NULL
        GROUP BY 1 ORDER BY mae DESC
    """
    return pd.read_sql(query, conn)


@st.cache_data(ttl=60)
def load_coverage() -> dict:
    conn = get_pg_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM fact_deliveries WHERE source = 'realtime'")
        n_realtime = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM fact_predictions")
        n_scored = cur.fetchone()[0]
    return {"n_realtime": n_realtime, "n_scored": n_scored}


def styled_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False, color=INK_SECONDARY)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False, color=INK_SECONDARY)
    return fig


# --- Header + sidebar trigger -------------------------------------------

st.title("Food Delivery Marketplace — Live Dashboard")
st.caption(
    "A simulated, DoorDash-style marketplace: synthetic historical + accelerated "
    "real-time data, a full data-quality/ML pipeline, and this dashboard reading "
    "straight from Postgres and Redis."
)

with st.sidebar:
    st.header("Pipeline control")
    st.write(
        "This simulated marketplace's clock moves faster than real time: each click "
        "below invents roughly **one week** of new simulated orders and runs them "
        "through the full pipeline (generate → publish → validate → clean → "
        "recompute features → score predictions), by triggering the same GitHub "
        "Actions workflow the scheduled cron uses. It takes about 1-2 minutes — "
        "refresh this page shortly after to see the new data."
    )
    if st.button("Generate next simulated week", type="primary"):
        triggered, message = trigger_github_workflow()
        if triggered:
            st.success("Triggered! Check back in a minute or two, then refresh.")
        else:
            st.error(f"Couldn't trigger the pipeline — {message}")

# --- KPI row --------------------------------------------------------------

kpis = load_kpis()
cols = st.columns(6)
cols[0].metric("Total deliveries", f"{kpis['total_deliveries']:,}")
cols[1].metric("Active users", f"{kpis['active_users']:,}")
cols[2].metric("Active merchants", f"{kpis['active_merchants']:,}")
cols[3].metric("Active dashers", f"{kpis['active_dashers']:,}")
cols[4].metric("Late rate (30d)", f"{kpis['late_rate_30d']:.1f}%" if kpis["late_rate_30d"] is not None else "—")
cols[5].metric(
    "Prediction MAE", f"{kpis['mae']:.1f} min" if kpis["mae"] is not None else "—",
    help=f"Mean absolute error across {kpis['n_predictions']} scored predictions",
)

# --- Most recent pipeline run (Redis) --------------------------------------

live = load_live_metrics()
if live["date_range_start"] and live["date_range_end"]:
    st.subheader(f"Most recent simulated week: {live['date_range_start']} to {live['date_range_end']}")
else:
    st.subheader("Most recent simulated week")
st.caption(
    "Each pipeline run advances the simulation by about one week of marketplace "
    "activity — the range above is simulated time, not when the run actually happened."
)
lcols = st.columns(4)
lcols[0].metric("New orders that week", live["latest_batch_orders"] or "—")
lcols[1].metric("Late rate that week", f"{live['latest_batch_late_rate']}%" if live["latest_batch_late_rate"] else "—")
lcols[2].metric("Avg promised ETA that week", f"{live['latest_batch_avg_eta']} min" if live["latest_batch_avg_eta"] else "—")
lcols[3].metric("Total orders processed to date", live["cumulative_orders_seen"] or "—")

st.divider()

# --- Growth trend -----------------------------------------------------------

st.subheader("Growth over time")
growth_df = load_growth_trend()
fig = px.line(growth_df, x="month", y="deliveries", markers=True)
fig.update_traces(line_color=BLUE, marker_color=BLUE, line_width=2)
fig.update_layout(yaxis_title="Deliveries per month", xaxis_title=None)
st.plotly_chart(styled_fig(fig), use_container_width=True)
st.caption(
    "Rapid growth through the launch phase, a peak, then market-maturity softening — "
    "see the README's Growth/churn section for how this shape was validated."
)

# --- Zone demand + prediction accuracy (side by side) -----------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top zones by demand (last 30 days)")
    zone_df = load_zone_demand()
    zone_df["label"] = zone_df["city"] + ", " + zone_df["state"]
    fig = px.bar(
        zone_df.sort_values("deliveries"), x="deliveries", y="label", orientation="h",
        color="deliveries", color_continuous_scale=BLUE_RAMP,
    )
    fig.update_layout(yaxis_title=None, xaxis_title="Deliveries", coloraxis_showscale=False)
    st.plotly_chart(styled_fig(fig), use_container_width=True)

with col_right:
    st.subheader("Predicted vs. actual ETA")
    pred_df = load_predictions()
    if len(pred_df) == 0:
        st.info("No predictions scored yet — click \"Generate next batch\" to produce some.")
    else:
        max_val = max(pred_df["predicted_eta_min"].max(), pred_df["actual_delivery_min"].max()) * 1.05
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df["actual_delivery_min"], y=pred_df["predicted_eta_min"],
            mode="markers", marker=dict(color=BLUE, size=8, opacity=0.6),
            name="Predictions",
        ))
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color=GRAY_MUTED, dash="dash", width=2),
            name="Perfect prediction", showlegend=True,
        ))
        fig.update_layout(xaxis_title="Actual delivery time (min)", yaxis_title="Predicted ETA (min)")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption(f"{len(pred_df):,} scored predictions shown (most recent 2,000).")

st.divider()

# --- Model monitoring ---------------------------------------------------

st.subheader("Model monitoring")
st.caption(
    "What we'd actually watch in production: is accuracy drifting over time, is the "
    "model systematically over- or under-predicting, does it perform worse in some "
    "conditions than others, and how much of the real traffic is even being scored."
)

coverage = load_coverage()
mcols = st.columns(3)
coverage_pct = 100.0 * coverage["n_scored"] / coverage["n_realtime"] if coverage["n_realtime"] else 0.0
mcols[0].metric(
    "Scoring coverage", f"{coverage_pct:.2f}%",
    help=f"{coverage['n_scored']:,} predictions scored out of {coverage['n_realtime']:,} realtime deliveries. "
         "Coverage is intentionally partial — each pipeline run samples up to 300 deliveries to score, "
         "not all of them, to keep runs fast.",
)

acc_df = load_accuracy_over_time()
if len(acc_df) > 0:
    latest_mae = acc_df["mae"].iloc[-1]
    latest_bias = acc_df["mean_residual"].iloc[-1]
    mcols[1].metric("Most recent day's MAE", f"{latest_mae:.1f} min")
    bias_label = "over-predicting" if latest_bias > 0 else "under-predicting"
    mcols[2].metric("Most recent day's bias", f"{abs(latest_bias):.1f} min {bias_label}")

mon_left, mon_right = st.columns(2)

with mon_left:
    st.markdown("**Accuracy over time**")
    if len(acc_df) < 2:
        st.info("Need predictions from at least two different days to show a trend — keep generating batches.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acc_df["day"], y=acc_df["mae"], mode="lines+markers",
            line=dict(color=BLUE, width=2), marker=dict(color=BLUE), name="MAE (min)",
        ))
        fig.update_layout(yaxis_title="Mean absolute error (min)", xaxis_title=None)
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    st.markdown("**Error by weather condition**")
    seg_df = load_error_by_segment("weather_condition")
    if len(seg_df) == 0:
        st.info("No scored predictions yet.")
    else:
        fig = px.bar(seg_df, x="mae", y="segment", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_title=None, xaxis_title="MAE (min)")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

with mon_right:
    st.markdown("**Residual distribution** (predicted − actual)")
    if len(pred_df) == 0:
        st.info("No scored predictions yet.")
    else:
        residuals = pred_df["predicted_eta_min"] - pred_df["actual_delivery_min"]
        fig = px.histogram(residuals, nbins=30, color_discrete_sequence=[BLUE])
        fig.add_vline(x=0, line_dash="dash", line_color=GRAY_MUTED)
        fig.update_layout(yaxis_title="Predictions", xaxis_title="Predicted − actual (min)", showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption(
            "Centered near zero and roughly symmetric is healthy; a shift to one side "
            "means the model is systematically over- or under-predicting."
        )

    st.markdown("**Error by zone density**")
    seg_df = load_error_by_segment("density_tier")
    if len(seg_df) == 0:
        st.info("No scored predictions yet.")
    else:
        fig = px.bar(seg_df, x="mae", y="segment", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_title=None, xaxis_title="MAE (min)")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

st.divider()

# --- Data overview / EDA -----------------------------------------------------

st.subheader("Data overview")
overview = load_data_overview()
st.caption(
    f"Historical + accelerated real-time data spans **{overview['min_date']} to "
    f"{overview['max_date']}** ({overview['days_span']:,} days)."
)

ocols = st.columns(6)
ocols[0].metric("Users (all-time)", f"{overview['n_users']:,}")
ocols[1].metric("Merchants (all-time)", f"{overview['n_merchants']:,}")
ocols[2].metric("Dashers (all-time)", f"{overview['n_dashers']:,}")
ocols[3].metric("Zones", f"{overview['n_zones']} ({overview['n_states']} states)")
ocols[4].metric("Promotions", f"{overview['n_promotions']:,}")
ocols[5].metric("Weather days logged", f"{overview['n_weather']:,}")

st.markdown("**Distribution shape** (1%-sample of all deliveries, for speed)")
dist_df = load_distribution_sample()
dcols = st.columns(3)
with dcols[0]:
    fig = px.histogram(dist_df, x="subtotal", nbins=40, color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title="Order subtotal ($)", yaxis_title="Deliveries", showlegend=False)
    st.plotly_chart(styled_fig(fig), use_container_width=True)
with dcols[1]:
    fig = px.histogram(dist_df, x="actual_delivery_min", nbins=40, color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title="Delivery time (min)", yaxis_title="Deliveries", showlegend=False)
    st.plotly_chart(styled_fig(fig), use_container_width=True)
with dcols[2]:
    fig = px.histogram(dist_df, x="item_count", nbins=int(dist_df["item_count"].max()), color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title="Items per order", yaxis_title="Deliveries", showlegend=False)
    st.plotly_chart(styled_fig(fig), use_container_width=True)

st.divider()

# --- Data quality panel -----------------------------------------------------

st.subheader("Data quality")
total_seen = kpis["total_deliveries"] + kpis["n_quarantined"]
pass_rate = 100.0 * kpis["total_deliveries"] / total_seen if total_seen else 100.0

dq_cols = st.columns([1, 2])
with dq_cols[0]:
    status_color = STATUS_GOOD if pass_rate >= 99 else STATUS_WARNING if pass_rate >= 95 else STATUS_CRITICAL
    st.markdown(
        f"<div style='padding:1rem;border-radius:0.5rem;background:{SURFACE};border:1px solid {GRIDLINE}'>"
        f"<span style='color:{INK_SECONDARY};font-size:0.85rem'>Contract pass rate</span><br/>"
        f"<span style='color:{status_color};font-size:2rem;font-weight:600'>{pass_rate:.2f}%</span><br/>"
        f"<span style='color:{INK_SECONDARY};font-size:0.85rem'>{kpis['n_quarantined']:,} rows quarantined "
        f"out of {total_seen:,} seen by the Great Expectations gate</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with dq_cols[1]:
    quarantine_df = load_quarantine_breakdown()
    if len(quarantine_df) == 0:
        st.success("No quarantined rows — every batch has passed the data contract gate so far.")
    else:
        fig = px.bar(quarantine_df, x="n", y="reason", orientation="h", color_discrete_sequence=[STATUS_CRITICAL])
        fig.update_layout(yaxis_title=None, xaxis_title="Rows quarantined")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
