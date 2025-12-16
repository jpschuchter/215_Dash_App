from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# 0) Environment + database config
# ============================================================
load_dotenv()

PGHOST = os.getenv("PGHOST", "")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGSCHEMA = os.getenv("PGSCHEMA", "public")
PGUSER = os.getenv("PGUSER", "")
PGPASSWORD = os.getenv("PGPASSWORD", "")
DASH_DEBUG = os.getenv("DASH_DEBUG", "1") == "1"

if not (PGHOST and PGUSER and PGPASSWORD):
    raise RuntimeError(
        "Missing PGHOST/PGUSER/PGPASSWORD. Put them in .env (recommended) or environment variables."
    )

def qname(table: str) -> str:
    return f'"{PGSCHEMA}"."{table}"'

def make_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=PGUSER,
        password=PGPASSWORD,
        host=PGHOST,
        port=PGPORT,
        database=PGDATABASE,
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"},
    )


ENGINE = make_engine()


def read_sql_df(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


# ============================================================
# 1) Plotly template + shared graph config
# ============================================================
jp_template = go.layout.Template(
    layout=dict(
        margin=dict(t=60, b=60, l=70, r=40),
        title=dict(x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="lightgray"),
        yaxis=dict(showgrid=False, gridcolor="lightgray", zeroline=True, zerolinecolor="lightgray"),
    )
)
pio.templates["jp_clean"] = jp_template
pio.templates.default = "jp_clean"

_GRAPH_CONFIG = {"displayModeBar": True, "displaylogo": False, "responsive": True}


def _warn_fig(msg: str, title: str = "Not available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(title=title, template="jp_clean", height=360)
    return fig


def _fmt_usd(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def _fmt_int(x: float) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def _fmt_usd2(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"


# ============================================================
# 2) Filter construction (sales + inventory + orders)
# ============================================================
def build_sales_filters(
    start_date: str,
    end_date: str,
    member_level: str,
    region_segment: str,
    ip_name: str,
    rarity: str,
    limited_mode: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Filters for queries driven by sales_fact.
    limited_mode: "All" | "Limited only" | "Non-limited only"
    """
    clauses = ["d.full_date >= :start_date", "d.full_date <= :end_date"]
    params: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if member_level != "All":
        clauses.append("u.member_level = :member_level")
        params["member_level"] = member_level

    if region_segment != "All":
        clauses.append("r.region_segment = :region_segment")
        params["region_segment"] = region_segment

    if ip_name != "All":
        clauses.append("ip.ip_name = :ip_name")
        params["ip_name"] = ip_name

    if rarity != "All":
        clauses.append("p.rarity = :rarity")
        params["rarity"] = rarity

    if limited_mode == "Limited only":
        clauses.append("sf.is_limited_edition = 'Y'")
    elif limited_mode == "Non-limited only":
        clauses.append("(sf.is_limited_edition IS NULL OR sf.is_limited_edition <> 'Y')")

    return " AND ".join(clauses), params


def build_inventory_filters(
    start_date: str,
    end_date: str,
    ip_name: str,
    rarity: str,
    limited_mode: str,
) -> Tuple[str, Dict[str, Any]]:
    clauses = ["d.full_date >= :start_date", "d.full_date <= :end_date"]
    params: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if ip_name != "All":
        clauses.append("ip.ip_name = :ip_name")
        params["ip_name"] = ip_name

    if rarity != "All":
        clauses.append("p.rarity = :rarity")
        params["rarity"] = rarity

    if limited_mode == "Limited only":
        clauses.append("p.is_limited_edition = 'Y'")
    elif limited_mode == "Non-limited only":
        clauses.append("(p.is_limited_edition IS NULL OR p.is_limited_edition <> 'Y')")

    return " AND ".join(clauses), params


def build_order_filters(
    start_date: str,
    end_date: str,
    member_level: str,
) -> Tuple[str, Dict[str, Any]]:
    clauses = ["o.order_time::date >= :start_date", "o.order_time::date <= :end_date"]
    params: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if member_level != "All":
        clauses.append("c.member_level = :member_level")
        params["member_level"] = member_level

    return " AND ".join(clauses), params


# ============================================================
# 3) Data fetchers (each chart is easy to locate and modify)
# ============================================================
def fetch_kpis(where_sales: str, params_sales: Dict[str, Any]) -> Dict[str, float]:
    df = read_sql_df(f"""
        SELECT
            COALESCE(SUM(sf.total_revenue), 0) AS total_revenue,
            COALESCE(SUM(sf.order_qty), 0) AS units_sold,
            COALESCE(COUNT(DISTINCT u.cus_id), 0) AS unique_customers
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d      ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u      ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r    ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p   ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip       ON sf.ip_id = ip.ip_id
        WHERE {where_sales};
    """, params_sales)

    total_revenue = float(df.loc[0, "total_revenue"])
    units_sold = float(df.loc[0, "units_sold"])
    unique_customers = float(df.loc[0, "unique_customers"])
    revenue_per_unit = (total_revenue / units_sold) if units_sold > 0 else 0.0

    return dict(
        total_revenue=total_revenue,
        units_sold=units_sold,
        unique_customers=unique_customers,
        revenue_per_unit=revenue_per_unit,
    )


def fetch_revenue_time(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT
            DATE_TRUNC('month', d.full_date) AS month_start,
            SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY 1;
    """, params_sales)


def fetch_units_time(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT
            DATE_TRUNC('month', d.full_date) AS month_start,
            SUM(sf.order_qty) AS units
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY 1;
    """, params_sales)


def fetch_revenue_by_member(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT u.member_level, SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC;
    """, params_sales)


def fetch_revenue_by_region(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT r.region_segment, SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC;
    """, params_sales)


def fetch_top_ips(where_sales: str, params_sales: Dict[str, Any], top_n: int) -> pd.DataFrame:
    params = dict(params_sales)
    params["top_n"] = int(top_n)
    return read_sql_df(f"""
        SELECT ip.ip_name, SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT :top_n;
    """, params)


def fetch_revenue_by_rarity(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT p.rarity, SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC;
    """, params_sales)


def fetch_limited_share(where_sales: str, params_sales: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT
            CASE WHEN sf.is_limited_edition = 'Y' THEN 'Limited' ELSE 'Standard' END AS edition_type,
            SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC;
    """, params_sales)


def fetch_top_products(where_sales: str, params_sales: Dict[str, Any], top_n: int) -> pd.DataFrame:
    params = dict(params_sales)
    params["top_n"] = int(top_n)
    return read_sql_df(f"""
        SELECT p.box_name, SUM(sf.total_revenue) AS revenue
        FROM {qname("sales_fact")} sf
        JOIN {qname("date_dim")} d ON sf.date_id = d.date_id
        JOIN {qname("user_dim")} u ON sf.user_dim_id = u.user_dim_id
        JOIN {qname("region_dim")} r ON sf.region_dim_id = r.region_dim_id
        JOIN {qname("product_dim")} p ON sf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON sf.ip_id = ip.ip_id
        WHERE {where_sales}
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT :top_n;
    """, params)


def fetch_inventory_trend(where_inv: str, params_inv: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT
            DATE_TRUNC('month', d.full_date) AS month_start,
            AVG(imf.remaining_stock) AS avg_remaining_stock
        FROM {qname("inventory_movement_fact")} imf
        JOIN {qname("date_dim")} d ON imf.date_id = d.date_id
        JOIN {qname("product_dim")} p ON imf.product_dim_id = p.product_dim_id
        JOIN {qname("ip_dim")} ip ON imf.ip_id = ip.ip_id
        WHERE {where_inv}
        GROUP BY 1
        ORDER BY 1;
    """, params_inv)


def fetch_order_pipeline(where_orders: str, params_orders: Dict[str, Any]) -> pd.DataFrame:
    return read_sql_df(f"""
        SELECT
            o.shipping_status,
            o.payment_status,
            COUNT(*) AS order_count
        FROM {qname("Order")} o
        JOIN {qname("customer")} c ON o.cus_id = c.cus_id
        WHERE {where_orders}
        GROUP BY 1, 2
        ORDER BY 1, 2;
    """, params_orders)


# ============================================================
# 4) Layout builders (keeps structure readable)
# ============================================================
def kpi_card(title: str, value: str) -> html.Div:
    return html.Div(
        className="kpi",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
        ],
    )

def graph_card(title: str, graph_id: str, height_px: int = 340) -> html.Div:
    return html.Div(
        className="card",
        style={"flex": "1 1 520px"},
        children=[
            html.Div(title, className="graph-title"),
            dcc.Graph(id=graph_id, config=_GRAPH_CONFIG, style={"height": f"{height_px}px"}),
        ],
    )

# ============================================================
# 5) Dash app
# ============================================================
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    className="container",
    children=[
        html.Div(
            className="header",
            children=[
                html.H2("INDENG 215 | Group 10 | Management Dashboard"),
                html.Div("This dashboard supports Fantasy Box Universe, a fictional blind-box toy business " \
                "that monetizes serialized IP storylines via offline/online sales and licensing. " \
                "It surfaces database-backed KPIs and trends to inform decisions on character launch cadence, inventory planning, "
                "and regional order performance.", className="subtle"),
            ],
        ),

        dcc.Interval(id="init_tick", interval=250, n_intervals=0, max_intervals=1),

        html.Div(
            className="row",
            children=[
                html.Div(
                    className="card controls",
                    children=[
                        html.H3("Filters"),

                        html.Div("Date range", className="control-label"),
                        dcc.DatePickerRange(
                            id="date_range",
                            minimum_nights=0,
                            display_format="YYYY-MM-DD",
                        ),

                        html.Hr(className="sep"),

                        html.Div("Member level", className="control-label"),
                        dcc.Dropdown(id="member_level_dd", clearable=False),

                        html.Div("Region segment", className="control-label"),
                        dcc.Dropdown(id="region_segment_dd", clearable=False),

                        html.Div("IP", className="control-label"),
                        dcc.Dropdown(id="ip_dd", clearable=False),

                        html.Div("Rarity", className="control-label"),
                        dcc.Dropdown(id="rarity_dd", clearable=False),

                        html.Div("Limited edition filter", className="control-label"),
                        dcc.RadioItems(
                            id="limited_mode",
                            options=[
                                {"label": "All", "value": "All"},
                                {"label": "Limited only", "value": "Limited only"},
                                {"label": "Non-limited only", "value": "Non-limited only"},
                            ],
                            value="All",
                            labelStyle={"display": "block", "marginBottom": "6px"},
                        ),

                        html.Div("Top N (Top IP / Top Products)", className="control-label"),
                        dcc.Slider(
                            id="top_n",
                            min=5,
                            max=25,
                            step=1,
                            value=10,
                            marks={5: "5", 10: "10", 15: "15", 20: "20", 25: "25"},
                        ),
                    ],
                ),

                html.Div(
                    className="card results",
                    children=[
                        html.H3("KPIs"),
                        html.Div(
                            className="kpi-grid",
                            children=[
                                kpi_card("Total revenue", "—"),
                                kpi_card("Units sold", "—"),
                                kpi_card("Unique customers", "—"),
                                kpi_card("Revenue per unit", "—"),
                            ],
                            id="kpi_grid",
                        ),

                        html.Hr(className="sep"),

                        html.H3("Visualizations (10)"),
                        html.Div(
                            className="row",
                            children=[
                                graph_card("Revenue over time", "fig_rev_time"),
                                graph_card("Units sold over time", "fig_units_time"),
                                graph_card("Revenue by member level", "fig_rev_member"),
                                graph_card("Revenue by region segment", "fig_rev_region"),
                                graph_card("Top IPs by revenue", "fig_rev_ip"),
                                graph_card("Revenue by rarity", "fig_rev_rarity"),
                                graph_card("Limited vs standard revenue share", "fig_limited_share"),
                                graph_card("Top products by revenue", "fig_top_products"),
                                graph_card("Inventory remaining stock trend", "fig_inventory"),
                                graph_card("Order pipeline: shipping x payment", "fig_order_pipeline"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# 6) Init callback (dropdown options + date range)
# ============================================================
@app.callback(
    Output("date_range", "min_date_allowed"),
    Output("date_range", "max_date_allowed"),
    Output("date_range", "start_date"),
    Output("date_range", "end_date"),
    Output("member_level_dd", "options"),
    Output("member_level_dd", "value"),
    Output("region_segment_dd", "options"),
    Output("region_segment_dd", "value"),
    Output("ip_dd", "options"),
    Output("ip_dd", "value"),
    Output("rarity_dd", "options"),
    Output("rarity_dd", "value"),
    Input("init_tick", "n_intervals"),
)
def init_filters(_n):
    df_dates = read_sql_df(f"""
        SELECT MIN(full_date) AS min_date, MAX(full_date) AS max_date
        FROM {qname("date_dim")};
    """)
    min_date = str(df_dates.loc[0, "min_date"])
    max_date = str(df_dates.loc[0, "max_date"])

    df_member = read_sql_df(f'SELECT DISTINCT member_level FROM {qname("user_dim")} ORDER BY member_level;')
    member_opts = [{"label": "All", "value": "All"}] + [
        {"label": str(x), "value": str(x)} for x in df_member["member_level"].dropna().tolist()
    ]

    df_region = read_sql_df(f'SELECT DISTINCT region_segment FROM {qname("region_dim")} ORDER BY region_segment;')
    region_opts = [{"label": "All", "value": "All"}] + [
        {"label": str(x), "value": str(x)} for x in df_region["region_segment"].dropna().tolist()
    ]

    df_ip = read_sql_df(f'SELECT DISTINCT ip_name FROM {qname("ip_dim")} ORDER BY ip_name;')
    ip_opts = [{"label": "All", "value": "All"}] + [
        {"label": str(x), "value": str(x)} for x in df_ip["ip_name"].dropna().tolist()
    ]

    df_rarity = read_sql_df(f'SELECT DISTINCT rarity FROM {qname("product_dim")} ORDER BY rarity;')
    rarity_opts = [{"label": "All", "value": "All"}] + [
        {"label": str(x), "value": str(x)} for x in df_rarity["rarity"].dropna().tolist()
    ]

    return (
        min_date, max_date, min_date, max_date,
        member_opts, "All",
        region_opts, "All",
        ip_opts, "All",
        rarity_opts, "All",
    )


# ============================================================
# 7) Main callback: KPIs + 10 figures
# ============================================================
@app.callback(
    Output("kpi_grid", "children"),
    Output("fig_rev_time", "figure"),
    Output("fig_units_time", "figure"),
    Output("fig_rev_member", "figure"),
    Output("fig_rev_region", "figure"),
    Output("fig_rev_ip", "figure"),
    Output("fig_rev_rarity", "figure"),
    Output("fig_limited_share", "figure"),
    Output("fig_top_products", "figure"),
    Output("fig_inventory", "figure"),
    Output("fig_order_pipeline", "figure"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("member_level_dd", "value"),
    Input("region_segment_dd", "value"),
    Input("ip_dd", "value"),
    Input("rarity_dd", "value"),
    Input("limited_mode", "value"),
    Input("top_n", "value"),
)
def update_dashboard(start_date, end_date, member_level, region_segment, ip_name, rarity, limited_mode, top_n):
    if not start_date or not end_date:
        blank = _warn_fig("Waiting for filters...", "Not available")
        kpis = [
            kpi_card("Total revenue", "—"),
            kpi_card("Units sold", "—"),
            kpi_card("Unique customers", "—"),
            kpi_card("Revenue per unit", "—"),
        ]
        return (kpis, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank)

    try:
        # Sales-driven charts
        where_sales, params_sales = build_sales_filters(
            start_date=start_date,
            end_date=end_date,
            member_level=member_level or "All",
            region_segment=region_segment or "All",
            ip_name=ip_name or "All",
            rarity=rarity or "All",
            limited_mode=limited_mode or "All",
        )

        # Inventory-driven chart
        where_inv, params_inv = build_inventory_filters(
            start_date=start_date,
            end_date=end_date,
            ip_name=ip_name or "All",
            rarity=rarity or "All",
            limited_mode=limited_mode or "All",
        )

        # Orders-driven chart
        where_orders, params_orders = build_order_filters(
            start_date=start_date,
            end_date=end_date,
            member_level=member_level or "All",
        )

        # KPIs
        kpi = fetch_kpis(where_sales, params_sales)
        kpis = [
            kpi_card("Total revenue", _fmt_usd(kpi["total_revenue"])),
            kpi_card("Units sold", _fmt_int(kpi["units_sold"])),
            kpi_card("Unique customers", _fmt_int(kpi["unique_customers"])),
            kpi_card("Revenue per unit", _fmt_usd2(kpi["revenue_per_unit"])),
        ]

        # 1) Revenue over time
        df_rev_time = fetch_revenue_time(where_sales, params_sales)
        fig_rev_time = px.line(df_rev_time, x="month_start", y="revenue", markers=True, template="jp_clean")
        fig_rev_time.update_layout(yaxis_title="Revenue", xaxis_title="Start Month")
        fig_rev_time.update_xaxes(tickformat="%m-%Y", ticklabelmode="period")
        
        # 2) Units over time
        df_units_time = fetch_units_time(where_sales, params_sales)
        fig_units_time = px.line(df_units_time, x="month_start", y="units", markers=True, template="jp_clean")
        fig_units_time.update_layout(yaxis_title="Units", xaxis_title="Start Month")
        fig_units_time.update_xaxes(tickformat="%m-%Y", ticklabelmode="period")

        # 3) Revenue by member
        df_rev_member = fetch_revenue_by_member(where_sales, params_sales)
        fig_rev_member = px.bar(df_rev_member, x="member_level", y="revenue", template="jp_clean")
        fig_rev_member.update_layout(xaxis_title="", yaxis_title="Revenue")

        # 4) Revenue by region segment
        df_rev_region = fetch_revenue_by_region(where_sales, params_sales)
        fig_rev_region = px.bar(df_rev_region, x="region_segment", y="revenue", template="jp_clean")
        fig_rev_region.update_layout(xaxis_title="", yaxis_title="Revenue")

        # 5) Top IPs
        df_rev_ip = fetch_top_ips(where_sales, params_sales, top_n=int(top_n))
        fig_rev_ip = px.bar(df_rev_ip, x="revenue", y="ip_name", orientation="h", template="jp_clean")
        fig_rev_ip.update_layout(xaxis_title="Revenue", yaxis_title="")
        fig_rev_ip.update_yaxes(autorange="reversed", automargin=True)

        # 6) Revenue by rarity
        df_rev_rarity = fetch_revenue_by_rarity(where_sales, params_sales)
        fig_rev_rarity = px.bar(df_rev_rarity, x="rarity", y="revenue", template="jp_clean")
        fig_rev_rarity.update_layout(xaxis_title="", yaxis_title="Revenue")

        # 7) Limited share
        df_lim = fetch_limited_share(where_sales, params_sales)
        fig_limited_share = px.pie(df_lim, names="edition_type", values="revenue", hole=0.45, template="jp_clean")
        fig_limited_share.update_layout(legend_title_text="")

        # 8) Top products
        df_top_products = fetch_top_products(where_sales, params_sales, top_n=int(top_n))
        fig_top_products = px.bar(df_top_products, x="revenue", y="box_name", orientation="h", template="jp_clean")
        fig_top_products.update_layout(xaxis_title="Revenue", yaxis_title="",)
        fig_top_products.update_yaxes(autorange="reversed", automargin=True)

        # 9) Inventory trend
        df_inv = fetch_inventory_trend(where_inv, params_inv)
        fig_inventory = px.line(df_inv, x="month_start", y="avg_remaining_stock", markers=True, template="jp_clean")
        fig_inventory.update_layout(yaxis_title="Avg remaining stock", xaxis_title="Start Month")
        fig_inventory.update_xaxes(tickformat="%m-%Y", ticklabelmode="period")

        # 10) Orders pipeline
        df_pipe = fetch_order_pipeline(where_orders, params_orders)
        fig_order_pipeline = px.bar(
            df_pipe,
            x="shipping_status",
            y="order_count",
            color="payment_status",
            barmode="stack",
            template="jp_clean",
        )
        fig_order_pipeline.update_layout(xaxis_title="", yaxis_title="Orders", legend_title_text="")

        return (
            kpis,
            fig_rev_time,
            fig_units_time,
            fig_rev_member,
            fig_rev_region,
            fig_rev_ip,
            fig_rev_rarity,
            fig_limited_share,
            fig_top_products,
            fig_inventory,
            fig_order_pipeline,
        )

    except Exception as e:
        msg = f"Query error: {type(e).__name__}: {e}"
        blank = _warn_fig(msg, "Query error")
        kpis = [
            kpi_card("Total revenue", "—"),
            kpi_card("Units sold", "—"),
            kpi_card("Unique customers", "—"),
            kpi_card("Revenue per unit", "—"),
        ]
        return (kpis, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank)


# ============================================================
# 8) Local run
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    #app.run(host="0.0.0.0", port=port, debug=DASH_DEBUG)
    app.run(debug=True)