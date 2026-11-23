"""
Rotation dashboard: spot sector, theme and country rotations at a glance.

Daily closing prices from Yahoo Finance (via yfinance).
- Themes and sectors are measured against the S&P 500 (SPY).
- Countries are measured against the MSCI All Country World index (ACWI).

Everything is relative: "+4.0" means the line beat its benchmark by about 4%
over that period.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Rotation dashboard", layout="wide")

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
TMT_KEY = "TMT-MOM"  # custom basket, built below from TMT_UNIVERSE

THEMES = [
    # (ticker, display name, group)
    ("XLK", "Tech", "Sectors"),
    ("XLV", "Health care", "Sectors"),
    ("XLF", "Financials", "Sectors"),
    ("XLE", "Energy", "Sectors"),
    ("XLI", "Industrials", "Sectors"),
    ("XLY", "Cons. discretionary", "Sectors"),
    ("XLP", "Cons. staples", "Sectors"),
    ("XLU", "Utilities", "Sectors"),
    ("XLB", "Materials", "Sectors"),
    ("XLRE", "Real estate", "Sectors"),
    ("XLC", "Comm. services", "Sectors"),
    ("SMH", "Semis", "Tech & health"),
    ("IGV", "Software", "Tech & health"),
    ("CIBR", "Cyber security", "Tech & health"),
    ("XBI", "Biotech", "Tech & health"),
    ("ITA", "Defense & aero", "Defense"),
    ("URA", "Uranium", "AI infrastructure"),
    ("GRID", "Power grid", "AI infrastructure"),
    ("GDX", "Gold miners", "Commodities"),
    ("OIH", "Oil services", "Commodities"),
    ("COPX", "Copper miners", "Commodities"),
    ("WGMI", "Bitcoin miners", "Crypto-linked"),
    ("MTUM", "US momentum", "Momentum"),
    (TMT_KEY, "TMT momentum", "Momentum"),
]

COUNTRIES = [
    ("SPY", "US", "Developed"),
    ("EWJ", "Japan", "Developed"),
    ("VGK", "Europe (broad)", "Developed"),
    ("EWU", "UK", "Developed"),
    ("EWG", "Germany", "Developed"),
    ("EWQ", "France", "Developed"),
    ("EWP", "Spain", "Developed"),
    ("EWI", "Italy", "Developed"),
    ("EWD", "Sweden", "Developed"),
    ("EDEN", "Denmark", "Developed"),
    ("NORW", "Norway", "Developed"),
    ("EWY", "Korea", "Asia EM"),
    ("EWT", "Taiwan", "Asia EM"),
    ("MCHI", "China", "Asia EM"),
    ("INDA", "India", "Asia EM"),
    ("EIDO", "Indonesia", "Asia EM"),
    ("VNM", "Vietnam", "Asia EM"),
    ("EWZ", "Brazil", "LatAm"),
    ("EWW", "Mexico", "LatAm"),
    ("ECH", "Chile", "LatAm"),
    ("ARGT", "Argentina", "LatAm"),
    ("EPOL", "Poland", "EM Europe"),
    ("GREK", "Greece", "EM Europe"),
    ("TUR", "Turkey", "EM Europe"),
]

# Large US tech, media and telecom names. Each month the basket holds the
# TMT_TOP_N names with the strongest 12-month return (skipping the latest
# month), equal-weighted.
TMT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "IBM",
    "INTU", "NOW", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "ADI", "MRVL",
    "INTC", "PANW", "CRWD", "FTNT", "ZS", "NET", "DDOG", "SNOW", "PLTR", "ANET",
    "SNPS", "CDNS", "DELL", "HPE", "WDC", "STX", "APP", "WDAY", "TEAM",
    "META", "GOOGL", "AMZN", "NFLX", "DIS", "CMCSA", "CHTR", "T", "VZ", "TMUS",
    "EA", "TTWO", "RDDT",
]
TMT_TOP_N = 15

VIEWS = {
    "themes": {
        "label": "Themes & sectors",
        "universe": THEMES,
        "bench": "SPY",
        "bench_name": "S&P 500",
    },
    "countries": {
        "label": "Countries",
        "universe": COUNTRIES,
        "bench": "ACWI",
        "bench_name": "MSCI ACWI",
    },
}

PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}

# Rotation speed: (trend lookback, momentum lookback) in trading days
SPEEDS = {
    "Fast": (21, 5),
    "Standard": (63, 21),
    "Slow": (126, 63),
}
SPEED_HELP = {
    "Fast": "Trend = 1M, momentum = 1W",
    "Standard": "Trend = 3M, momentum = 1M",
    "Slow": "Trend = 6M, momentum = 3M",
}
LOOKBACK_LABEL = {5: "1W", 21: "1M", 63: "3M", 126: "6M"}

QUADRANTS = ["Leading", "Weakening", "Lagging", "Improving"]

# --------------------------------------------------------------------------
# Colors (light theme; see .streamlit/config.toml)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HEAD = "#2a78d6"      # current position marker
TAIL = "#9ec5f4"      # path over previous weeks
TAIL_DARK = "#3987e5" # trail start ring and direction arrow
DIV_POS = "#1c5cab"   # beating the benchmark
DIV_NEG = "#c03b3a"   # trailing the benchmark
DIV_MID = "#f0efec"   # neutral
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner="Fetching daily prices…")
def load_prices(tickers: tuple) -> pd.DataFrame:
    """Daily adjusted closes (dividends reinvested), one column per ticker."""
    raw = yf.download(
        list(tickers),
        period="3y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index().dropna(how="all")
    close = close.ffill(limit=3)  # bridge single missing days
    return close


def build_tmt_momentum(px: pd.DataFrame, top_n: int = TMT_TOP_N):
    """Equal-weight basket of the top-N momentum names in TMT_UNIVERSE.

    Rebalanced on the first trading day of each month. Score = return from
    12 months ago to 1 month ago (the standard "12-1" momentum measure).
    Returns (index level series, current members with their scores).
    """
    cols = [t for t in TMT_UNIVERSE if t in px.columns]
    if len(cols) < top_n:
        return None, None
    p = px[cols]
    rets = p.pct_change(fill_method=None)
    idx = p.index
    month_starts = idx[~idx.to_period("M").duplicated()]

    members = {}
    last_scores = None
    for d in month_starts:
        loc = idx.get_loc(d)
        if loc < 252:
            continue
        score = (p.iloc[loc - 21] / p.iloc[loc - 252] - 1).dropna()
        if len(score) < top_n:
            continue
        top = score.nlargest(top_n)
        members[d] = top.index
        last_scores = top

    if not members:
        return None, None

    mask = pd.DataFrame(False, index=list(members), columns=cols)
    for d, names in members.items():
        mask.loc[d, names] = True
    # Picked at the close of the rebalance day, held from the next day on
    mask = mask.reindex(idx).ffill().shift(1)
    mask = mask.astype("boolean").fillna(False).astype(bool)

    basket_ret = rets.where(mask).mean(axis=1)
    first = mask.any(axis=1).idxmax()
    basket_ret = basket_ret.loc[first:].fillna(0.0)
    level = 100 * (1 + basket_ret).cumprod()
    level = level.reindex(idx)
    return level, last_scores


# --------------------------------------------------------------------------
# Calculations
# --------------------------------------------------------------------------
def relative_returns(px: pd.DataFrame, bench: pd.Series, n: int) -> pd.DataFrame:
    """n-day return of each line relative to the benchmark, in %, for every date."""
    rs = px.div(bench, axis=0)
    return (rs / rs.shift(n) - 1) * 100


def absolute_returns(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return (px / px.shift(n) - 1) * 100


def ytd(series_or_df):
    year_start = pd.Timestamp(datetime(series_or_df.index[-1].year, 1, 1))
    prior = series_or_df[series_or_df.index < year_start]
    if prior.empty:
        return series_or_df.iloc[-1] * np.nan
    return (series_or_df.iloc[-1] / prior.iloc[-1] - 1) * 100


def quadrant(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y):
        return "—"
    if x > 0 and y > 0:
        return "Leading"
    if x > 0:
        return "Weakening"
    if y <= 0:
        return "Lagging"
    return "Improving"


# --------------------------------------------------------------------------
# Charts and tables
# --------------------------------------------------------------------------
def rotation_figure(xs: pd.DataFrame, ys: pd.DataFrame, names: dict,
                    trend_lbl: str, mom_lbl: str, bench_name: str,
                    trail_for: list, unit: str = "week") -> go.Figure:
    """Quadrant chart. xs / ys: rows = weekly points (oldest -> newest), cols = tickers.

    Every line gets a dot for today. Only lines in `trail_for` get a trail.
    """
    fig = go.Figure()
    trail_for = [t for t in xs.columns if t in trail_for]

    # Fit the axes to today's dots plus the trails actually drawn
    fit_x = pd.concat([xs.iloc[-1], xs[trail_for].stack()]) if trail_for else xs.iloc[-1]
    fit_y = pd.concat([ys.iloc[-1], ys[trail_for].stack()]) if trail_for else ys.iloc[-1]
    xr = float(np.nanmax(np.abs(fit_x.values))) * 1.25 if len(fit_x) else 1
    yr = float(np.nanmax(np.abs(fit_y.values))) * 1.2 if len(fit_y) else 1
    xr, yr = max(xr, 0.5), max(yr, 0.5)

    # Quadrant tints: leading = the "beating" hue, lagging = the "trailing" hue
    fig.add_shape(type="rect", x0=0, x1=xr, y0=0, y1=yr, line_width=0,
                  fillcolor="rgba(42,120,214,0.07)", layer="below")
    fig.add_shape(type="rect", x0=-xr, x1=0, y0=-yr, y1=0, line_width=0,
                  fillcolor="rgba(227,73,72,0.07)", layer="below")
    corner = dict(showarrow=False, font=dict(size=13, color=MUTED, family=FONT))
    fig.add_annotation(x=xr, y=yr, text="<b>Leading</b>", xanchor="right", yanchor="top", **corner)
    fig.add_annotation(x=xr, y=-yr, text="<b>Weakening</b>", xanchor="right", yanchor="bottom", **corner)
    fig.add_annotation(x=-xr, y=-yr, text="<b>Lagging</b>", xanchor="left", yanchor="bottom", **corner)
    fig.add_annotation(x=-xr, y=yr, text="<b>Improving</b>", xanchor="left", yanchor="top", **corner)

    # Trails, oldest -> newest. The start is a hollow ring, the weekly dots grow
    # and darken towards today, and an arrow points into today's dot.
    n = len(xs)
    if n > 1 and trail_for:
        periods_ago = list(range(n - 1, 0, -1))  # excludes today (the big dot)
        sizes = np.linspace(4, 8, n - 1).tolist()
        alphas = np.linspace(0.35, 0.9, n - 1)
        colors = [f"rgba(57,135,229,{a:.2f})" for a in alphas]
        for t in trail_for:
            fig.add_trace(go.Scatter(
                x=xs[t].values, y=ys[t].values, mode="lines",
                hoverinfo="skip", showlegend=False,
                line=dict(color=TAIL, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=xs[t].values[:-1], y=ys[t].values[:-1], mode="markers",
                showlegend=False,
                marker=dict(size=sizes, color=colors),
                customdata=[[names[t], w, unit + ("s" if w > 1 else "")] for w in periods_ago],
                hovertemplate=("<b>%{customdata[0]}</b>, %{customdata[1]} %{customdata[2]} ago"
                               "<br>Trend: %{x:+.1f}%<br>Momentum: %{y:+.1f}%<extra></extra>"),
            ))
            # Start of the trail
            fig.add_trace(go.Scatter(
                x=[xs[t].values[0]], y=[ys[t].values[0]], mode="markers",
                hoverinfo="skip", showlegend=False,
                marker=dict(size=11, symbol="circle-open", color=TAIL_DARK, line=dict(width=2)),
            ))
            # Arrow from last week into today
            fig.add_annotation(
                x=xs[t].values[-1], y=ys[t].values[-1],
                ax=xs[t].values[-2], ay=ys[t].values[-2],
                xref="x", yref="y", axref="x", ayref="y", text="",
                showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2,
                arrowcolor=TAIL_DARK, standoff=7,
            )

    # Dots: today's position. Names are shown for trailed lines and for dots
    # away from the crowded middle; the rest show their name on hover.
    hx, hy = xs.iloc[-1], ys.iloc[-1]
    labels = [names[t] for t in xs.columns]
    quads = [quadrant(a, b) for a, b in zip(hx, hy)]
    shown_text = [
        names[t] if (t in trail_for or abs(a) > 0.3 * xr or abs(b) > 0.3 * yr) else ""
        for t, a, b in zip(xs.columns, hx, hy)
    ]
    positions = [("top" if b >= 0 else "bottom") + (" right" if a >= 0 else " left")
                 for a, b in zip(hx, hy)]
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers+text", showlegend=False,
        text=shown_text, textposition=positions, cliponaxis=False,
        textfont=dict(size=11, color=INK_2, family=FONT),
        marker=dict(size=11, color=HEAD, line=dict(width=2, color=SURFACE)),
        customdata=np.stack([labels, quads, list(xs.columns)], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
            f"Trend ({trend_lbl} vs {bench_name}): " + "%{x:+.1f}%<br>"
            f"Momentum ({mom_lbl} vs {bench_name}): " + "%{y:+.1f}%<br>"
            "%{customdata[1]}<extra></extra>"
        ),
    ))

    axis = dict(zeroline=True, zerolinecolor=AXIS, zerolinewidth=1,
                gridcolor=GRID, gridwidth=1, showline=False, automargin=True,
                tickfont=dict(color=MUTED, size=11), tickformat="+.0f", ticksuffix="%",
                title_font=dict(color=INK_2, size=12))
    fig.update_layout(
        height=620, margin=dict(l=70, r=20, t=10, b=60),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
        xaxis=dict(range=[-xr, xr],
                   title=f"Trend → return vs {bench_name}, last {trend_lbl}", **axis),
        yaxis=dict(range=[-yr, yr],
                   title=f"Momentum → return vs {bench_name}, last {mom_lbl}", **axis),
        dragmode="zoom",
    )
    return fig


def _blend(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def style_heatmap(df: pd.DataFrame, value_cols: list):
    """Diverging blue (beating) / red (trailing) per column, gray at zero."""
    caps = {}
    for c in value_cols:
        v = df[c].abs().dropna()
        caps[c] = max(float(v.quantile(0.9)) if len(v) else 1.0, 0.5)

    def cell_css(v, cap):
        if pd.isna(v):
            return f"background-color: {SURFACE}; color: {MUTED}"
        t = max(-1.0, min(1.0, v / cap))
        bg = _blend(DIV_MID, DIV_POS if t >= 0 else DIV_NEG, abs(t))
        fg = "#ffffff" if abs(t) > 0.6 else INK
        return f"background-color: {bg}; color: {fg}"

    styler = df.style
    for c in value_cols:
        styler = styler.map(lambda v, cap=caps[c]: cell_css(v, cap), subset=[c])
    styler = styler.format({c: "{:+.1f}" for c in value_cols}, na_rep="–")
    return styler


# --------------------------------------------------------------------------
# One view (themes or countries)
# --------------------------------------------------------------------------
def render_view(key: str, px_all: pd.DataFrame, tmt_level, tmt_members):
    cfg = VIEWS[key]
    bench_t, bench_name = cfg["bench"], cfg["bench_name"]
    universe = cfg["universe"]
    names = {t: n for t, n, _ in universe}
    groups = {t: g for t, _, g in universe}

    px = px_all.copy()
    if key == "themes" and tmt_level is not None:
        px[TMT_KEY] = tmt_level

    if bench_t not in px.columns or px[bench_t].dropna().empty:
        st.error(f"No benchmark data for {bench_t}. Try refreshing in a few minutes.")
        return
    bench = px[bench_t]

    available = [t for t, _, _ in universe if t in px.columns and px[t].dropna().size > 130]
    missing = [t for t, _, _ in universe if t not in available]

    # --- Benchmark context -------------------------------------------------
    b = bench.dropna()
    cols = st.columns(5)
    cols[0].markdown(f"**{bench_name}** ({bench_t})  \nabsolute return")
    for col, (lbl, n) in zip(cols[1:], [("1W", 5), ("1M", 21), ("3M", 63)]):
        col.metric(lbl, f"{(b.iloc[-1] / b.iloc[-1 - n] - 1) * 100:+.1f}%")
    cols[4].metric("YTD", f"{ytd(b):+.1f}%")

    # --- Filters (one row) -------------------------------------------------
    all_groups = list(dict.fromkeys(groups[t] for t in available))
    f1, f2, f3, f4 = st.columns([4, 2, 2, 2])
    with f1:
        chosen = st.pills("Groups", all_groups, selection_mode="multi",
                          default=all_groups, key=f"{key}_groups")
    with f2:
        speed = st.segmented_control("Rotation speed", list(SPEEDS), default="Standard",
                                     key=f"{key}_speed",
                                     help=" · ".join(f"{k}: {v}" for k, v in SPEED_HELP.items()))
        speed = speed or "Standard"
    with f3:
        trail_unit = st.segmented_control(
            "Trail in", ["Weeks", "Months"], default="Weeks", key=f"{key}_unit",
            help="Weeks: one dot per week, for recent turns. Months: one dot per month, "
                 "for the bigger picture (pairs well with Slow speed).")
        trail_unit = trail_unit or "Weeks"
    with f4:
        if trail_unit == "Weeks":
            trail_n = st.slider("Trail length (weeks)", 1, 8, 4, key=f"{key}_tail_w")
        else:
            trail_n = st.slider("Trail length (months)", 1, 12, 6, key=f"{key}_tail_m")
    step = 5 if trail_unit == "Weeks" else 21  # trading days per dot
    if trail_unit == "Months" and speed != "Slow":
        st.caption(
            "Tip: monthly trails read best with **Slow** speed (6M trend, 3M momentum). "
            "Then the zones are measured over the same longer time frame the trail covers; "
            f"with {speed} speed the zones reflect only the last "
            f"{LOOKBACK_LABEL[SPEEDS[speed][0]]} / {LOOKBACK_LABEL[SPEEDS[speed][1]]}."
        )

    shown = [t for t in available if groups[t] in (chosen or [])]
    if not shown:
        st.info("Pick at least one group.")
        return

    n_trend, n_mom = SPEEDS[speed]
    trend_lbl, mom_lbl = LOOKBACK_LABEL[n_trend], LOOKBACK_LABEL[n_mom]

    rel_trend = relative_returns(px[shown], bench, n_trend)
    rel_mom = relative_returns(px[shown], bench, n_mom)

    # Weekly sample points for tails, oldest -> newest
    last = len(px) - 1
    pts = [last - step * k for k in range(trail_n, -1, -1) if last - step * k >= 0]
    xs = rel_trend.iloc[pts].dropna(axis=1, how="any")
    ys = rel_mom.iloc[pts][xs.columns]
    ys = ys.dropna(axis=1, how="any")
    xs = xs[ys.columns]

    x_now, y_now = rel_trend.iloc[-1], rel_mom.iloc[-1]
    x_prev, y_prev = rel_trend.iloc[-6], rel_mom.iloc[-6]
    q_now = {t: quadrant(x_now[t], y_now[t]) for t in shown}
    q_prev = {t: quadrant(x_prev[t], y_prev[t]) for t in shown}

    # --- Rotation map + what changed --------------------------------------
    movers = [t for t in shown
              if q_now[t] != q_prev[t] and "—" not in (q_now[t], q_prev[t])]

    left, right = st.columns([7, 3], gap="large")
    with left:
        st.markdown("#### Rotation map")
        st.caption(
            "The big dot is where a line sits today. A trail runs back in time, one small "
            "dot per week (or per month, if you switch \"Trail in\" to Months): the hollow ring is where it started and the arrow shows which way "
            "it's heading. Lines tend to rotate clockwise: Improving → Leading → Weakening → "
            "Lagging. By default, trails show for the lines that changed zone this week. Hover any dot for "
            "its name and date, or drag a box to zoom."
        )
        trail_for = st.multiselect(
            "Show trail for", options=list(xs.columns),
            default=[t for t in movers if t in xs.columns],
            format_func=lambda t: names[t],
            key=f"{key}_trails_{hash(tuple(xs.columns))}",
            placeholder="Pick lines to see their path",
        )
        if xs.shape[1] == 0:
            st.info("Not enough history for the chosen speed.")
        else:
            fig = rotation_figure(xs, ys, names, trend_lbl, mom_lbl, bench_name, trail_for,
                                  unit="week" if trail_unit == "Weeks" else "month")
            st.plotly_chart(fig, theme=None, key=f"{key}_rrg",
                            config={"displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    with right:
        st.markdown("#### What changed this week")
        moves = [(t, q_prev[t], q_now[t]) for t in movers]
        if moves:
            for target in ["Leading", "Improving", "Weakening", "Lagging"]:
                into = [m for m in moves if m[2] == target]
                if not into:
                    continue
                st.markdown(f"**Into {target}**")
                st.markdown("\n".join(f"- {names[t]} <span style='color:{MUTED}'>from {frm}</span>"
                                      for t, frm, _ in into), unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:{MUTED}'>No quadrant changes in the past week.</span>",
                        unsafe_allow_html=True)

        st.markdown(f"**Momentum ranking ({mom_lbl}), 1-week change**")
        r_now = y_now[shown].rank(ascending=False)
        r_prev = y_prev[shown].rank(ascending=False)
        delta = (r_prev - r_now).dropna()
        if len(delta):
            up = delta[delta > 0].sort_values(ascending=False).head(3)
            down = delta[delta < 0].sort_values().head(3)
            lines = []
            for t, d in up.items():
                lines.append(f"- ▲ {names[t]} up {int(d)} to #{int(r_now[t])}")
            for t, d in down.items():
                lines.append(f"- ▼ {names[t]} down {int(-d)} to #{int(r_now[t])}")
            st.markdown("\n".join(lines) if lines else "No rank changes.")

        counts = pd.Series([q_now[t] for t in shown]).value_counts()
        st.markdown("**Where things sit now**")
        st.markdown("  \n".join(f"{q}: {int(counts.get(q, 0))}" for q in QUADRANTS))

    # --- Heatmap table ----------------------------------------------------
    st.markdown("#### Performance heatmap")
    h1, h2 = st.columns([3, 5])
    with h1:
        mode = st.segmented_control("Show", [f"vs {bench_name}", "Absolute"],
                                    default=f"vs {bench_name}", key=f"{key}_mode")
        mode = mode or f"vs {bench_name}"
    relative = mode != "Absolute"

    rows = {}
    for lbl, n in PERIODS.items():
        rows[lbl] = (relative_returns(px[shown], bench, n) if relative
                     else absolute_returns(px[shown], n)).iloc[-1]
    if relative:
        rows["YTD"] = ytd(px[shown].div(bench, axis=0))
    else:
        rows["YTD"] = ytd(px[shown])
    table = pd.DataFrame(rows)
    table.insert(0, "Quadrant", [q_now[t] for t in table.index])
    table.insert(0, "Group", [groups[t] for t in table.index])
    table.insert(0, "Name", [names[t] for t in table.index])
    table.index.name = "Ticker"
    table = table.sort_values("1M", ascending=False)

    value_cols = list(PERIODS) + ["YTD"]
    st.dataframe(style_heatmap(table, value_cols),
                 height=min(38 + 35 * len(table), 900), key=f"{key}_table")
    st.caption(
        ("Return vs " + bench_name + ", %. Blue = beating it, red = trailing it. "
         if relative else "Absolute return, %. ")
        + "Colour scale is per column, so a 1W and a YTD cell of the same shade aren't "
          "the same size move. Click a column header to sort."
    )

    if key == "themes" and tmt_members is not None:
        with st.expander("What's in the TMT momentum basket"):
            st.markdown(
                f"My own basket, not a bank's: the {TMT_TOP_N} US tech, media and telecom stocks "
                "with the strongest 12-month return (skipping the latest month), equal-weighted, "
                "re-picked on the first trading day of each month. Current holdings and the "
                "score they were picked on:"
            )
            st.dataframe(
                tmt_members.mul(100).round(1).rename("12-1M return, %").to_frame(),
                height=38 + 35 * len(tmt_members),
            )

    if missing:
        st.caption("No usable data for: " + ", ".join(missing))


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main():
    st.title("Rotation dashboard")

    tickers = {t for v in VIEWS.values() for t, _, _ in v["universe"] if t != TMT_KEY}
    tickers |= {v["bench"] for v in VIEWS.values()}
    tickers |= set(TMT_UNIVERSE)
    px = load_prices(tuple(sorted(tickers)))

    if px.empty:
        st.error("Couldn't fetch prices from Yahoo Finance right now. "
                 "Wait a few minutes and press Refresh.")
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    tmt_level, tmt_members = build_tmt_momentum(px)

    c1, c2 = st.columns([6, 1])
    c1.caption(f"Daily closes from Yahoo Finance · data as of {px.index[-1]:%d %b %Y} · "
               "refreshes every 6 hours")
    if c2.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    tab_t, tab_c, tab_h = st.tabs(["Themes & sectors (vs S&P 500)",
                                   "Countries (vs MSCI ACWI)",
                                   "How to read this"])
    with tab_t:
        render_view("themes", px, tmt_level, tmt_members)
    with tab_c:
        render_view("countries", px, tmt_level, tmt_members)
    with tab_h:
        st.markdown(
            """
**Everything is relative.** In a rising market almost everything is up. What matters
for rotation is what's beating the market and what's falling behind it. Themes and sectors
are measured against the S&P 500 (SPY), countries against MSCI ACWI (the whole world,
emerging markets included). All prices are in USD, so country lines include currency moves.

**Rotation map.** Two numbers per line:
- **Trend** (across): how it has done against the benchmark over the longer window.
- **Momentum** (up): how it has done against the benchmark over the shorter, recent window.

That gives four zones:
- **Leading**: strong trend, still gaining. The current winners.
- **Weakening**: strong trend, but losing ground lately. Leadership may be fading.
- **Lagging**: weak trend, still losing. The current losers.
- **Improving**: weak trend, but gaining lately. Where a new rotation often starts.

Lines tend to travel clockwise through the zones. A trail shows a line's path over the
last few weeks, so you can see the direction, not just the position. It starts at the hollow
ring, has one small dot per week (getting bigger and darker towards today), and ends with an
arrow into today's big dot. Switch **Trail in** to *Months* for one dot per month going back
up to a year. Weekly trails show recent turns; monthly trails show the bigger cycle. Use
monthly trails with the *Slow* rotation speed (6M trend, 3M momentum), so the zones are
measured over the same longer time frame the trail covers. By default, trails are shown
for the lines that changed zone this week; use "Show trail for" to add others. Dots near the
centre are close to the benchmark, so their names only show on hover.

**Rotation speed.** *Standard* uses 3 months for trend and 1 month for momentum.
*Fast* (1M / 1W) reacts quicker but is noisier. *Slow* (6M / 3M) shows the big shifts.

**What changed this week.** Lines that moved into a new zone over the past five trading
days, and the biggest moves up and down the momentum ranking. This is the actual signal.

**Heatmap.** Returns over 1 week, 1 month, 3 months, 6 months and year to date. Switch
between "vs benchmark" and "Absolute".

**TMT momentum.** There's no public ETF for this, so the dashboard builds its own basket:
the 15 US tech, media and telecom stocks with the strongest 12-month return (skipping
the latest month), equal-weighted and re-picked monthly. See the expander under the
themes heatmap for what it holds now.
"""
        )


main()
