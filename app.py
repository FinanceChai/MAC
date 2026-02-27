"""
MAC — Multi-Asset Calculator
Portfolio benchmarking tool: compare a custom allocation across 5 ETFs
(SPY, GLD, BIL, AGG, DBC) against the S&P 500, US Agg Bonds, 60/40, and
Butterfly (equal-weight) portfolio.

Charts:  Growth of $100  |  30-Day Rolling Vol  |  30-Day Rolling Sharpe
Tables:  Annualised Returns, Vols, and Sharpe across 1 / 3 / 5 / 10 Y windows.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ── Constants ─────────────────────────────────────────────────────────────────

ETF_TICKERS: list[str] = ["SPY", "GLD", "BIL", "AGG", "DBC"]

ETF_LABELS: dict[str, str] = {
    "SPY": "S&P 500",
    "GLD": "Gold",
    "BIL": "T-Bills (Cash)",
    "AGG": "US Agg Bonds",
    "DBC": "Commodities",
}

ETF_COLORS: dict[str, str] = {
    "SPY": "#2196F3",
    "GLD": "#FFD700",
    "BIL": "#4CAF50",
    "AGG": "#CE93D8",
    "DBC": "#FF7043",
}

# Benchmark definitions
BENCHMARKS: dict[str, dict[str, float]] = {
    "S&P 500":   {"SPY": 1.0},
    "US Agg":    {"AGG": 1.0},
    "60 / 40":   {"SPY": 0.60, "AGG": 0.40},
    "Butterfly": {"SPY": 0.20, "GLD": 0.20, "BIL": 0.20, "AGG": 0.20, "DBC": 0.20},
}

# Style map: (color, dash)
BENCH_STYLES: dict[str, tuple[str, str]] = {
    "S&P 500":      ("#2196F3", "solid"),
    "US Agg":       ("#CE93D8", "solid"),
    "60 / 40":      ("#4CAF50", "dash"),
    "Butterfly":    ("#FFD700", "dash"),
    "Your Portfolio": ("#FF5722", "solid"),
}

PERIODS: dict[str, int] = {
    "1Y": 1,
    "3Y": 3,
    "5Y": 5,
    "10Y": 10,
}

VOL_WINDOW = 30      # trading days
ANNUALIZE  = np.sqrt(252)
RF_TICKER  = "BIL"  # risk-free proxy

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MAC — Multi-Asset Calculator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* tighter metric cards */
    .stMetric { border: 1px solid #1E2A3A; border-radius: 6px; padding: 8px 12px; }
    div[data-testid="stHorizontalBlock"] > div { gap: 0.5rem; }
    /* table styling */
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #1E2A3A; color: #90CAF9; text-align: center; padding: 6px 10px; }
    td { text-align: center; padding: 5px 10px; border-bottom: 1px solid #1E2A3A; }
    tr:hover td { background: #1A2332; }
    .positive { color: #66BB6A; }
    .negative { color: #EF5350; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — portfolio builder ───────────────────────────────────────────────

with st.sidebar:
    st.markdown("## MAC Controls")
    st.caption("Multi-Asset Calculator")
    st.divider()

    st.markdown("### Your Portfolio")
    st.caption("Allocate across the 5 ETFs — must sum to **100 %**.")

    # Default weights
    if "mac_weights" not in st.session_state:
        st.session_state.mac_weights = {
            "SPY": 20.0,
            "GLD": 20.0,
            "BIL": 20.0,
            "AGG": 20.0,
            "DBC": 20.0,
        }

    weights: dict[str, float] = {}
    for tk in ETF_TICKERS:
        weights[tk] = st.number_input(
            f"{tk} — {ETF_LABELS[tk]}",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.mac_weights[tk]),
            step=5.0,
            key=f"mac_w_{tk}",
        )

    total = sum(weights.values())
    if abs(total - 100.0) < 0.05:
        st.success(f"Total: {total:.1f} % ✓")
    else:
        st.warning(f"Total: {total:.1f} % — need 100 %")
        if st.button("Normalize to 100 %", use_container_width=True):
            if total > 0:
                for tk in ETF_TICKERS:
                    weights[tk] = round(weights[tk] / total * 100, 2)
            st.session_state.mac_weights = weights
            st.rerun()

    st.divider()
    st.markdown("### Benchmark Definitions")
    for name, alloc in BENCHMARKS.items():
        color, _ = BENCH_STYLES[name]
        parts = "  ·  ".join(f"{t}: {int(w*100)}%" for t, w in alloc.items())
        st.markdown(
            f"<span style='color:{color}'>■</span> **{name}**  \n"
            f"<span style='font-size:0.75rem;color:#90A4AE'>{parts}</span>",
            unsafe_allow_html=True,
        )

# Guard: weights must sum to 100 % before proceeding
user_weights_raw = {tk: w for tk, w in weights.items() if w > 0}
if abs(total - 100.0) >= 0.05:
    st.warning(
        "⚠️  Portfolio weights don't sum to 100 %. "
        "Adjust the sliders in the sidebar, then use **Normalize** or fix manually."
    )
    st.stop()

# Normalise to fractions
user_weights: dict[str, float] = {tk: w / 100.0 for tk, w in user_weights_raw.items()}

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices for all 5 ETFs."""
    raw = yf.download(
        ETF_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=ETF_TICKERS[0])
    else:
        close = raw[["Close"]]
    return close.dropna(how="all")


today      = datetime.today()
fetch_start = today - timedelta(days=365 * 11)   # 11 Y buffer for 10-Y windows + warm-up
warmup_days = 60                                  # extra days so rolling starts clean

with st.spinner("Fetching market data…"):
    try:
        prices = fetch_prices(
            (fetch_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

if prices.empty:
    st.error("No price data returned. Check your connection.")
    st.stop()

# Align columns to available tickers
available = [t for t in ETF_TICKERS if t in prices.columns]
prices = prices[available].dropna(how="all")

# ── Calculation helpers ───────────────────────────────────────────────────────

def daily_ret(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change()


def port_ret(px: pd.DataFrame, wts: dict[str, float]) -> pd.Series:
    """Weighted portfolio daily return series."""
    tks = [t for t in wts if t in px.columns]
    w   = np.array([wts[t] for t in tks], dtype=float)
    w  /= w.sum()
    return daily_ret(px[tks]).dot(w)


def growth_of_100(px: pd.DataFrame, wts: dict[str, float], start: pd.Timestamp) -> pd.Series:
    """Normalised cumulative return indexed to 100 at `start`."""
    ret = port_ret(px, wts)
    ret = ret[ret.index >= start].dropna()
    cumret = (1 + ret).cumprod()
    return cumret * 100


def roll_vol(px: pd.DataFrame, wts: dict[str, float]) -> pd.Series:
    """30-day rolling annualised volatility."""
    pr = port_ret(px, wts)
    return (pr.rolling(VOL_WINDOW).std() * ANNUALIZE)


def roll_sharpe(px: pd.DataFrame, wts: dict[str, float]) -> pd.Series:
    """30-day rolling annualised Sharpe (excess over BIL)."""
    pr = port_ret(px, wts)
    rf = daily_ret(px)[RF_TICKER].reindex(pr.index).fillna(0) if RF_TICKER in px.columns else pd.Series(0.0, index=pr.index)
    ex = pr - rf
    roll_mean = ex.rolling(VOL_WINDOW).mean() * 252
    roll_std  = ex.rolling(VOL_WINDOW).std() * ANNUALIZE
    return (roll_mean / roll_std.replace(0, np.nan))


def period_start(years: int) -> pd.Timestamp:
    return pd.Timestamp(today - timedelta(days=int(years * 365.25)))


def annualised_return(px: pd.DataFrame, wts: dict[str, float], years: int) -> float | None:
    """CAGR over `years` years."""
    ps = period_start(years)
    ret = port_ret(px, wts)
    ret = ret[ret.index >= ps].dropna()
    if len(ret) < 20:
        return None
    total = (1 + ret).prod()
    n_years = len(ret) / 252
    return total ** (1 / n_years) - 1 if n_years > 0 else None


def annualised_vol(px: pd.DataFrame, wts: dict[str, float], years: int) -> float | None:
    """Annualised vol over `years` years."""
    ps = period_start(years)
    ret = port_ret(px, wts)
    ret = ret[ret.index >= ps].dropna()
    if len(ret) < 20:
        return None
    return ret.std() * ANNUALIZE


def sharpe_ratio(px: pd.DataFrame, wts: dict[str, float], years: int) -> float | None:
    """Sharpe ratio over `years` years, using BIL as rf."""
    ps = period_start(years)
    pr = port_ret(px, wts)
    pr = pr[pr.index >= ps].dropna()
    if len(pr) < 20:
        return None
    rf = daily_ret(px)[RF_TICKER].reindex(pr.index).fillna(0) if RF_TICKER in px.columns else pd.Series(0.0, index=pr.index)
    ex = pr - rf
    ann_ret = ex.mean() * 252
    ann_std = ex.std() * ANNUALIZE
    return ann_ret / ann_std if ann_std > 0 else None


# ── Build benchmark + portfolio weight map ────────────────────────────────────

all_portfolios: dict[str, dict[str, float]] = {**BENCHMARKS, "Your Portfolio": user_weights}

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# MAC — Multi-Asset Calculator")
st.caption(
    f"Portfolio benchmarking across SPY · GLD · BIL · AGG · DBC  ·  "
    f"Data through {today.strftime('%b %d, %Y')}"
)

# ── Metric strip — current snapshot (1Y values) ───────────────────────────────

st.markdown("#### Current Snapshot  <span style='font-size:0.75rem;color:#90A4AE'>(1-Year)</span>", unsafe_allow_html=True)
metric_cols = st.columns(len(all_portfolios))
for col, (name, wts) in zip(metric_cols, all_portfolios.items()):
    color, _ = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
    ret_1y = annualised_return(prices, wts, 1)
    vol_1y = annualised_vol(prices, wts, 1)
    shr_1y = sharpe_ratio(prices, wts, 1)
    col.markdown(
        f"<span style='color:{color};font-weight:700'>{name}</span>",
        unsafe_allow_html=True,
    )
    col.metric("Return", f"{ret_1y*100:.1f}%" if ret_1y is not None else "—")
    col.metric("Vol",    f"{vol_1y*100:.1f}%" if vol_1y is not None else "—")
    col.metric("Sharpe", f"{shr_1y:.2f}"       if shr_1y is not None else "—")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_growth, tab_vol, tab_sharpe, tab_table = st.tabs([
    "📈  Growth of $100",
    "〰  Rolling Volatility",
    "⚡  Rolling Sharpe",
    "📊  Summary Table",
])

# ─── Tab 1 — Growth of $100 ───────────────────────────────────────────────────

with tab_growth:
    period_sel = st.radio(
        "Period", list(PERIODS.keys()), index=0, horizontal=True, key="growth_period"
    )
    years = PERIODS[period_sel]
    ps    = period_start(years)

    fig_g = go.Figure()

    for name, wts in all_portfolios.items():
        color, dash = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
        tks = [t for t in wts if t in prices.columns]
        if not tks:
            continue
        g = growth_of_100(prices, wts, ps)
        if g.empty:
            continue
        width = 2.5 if name == "Your Portfolio" else 2.0
        fig_g.add_trace(go.Scatter(
            x=g.index, y=g.round(2),
            name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"$%{{y:.2f}}<extra>{name}</extra>",
        ))

    fig_g.update_layout(
        title=f"Growth of $100 — {period_sel}",
        xaxis_title="",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=480,
        hovermode="x unified",
        margin=dict(t=55, b=35, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # Small total-return table below the chart
    st.markdown("**Total Return over Period**")
    tr_data: dict[str, str] = {}
    for name, wts in all_portfolios.items():
        tks = [t for t in wts if t in prices.columns]
        if not tks:
            tr_data[name] = "—"
            continue
        g = growth_of_100(prices, wts, ps)
        if g.empty or len(g) < 2:
            tr_data[name] = "—"
            continue
        tr = (g.iloc[-1] / 100 - 1) * 100
        color_cls = "positive" if tr >= 0 else "negative"
        sign = "+" if tr >= 0 else ""
        tr_data[name] = f"<span class='{color_cls}'>{sign}{tr:.1f}%</span>"

    cols_tr = st.columns(len(all_portfolios))
    for col, (name, val) in zip(cols_tr, tr_data.items()):
        col.markdown(f"**{name}**  \n{val}", unsafe_allow_html=True)


# ─── Tab 2 — Rolling Volatility ───────────────────────────────────────────────

with tab_vol:
    period_vol = st.radio(
        "Period", list(PERIODS.keys()), index=1, horizontal=True, key="vol_period"
    )
    years_v = PERIODS[period_vol]
    ps_v    = period_start(years_v)

    fig_v = go.Figure()

    for name, wts in all_portfolios.items():
        color, dash = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
        tks = [t for t in wts if t in prices.columns]
        if not tks:
            continue
        rv = roll_vol(prices, wts)
        rv = rv[rv.index >= ps_v].dropna()
        if rv.empty:
            continue
        width = 2.5 if name == "Your Portfolio" else 2.0
        fill = "tozeroy" if name == "Butterfly" else "none"
        fillcolor = "rgba(255,215,0,0.06)" if name == "Butterfly" else None
        fig_v.add_trace(go.Scatter(
            x=rv.index, y=(rv * 100).round(2),
            name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            fill=fill,
            fillcolor=fillcolor,
            hovertemplate=f"%{{y:.2f}}%<extra>{name}</extra>",
        ))

    fig_v.update_layout(
        title=f"30-Day Rolling Annualised Volatility — {period_vol}",
        xaxis_title="",
        yaxis_title="Annualised Vol (%)",
        yaxis_ticksuffix="%",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=480,
        hovermode="x unified",
        margin=dict(t=55, b=35, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_v, use_container_width=True)


# ─── Tab 3 — Rolling Sharpe ───────────────────────────────────────────────────

with tab_sharpe:
    period_shr = st.radio(
        "Period", list(PERIODS.keys()), index=1, horizontal=True, key="sharpe_period"
    )
    years_s = PERIODS[period_shr]
    ps_s    = period_start(years_s)

    fig_s = go.Figure()

    for name, wts in all_portfolios.items():
        color, dash = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
        tks = [t for t in wts if t in prices.columns]
        if not tks:
            continue
        rs = roll_sharpe(prices, wts)
        rs = rs[rs.index >= ps_s].dropna()
        if rs.empty:
            continue
        width = 2.5 if name == "Your Portfolio" else 2.0
        fig_s.add_trace(go.Scatter(
            x=rs.index, y=rs.round(3),
            name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"%{{y:.2f}}<extra>{name} Sharpe</extra>",
        ))

    # Zero reference line
    fig_s.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="0",
        annotation_position="right",
        annotation_font_color="rgba(255,255,255,0.5)",
        annotation_font_size=11,
    )

    fig_s.update_layout(
        title=f"30-Day Rolling Sharpe Ratio — {period_shr}  (rf = BIL)",
        xaxis_title="",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=480,
        hovermode="x unified",
        margin=dict(t=55, b=35, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_s, use_container_width=True)


# ─── Tab 4 — Summary Table ────────────────────────────────────────────────────

with tab_table:
    st.markdown("### Annualised Performance Summary")
    st.caption(
        "Returns, volatilities, and Sharpe ratios computed over trailing windows. "
        "Sharpe uses BIL as the risk-free rate."
    )

    year_cols = list(PERIODS.keys())   # ["1Y", "3Y", "5Y", "10Y"]

    def fmt_pct(v: float | None) -> str:
        if v is None:
            return "—"
        cls = "positive" if v >= 0 else "negative"
        sign = "+" if v >= 0 else ""
        return f"<span class='{cls}'>{sign}{v*100:.1f}%</span>"

    def fmt_sharpe(v: float | None) -> str:
        if v is None:
            return "—"
        cls = "positive" if v >= 0 else "negative"
        sign = "+" if v >= 0 else ""
        return f"<span class='{cls}'>{sign}{v:.2f}</span>"

    # Build three tables side by side
    c1, c2, c3 = st.columns(3)

    # ── Returns Table ──────────────────────────────────────────────────────────
    with c1:
        st.markdown("**Annualised Returns**")
        header = "| Portfolio | " + " | ".join(year_cols) + " |"
        sep    = "| --- | " + " | ".join(["---"] * len(year_cols)) + " |"
        rows   = [header, sep]
        for name, wts in all_portfolios.items():
            color, _ = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
            vals = [fmt_pct(annualised_return(prices, wts, PERIODS[yc])) for yc in year_cols]
            label = f"<span style='color:{color}'>{name}</span>"
            rows.append(f"| {label} | " + " | ".join(vals) + " |")
        html_ret = "<table><tr><th>Portfolio</th>" + "".join(f"<th>{y}</th>" for y in year_cols) + "</tr>"
        for name, wts in all_portfolios.items():
            color, _ = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
            html_ret += f"<tr><td style='color:{color};text-align:left'>{name}</td>"
            for yc in year_cols:
                html_ret += f"<td>{fmt_pct(annualised_return(prices, wts, PERIODS[yc]))}</td>"
            html_ret += "</tr>"
        html_ret += "</table>"
        st.markdown(html_ret, unsafe_allow_html=True)

    # ── Volatility Table ───────────────────────────────────────────────────────
    with c2:
        st.markdown("**Annualised Volatility**")
        html_vol = "<table><tr><th>Portfolio</th>" + "".join(f"<th>{y}</th>" for y in year_cols) + "</tr>"
        for name, wts in all_portfolios.items():
            color, _ = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
            html_vol += f"<tr><td style='color:{color};text-align:left'>{name}</td>"
            for yc in year_cols:
                v = annualised_vol(prices, wts, PERIODS[yc])
                html_vol += f"<td>{f'{v*100:.1f}%' if v is not None else '—'}</td>"
            html_vol += "</tr>"
        html_vol += "</table>"
        st.markdown(html_vol, unsafe_allow_html=True)

    # ── Sharpe Table ───────────────────────────────────────────────────────────
    with c3:
        st.markdown("**Sharpe Ratio**")
        html_shr = "<table><tr><th>Portfolio</th>" + "".join(f"<th>{y}</th>" for y in year_cols) + "</tr>"
        for name, wts in all_portfolios.items():
            color, _ = BENCH_STYLES.get(name, ("#FFFFFF", "solid"))
            html_shr += f"<tr><td style='color:{color};text-align:left'>{name}</td>"
            for yc in year_cols:
                html_shr += f"<td>{fmt_sharpe(sharpe_ratio(prices, wts, PERIODS[yc]))}</td>"
            html_shr += "</tr>"
        html_shr += "</table>"
        st.markdown(html_shr, unsafe_allow_html=True)

    st.divider()

    # ── Per-ETF breakdown for the user's portfolio ─────────────────────────────
    st.markdown("### Your Portfolio — Asset Breakdown")
    st.caption("Individual ETF contribution (weighted) to Your Portfolio.")

    etf_header = "<table><tr><th>ETF</th><th>Asset Class</th><th>Weight</th>" + \
                 "".join(f"<th>1Y Return</th><th>Vol</th><th>Sharpe</th>") + "</tr>"
    # Simplify: just one set of 1Y metrics per ETF
    html_etf = "<table><tr><th>ETF</th><th>Asset Class</th><th>Weight</th>" \
               "<th>1Y Return</th><th>1Y Vol</th><th>1Y Sharpe</th></tr>"
    for tk in ETF_TICKERS:
        if tk not in prices.columns:
            continue
        w = user_weights.get(tk, 0.0)
        r = annualised_return(prices, {tk: 1.0}, 1)
        v = annualised_vol(prices, {tk: 1.0}, 1)
        s = sharpe_ratio(prices, {tk: 1.0}, 1)
        color = ETF_COLORS[tk]
        html_etf += (
            f"<tr>"
            f"<td style='color:{color};text-align:left;font-weight:700'>{tk}</td>"
            f"<td style='text-align:left'>{ETF_LABELS[tk]}</td>"
            f"<td>{w*100:.1f}%</td>"
            f"<td>{fmt_pct(r)}</td>"
            f"<td>{f'{v*100:.1f}%' if v is not None else '—'}</td>"
            f"<td>{fmt_sharpe(s)}</td>"
            f"</tr>"
        )
    html_etf += "</table>"
    st.markdown(html_etf, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data via Yahoo Finance (yfinance).  "
    "Volatility = annualised std of daily returns (×√252).  "
    "Sharpe = annualised excess return ÷ annualised vol, using BIL as risk-free rate.  "
    "Rolling metrics use a 30-trading-day window.  "
    "Not financial advice."
)
