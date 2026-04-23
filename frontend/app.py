"""
PRISM — Streamlit Frontend
Modern dark-mode data intelligence dashboard.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRISM — Agentic Analyst",
    layout="wide",
    page_icon="🔮",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Hero header ── */
.prism-hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 40%, #0d1117 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.prism-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.prism-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px 0;
}
.prism-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
}

/* ── Stat cards ── */
.stat-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #6366f1; }
.stat-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.stat-value {
    color: #f1f5f9;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
}
.stat-sub {
    color: #475569;
    font-size: 0.75rem;
    margin-top: 4px;
}

/* ── Quality pill ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.pill-green  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-yellow { background: #422006; color: #fbbf24; border: 1px solid #92400e; }
.pill-red    { background: #3b0a0a; color: #f87171; border: 1px solid #991b1b; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e293b;
    gap: 0;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8; }
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 2px dashed #1e293b;
    border-radius: 12px;
    padding: 8px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin: 20px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}

/* ── Narrative box ── */
.narrative-box {
    background: linear-gradient(135deg, #0f172a, #0d1117);
    border: 1px solid #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    line-height: 1.75;
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* ── Test result cards ── */
.test-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.test-card-sig   { border-left: 3px solid #4ade80; }
.test-card-insig { border-left: 3px solid #f43f5e; }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #1e293b !important;
    border-radius: 10px;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    width: 100%;
    transition: all 0.15s;
}
.stDownloadButton > button:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px !important;
}
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.75rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #1e293b; }

/* ── Status ── */
[data-testid="stStatusWidget"] {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_chart(spec: dict, key: str = ""):
    try:
        pj = spec.get("plotly_json", "{}")
        fig = pio.from_json(pj if isinstance(pj, str) else json.dumps(pj))
        chart_key = key or f"chart_{id(spec)}_{spec.get('chart_type', 'unknown')}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key, config={"displayModeBar": False})
        if spec.get("description"):
            st.caption(f"↑ {spec['description']}")
    except Exception as exc:
        st.warning(f"Chart render error: {exc}")


def pill(value: float, low: float, high: float, fmt: str = "{:.1f}") -> str:
    label = fmt.format(value)
    if value < low:
        return f'<span class="pill pill-green">{label}</span>'
    if value < high:
        return f'<span class="pill pill-yellow">{label}</span>'
    return f'<span class="pill pill-red">{label}</span>'


def section(icon: str, label: str):
    st.markdown(f'<div class="section-header">{icon} {label}</div>', unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="prism-hero">
  <p class="prism-title">🔮 PRISM</p>
  <p class="prism-subtitle">Agentic Data Intelligence &nbsp;·&nbsp; Upload a CSV and the agents take over</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your CSV here",
    type=["csv"],
    label_visibility="collapsed",
    help="Profile Agent → Stat Agent → Chart Agent run automatically",
)

if uploaded and "session_id" not in st.session_state:
    with st.status("Running pipeline…", expanded=True) as status:
        st.write("📋 **Profile Agent** — inspecting columns, cleaning data…")
        st.write("📊 **Stat Agent** — running correlation, t-tests, ANOVA…")
        st.write("🎨 **Chart Agent** — generating visualisations…")
        st.write("💡 **Insight Agent** — generating & testing data hypotheses…")

        res = requests.post(
            f"{BACKEND}/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            timeout=300,
        )
        if res.status_code == 200:
            data = res.json()
            st.session_state.update({
                "session_id":     data["session_id"],
                "profile":        data.get("profile", {}),
                "cleaning":       data.get("cleaning", {}),
                "charts":         data.get("charts", []),
                "stats":          data.get("stat_results", []),
                "time_series":    data.get("time_series", []),
                "narrative":      data.get("narrative", ""),
                "quality_score":  data.get("quality_score", {}),
                "insights":       data.get("insights", []),
                "errors":         data.get("errors", []),
                "messages":       [],
                "df_preview":     pd.read_csv(uploaded),
            })
            status.update(label="✅ Analysis complete", state="complete", expanded=False)
        else:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Backend error {res.status_code}: {res.text[:400]}")

# ── No session yet ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📋", "Profile Agent", "Inspects every column, detects issues, auto-cleans"),
        (c2, "📊", "Stat Agent", "Pearson, chi-square, t-test, ANOVA, time trends"),
        (c3, "🎨", "Chart Agent", "Picks and generates the right charts with self-review"),
        (c4, "💬", "Chat Agent", "Answer any question, run custom queries, export files"),
    ]:
        col.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:22px 16px;">
          <div style="font-size:1.8rem;margin-bottom:10px">{icon}</div>
          <div style="font-weight:600;color:#e2e8f0;margin-bottom:6px">{title}</div>
          <div style="color:#475569;font-size:0.78rem">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Session data ──────────────────────────────────────────────────────────────
profile       = st.session_state["profile"]
cleaning      = st.session_state["cleaning"]
charts        = st.session_state["charts"]
stats         = st.session_state["stats"]
quality_score = st.session_state.get("quality_score", {})
insights      = st.session_state.get("insights", [])
shape         = profile.get("shape", [0, 0])

tabs = st.tabs(["  Overview  ", "  Data Quality  ", "  Distributions  ",
                "  Correlations  ", "  Statistics  ", "  Insights  ", "  Chat & Export  "])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{shape[0]:,}")
    c2.metric("Columns", shape[1])
    c3.metric("Numeric", len(profile.get("numeric_cols", [])))
    c4.metric("Categorical", len(profile.get("categorical_cols", [])))
    c5.metric("Date cols", len(profile.get("date_cols", [])))

    if st.session_state.get("narrative"):
        st.markdown("<br>", unsafe_allow_html=True)
        section("📝", "Analysis Summary")
        st.markdown(
            f'<div class="narrative-box">{st.session_state["narrative"]}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("errors"):
        with st.expander(f"⚠️ {len(st.session_state['errors'])} pipeline warning(s)", expanded=False):
            for e in st.session_state["errors"]:
                st.warning(e)

    section("📄", "Data Preview")
    st.dataframe(
        st.session_state["df_preview"].head(25),
        use_container_width=True,
        height=380,
    )

# ── TAB 2: Data Quality ───────────────────────────────────────────────────────
with tabs[1]:
    # ── Quality Score panel ───────────────────────────────────────────────────
    if quality_score:
        qs_overall = quality_score.get("overall", 0)
        qs_grade   = quality_score.get("grade", "?")
        qs_verdict = quality_score.get("verdict", "")
        grade_color = {"A": "#4ade80", "B": "#a3e635", "C": "#facc15", "D": "#f87171"}.get(qs_grade, "#94a3b8")

        section("🏅", "Data Quality Score")
        st.markdown(f"""
        <div class="stat-card" style="display:flex;align-items:center;gap:28px;padding:22px 28px;margin-bottom:12px">
          <div style="text-align:center;min-width:90px">
            <div style="font-size:3rem;font-weight:800;color:{grade_color};line-height:1">{qs_overall}</div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:2px">/ 100</div>
          </div>
          <div style="border-left:1px solid #1e293b;padding-left:24px;flex:1">
            <div style="font-size:2rem;font-weight:700;color:{grade_color};line-height:1">Grade {qs_grade}</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px">{qs_verdict}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        for col, label, key in [
            (d1, "Completeness", "completeness"),
            (d2, "Uniqueness",   "uniqueness"),
            (d3, "Validity",     "validity"),
            (d4, "Consistency",  "consistency"),
        ]:
            val = quality_score.get(key, 0)
            bar_color = "#4ade80" if val >= 90 else "#facc15" if val >= 70 else "#f87171"
            col.markdown(f"""
            <div class="stat-card" style="padding:14px 16px">
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="font-size:1.5rem;color:{bar_color}">{val}</div>
              <div style="background:#1e293b;border-radius:4px;height:6px;margin-top:8px">
                <div style="background:{bar_color};width:{val}%;height:6px;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    section("🔍", "Data Health")

    null_pcts    = profile.get("null_pcts", {})
    max_null     = max(null_pcts.values(), default=0)
    dupe_pct     = profile.get("duplicate_pct", 0)
    outlier_total = sum(
        v.get("count", v) if isinstance(v, dict) else v
        for v in profile.get("outlier_flags", {}).values()
    )

    qc1, qc2, qc3 = st.columns(3)
    qc1.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Max Null Rate</div>
      <div class="stat-value">{max_null:.1f}%</div>
      <div class="stat-sub">{pill(max_null, 5, 20)} across all columns</div>
    </div>""", unsafe_allow_html=True)
    qc2.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Duplicates</div>
      <div class="stat-value">{dupe_pct:.1f}%</div>
      <div class="stat-sub">{pill(dupe_pct, 1, 5)} · {profile.get('duplicate_count', 0)} rows</div>
    </div>""", unsafe_allow_html=True)
    qc3.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Outlier Rows</div>
      <div class="stat-value">{outlier_total:,}</div>
      <div class="stat-sub">{pill(float(outlier_total), 0.001, 100, "{:.0f} rows flagged")}</div>
    </div>""", unsafe_allow_html=True)

    if cleaning:
        section("✅", "Cleaning Actions")
        actions = []
        for col, info in cleaning.get("nulls_fixed", {}).items():
            if isinstance(info, dict):
                actions.append(f"**{col}** — {info.get('count_fixed', '?')} nulls → `{info.get('strategy', '?')}`")
        if cleaning.get("duplicates_removed", 0) > 0:
            actions.append(f"**{cleaning['duplicates_removed']} duplicate rows** removed")
        for col, info in cleaning.get("dtypes_fixed", {}).items():
            if isinstance(info, dict):
                actions.append(f"**{col}** — dtype `{info.get('from', '?')}` → `{info.get('to', '?')}`")
        for a in actions:
            st.markdown(f"› {a}")
        rb, ra = cleaning.get("rows_before", shape[0]), cleaning.get("rows_after", shape[0])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card" style="display:flex;align-items:center;gap:20px">
          <div>
            <div class="stat-label">Before</div>
            <div class="stat-value" style="font-size:1.2rem">{rb:,}</div>
          </div>
          <div style="color:#6366f1;font-size:1.4rem">→</div>
          <div>
            <div class="stat-label">After</div>
            <div class="stat-value" style="font-size:1.2rem;color:#4ade80">{ra:,}</div>
          </div>
          <div style="margin-left:auto;color:#475569;font-size:0.8rem">rows</div>
        </div>""", unsafe_allow_html=True)

    missing_chart = next((c for c in charts if c.get("chart_type") == "missing_heatmap"), None)
    if missing_chart:
        section("🗺️", "Missing Value Map")
        render_chart(missing_chart)

# ── TAB 3: Distributions ──────────────────────────────────────────────────────
with tabs[2]:
    hist_charts   = [c for c in charts if c.get("chart_type") == "histogram"]
    box_charts    = [c for c in charts if c.get("chart_type") == "box"]
    violin_charts = [c for c in charts if c.get("chart_type") == "violin"]

    if hist_charts:
        section("📈", "Distributions")
        cols = st.columns(2)
        for i, chart in enumerate(hist_charts):
            with cols[i % 2]: render_chart(chart)

    if box_charts:
        section("📦", "Box Plots — Outlier View")
        cols = st.columns(2)
        for i, chart in enumerate(box_charts):
            with cols[i % 2]: render_chart(chart)

    if violin_charts:
        section("🎻", "Violin Plots")
        cols = st.columns(2)
        for i, chart in enumerate(violin_charts):
            with cols[i % 2]: render_chart(chart)

    if not (hist_charts or box_charts or violin_charts):
        st.info("No distribution charts were generated for this dataset.")

# ── TAB 4: Correlations ───────────────────────────────────────────────────────
with tabs[3]:
    heatmap = next((c for c in charts if c.get("chart_type") == "heatmap"), None)
    if heatmap:
        section("🔥", "Correlation Heatmap")
        render_chart(heatmap)

    scatter_charts = [c for c in charts if c.get("chart_type") == "scatter"]
    if scatter_charts:
        section("🔵", "Scatter Plots")
        cols = st.columns(2)
        for i, chart in enumerate(scatter_charts):
            with cols[i % 2]: render_chart(chart)

    pairplot = next((c for c in charts if c.get("chart_type") == "pairplot"), None)
    if pairplot:
        section("🔷", "Pair Plot")
        render_chart(pairplot)

    line_charts = [c for c in charts if c.get("chart_type") == "line"]
    if line_charts:
        section("📅", "Time Series Trends")
        for chart in line_charts: render_chart(chart)

    if not any([heatmap, scatter_charts, pairplot, line_charts]):
        st.info("No correlation charts available — need ≥ 2 numeric columns.")

# ── TAB 5: Statistics ─────────────────────────────────────────────────────────
with tabs[4]:
    bar_charts = [c for c in charts if c.get("chart_type") == "bar"]
    if bar_charts:
        section("📊", "Group Comparisons")
        cols = st.columns(2)
        for i, chart in enumerate(bar_charts):
            with cols[i % 2]: render_chart(chart)

    if stats:
        section("🔬", "Statistical Test Results")
        sig   = [s for s in stats if s.get("significant")]
        insig = [s for s in stats if not s.get("significant")]

        sc1, sc2 = st.columns(2)
        sc1.markdown(f"""
        <div class="stat-card" style="text-align:center">
          <div class="stat-label">Significant findings</div>
          <div class="stat-value" style="color:#4ade80">{len(sig)}</div>
        </div>""", unsafe_allow_html=True)
        sc2.markdown(f"""
        <div class="stat-card" style="text-align:center">
          <div class="stat-label">Not significant</div>
          <div class="stat-value" style="color:#64748b">{len(insig)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        for s in stats:
            sig_flag = s.get("significant", False)
            cls  = "test-card-sig" if sig_flag else "test-card-insig"
            badge = '<span class="pill pill-green">✓ Significant</span>' if sig_flag else '<span class="pill pill-red">✗ Not Significant</span>'
            col_b = f" & {s['col_b']}" if s.get("col_b") else ""
            st.markdown(f"""
            <div class="test-card {cls}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-weight:600;color:#e2e8f0;font-size:0.85rem">
                  {s.get('test_name','?').upper()} &nbsp;·&nbsp; {s.get('col_a','?')}{col_b}
                </span>
                {badge}
              </div>
              <div style="color:#94a3b8;font-size:0.82rem;line-height:1.5">
                {s.get('interpretation','—')}
              </div>
              <div style="color:#475569;font-size:0.75rem;margin-top:6px">
                p = {s.get('p_value','?')} &nbsp;·&nbsp; stat = {s.get('statistic','?')}
                {'&nbsp;·&nbsp; effect = ' + str(s['effect_size']) if s.get('effect_size') is not None else ''}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No statistical results available yet.")

# ── TAB 6: Insights ──────────────────────────────────────────────────────────
with tabs[5]:
    section("💡", "Autonomous Data Insights")

    if not insights:
        st.info("No insights were generated for this dataset. "
                "This may happen when the data is too sparse or the pipeline skipped the Insight Agent.")
    else:
        confirmed   = [i for i in insights if i.get("verdict") == "confirmed"]
        refuted     = [i for i in insights if i.get("verdict") == "refuted"]
        inconclusive = [i for i in insights if i.get("verdict") == "inconclusive"]

        ia1, ia2, ia3 = st.columns(3)
        ia1.markdown(f"""
        <div class="stat-card" style="text-align:center">
          <div class="stat-label">Confirmed</div>
          <div class="stat-value" style="color:#4ade80">{len(confirmed)}</div>
        </div>""", unsafe_allow_html=True)
        ia2.markdown(f"""
        <div class="stat-card" style="text-align:center">
          <div class="stat-label">Refuted</div>
          <div class="stat-value" style="color:#f87171">{len(refuted)}</div>
        </div>""", unsafe_allow_html=True)
        ia3.markdown(f"""
        <div class="stat-card" style="text-align:center">
          <div class="stat-label">Inconclusive</div>
          <div class="stat-value" style="color:#facc15">{len(inconclusive)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        _verdict_color  = {"confirmed": "#4ade80", "refuted": "#f87171", "inconclusive": "#facc15"}
        _verdict_icon   = {"confirmed": "✓", "refuted": "✗", "inconclusive": "~"}

        for idx, ins in enumerate(insights):
            verdict  = ins.get("verdict", "inconclusive")
            conf     = ins.get("confidence", 0.0)
            color    = _verdict_color.get(verdict, "#94a3b8")
            icon     = _verdict_icon.get(verdict, "?")
            conf_bar = int(conf * 100)

            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:14px;padding:18px 22px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;line-height:1.4;flex:1;padding-right:16px">
                  {idx + 1}. {ins.get("hypothesis", "")}
                </div>
                <div style="text-align:right;min-width:100px">
                  <div style="color:{color};font-weight:700;font-size:0.85rem">{icon} {verdict.upper()}</div>
                  <div style="color:#475569;font-size:0.72rem;margin-top:2px">confidence {conf_bar}%</div>
                </div>
              </div>
              <div style="background:#1e293b;border-radius:4px;height:4px;margin-bottom:12px">
                <div style="background:{color};width:{conf_bar}%;height:4px;border-radius:4px"></div>
              </div>
              <div style="color:#cbd5e1;font-size:0.84rem;line-height:1.55;margin-bottom:8px">
                {ins.get("finding", "")}
              </div>
              {f'<div style="color:#6366f1;font-size:0.78rem;font-family:monospace;background:#0f172a;padding:6px 10px;border-radius:6px;margin-top:6px">{ins["supporting_stat"]}</div>' if ins.get("supporting_stat") else ""}
            </div>""", unsafe_allow_html=True)

            with st.expander(f"View test code — insight {idx + 1}", expanded=False):
                st.code(ins.get("test_code", ""), language="python")

# ── TAB 7: Chat & Export ──────────────────────────────────────────────────────
with tabs[6]:
    chat_col, export_col = st.columns([3, 1], gap="large")

    with chat_col:
        section("💬", "Chat Agent — Ask Anything")

        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("chart"):
                    render_chart(msg["chart"])

        if prompt := st.chat_input("Ask about your data…"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                res = requests.post(
                    f"{BACKEND}/chat",
                    json={"session_id": st.session_state["session_id"], "message": prompt},
                    timeout=300,
                )
            if res.status_code == 200:
                data = res.json()
                reply: dict = {"role": "assistant", "content": data.get("response", "")}
                if data.get("new_charts"):
                    reply["chart"] = data["new_charts"][0]
                st.session_state["messages"].append(reply)
                st.rerun()
            else:
                st.error(f"Chat error {res.status_code}: {res.text[:300]}")

    with export_col:
        section("⬇️", "Exports")
        sid = st.session_state["session_id"]

        for export_type, label, icon, mime, fname in [
            ("csv",        "Cleaned Dataset",  "🗂️", "text/csv",         "prism_clean.csv"),
            ("charts_zip", "All Charts (PNG)", "🖼️", "application/zip",  "prism_charts.zip"),
            ("pdf",        "Full Report",       "📄", "application/pdf", "prism_report.pdf"),
        ]:
            st.markdown(f"""
            <div style="color:#64748b;font-size:0.75rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;
                        margin-top:14px;margin-bottom:6px">{icon} {label}</div>
            """, unsafe_allow_html=True)
            r = requests.get(f"{BACKEND}/export/{sid}/{export_type}", timeout=120)
            if r.status_code == 200:
                st.download_button(
                    f"Download {label}",
                    r.content,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{export_type}",
                )
            else:
                st.caption("Not ready yet")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  New Analysis", use_container_width=True):
            for k in ["session_id","profile","cleaning","charts","stats",
                      "time_series","narrative","quality_score","insights",
                      "errors","messages","df_preview"]:
                st.session_state.pop(k, None)
            st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 16px">
      <div style="font-size:1.1rem;font-weight:700;color:#a5b4fc">🔮 PRISM</div>
      <div style="color:#475569;font-size:0.75rem">Agentic Analyst</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if "profile" in st.session_state:
        p = st.session_state["profile"]
        c = st.session_state.get("cleaning", {})
        shape = p.get("shape", [0, 0])

        st.markdown('<div class="section-header">📋 Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
          <div class="stat-card" style="padding:10px 12px">
            <div class="stat-label">Rows</div>
            <div style="font-weight:700;color:#e2e8f0;font-size:1rem">{shape[0]:,}</div>
          </div>
          <div class="stat-card" style="padding:10px 12px">
            <div class="stat-label">Cols</div>
            <div style="font-weight:700;color:#e2e8f0;font-size:1rem">{shape[1]}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if c.get("rows_after"):
            st.caption(f"After clean: **{c['rows_after']:,}** rows")

        st.divider()
        st.markdown('<div class="section-header">🗂️ Columns</div>', unsafe_allow_html=True)
        for col in p.get("numeric_cols", []):
            st.markdown(f"<span style='color:#64748b;font-size:0.7rem'>🔢</span> `{col}`", unsafe_allow_html=True)
        for col in p.get("categorical_cols", []):
            st.markdown(f"<span style='color:#64748b;font-size:0.7rem'>🔤</span> `{col}`", unsafe_allow_html=True)
        for col in p.get("date_cols", []):
            st.markdown(f"<span style='color:#64748b;font-size:0.7rem'>📅</span> `{col}`", unsafe_allow_html=True)

        skew = p.get("skewness", {})
        if skew:
            st.divider()
            st.markdown('<div class="section-header">📐 Skewness</div>', unsafe_allow_html=True)
            for col, val in skew.items():
                flag = "⚠️" if abs(val) > 1 else "✅"
                st.markdown(
                    f"<span style='font-size:0.78rem;color:#94a3b8'>{flag} `{col}`: {val:.2f}</span>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown('<div class="section-header">🔑 Session</div>', unsafe_allow_html=True)
        st.code(st.session_state.get("session_id", "")[:16] + "…", language=None)
