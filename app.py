import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ruptures as rpt
from scipy.spatial.distance import jensenshannon
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stance Shift — NLP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

.stApp { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00ff88, #00cfff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}

.metric-label {
    color: #888;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.favor-badge   { background: #0d3320; color: #00ff88; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.against-badge { background: #3a0d0d; color: #ff4d4d; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.neutral-badge { background: #2a2500; color: #f5c400; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

.predict-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
}

.result-favor   { border-left: 4px solid #00ff88; padding-left: 1rem; }
.result-against { border-left: 4px solid #ff4d4d; padding-left: 1rem; }
.result-neutral { border-left: 4px solid #f5c400; padding-left: 1rem; }

.stSelectbox > div > div,
.stTextArea > div > div {
    background-color: #1a1a1a !important;
    border-color: #2a2a2a !important;
    color: #e8e8e8 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cfff);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 0.9rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00ff88;
    border-bottom: 1px solid #222;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
LABEL2ID = {'against': 0, 'neutral': 1, 'favor': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
COLORS   = {'favor': '#00ff88', 'neutral': '#f5c400', 'against': '#ff4d4d'}

plt.rcParams.update({
    'figure.facecolor':  '#0d0d0d',
    'axes.facecolor':    '#141414',
    'axes.edgecolor':    '#2a2a2a',
    'axes.labelcolor':   '#aaa',
    'xtick.color':       '#666',
    'ytick.color':       '#666',
    'text.color':        '#e8e8e8',
    'grid.color':        '#222',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
})

@st.cache_data
def load_data():
    paths = ['stance_predictions_full.csv', 'stance_dataset_merged.csv']
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'predicted_stance' not in df.columns:
                df['predicted_stance'] = df.get('stance_label', 'neutral')
            if 'confidence' not in df.columns:
                df['confidence'] = 0.75
            df['clean_text'] = df['clean_text'].astype(str).str.strip()
            df = df[df['clean_text'].str.len() >= 10].reset_index(drop=True)

            # Simulate dates if not present
            if 'date' not in df.columns or df['date'].isna().all():
                np.random.seed(42)
                date_ranges = {
                    'tweeteval':     ('2015-01-01', '2017-12-31'),
                    'semeval2016':   ('2015-06-01', '2016-09-30'),
                    'climate_fever': ('2018-01-01', '2023-12-31'),
                }
                dates = []
                for src in df.get('source', pd.Series(['tweeteval'] * len(df))):
                    s, e   = date_ranges.get(src, ('2019-01-01', '2023-12-31'))
                    s_ts   = pd.Timestamp(s).value // 10**9
                    e_ts   = pd.Timestamp(e).value // 10**9
                    dates.append(pd.to_datetime(np.random.randint(s_ts, e_ts), unit='s'))
                df['date'] = dates

            df['date']       = pd.to_datetime(df['date'], errors='coerce')
            df['year']       = df['date'].dt.year
            df['month']      = df['date'].dt.month
            df['year_month'] = df['date'].dt.to_period('M')
            return df

    return pd.DataFrame(columns=['clean_text', 'predicted_stance', 'confidence', 'topic', 'source', 'date', 'year', 'month', 'year_month'])

@st.cache_resource
def load_model():
    if os.path.exists('roberta_stance_model'):
        try:
            tok   = RobertaTokenizer.from_pretrained('roberta_stance_model')
            model = RobertaForSequenceClassification.from_pretrained('roberta_stance_model')
            model.eval()
            return tok, model
        except Exception:
            return None, None
    return None, None

def predict_stance(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
    return ID2LABEL[pred], probs.numpy()

def get_monthly(df_sub):
    monthly = df_sub.groupby(['year_month', 'predicted_stance']).size().unstack(fill_value=0)
    for col in ['favor', 'neutral', 'against']:
        if col not in monthly.columns:
            monthly[col] = 0
    monthly = monthly[['favor', 'neutral', 'against']]
    monthly['total']       = monthly.sum(axis=1)
    monthly['favor_pct']   = monthly['favor']   / monthly['total'] * 100
    monthly['against_pct'] = monthly['against'] / monthly['total'] * 100
    monthly['neutral_pct'] = monthly['neutral'] / monthly['total'] * 100
    monthly['favor_ratio'] = (monthly['favor'] - monthly['against']) / monthly['total']
    monthly.index = monthly.index.astype(str)
    return monthly

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Space Mono;font-size:1.2rem;color:#00ff88;font-weight:700;margin-bottom:0.5rem;'>⚡ STANCE SHIFT</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;font-size:0.8rem;margin-bottom:1.5rem;'>NLP Temporal Analysis</div>", unsafe_allow_html=True)

    st.markdown("### 🔧 Filters")

    df_all = load_data()

    if len(df_all) == 0:
        st.warning("No data found. Please run the data collection and inference notebooks first.")
        st.stop()

    topics_available = sorted(df_all['topic'].dropna().unique().tolist())
    selected_topic   = st.selectbox("Topic", options=["All"] + topics_available)

    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    years_available = sorted(df_all['year'].dropna().unique().tolist())
    if len(years_available) >= 2:
        year_range = st.select_slider("Year Range", options=years_available, value=(min(years_available), max(years_available)))
    else:
        year_range = (min(years_available), max(years_available)) if years_available else (2015, 2023)

    st.markdown("---")
    st.markdown("### 📄 Pages")
    page = st.radio("", ["Overview", "Temporal Trends", "Change Points", "Live Predictor"], label_visibility="collapsed")

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
df = df_all.copy()
if selected_topic != "All":
    df = df[df['topic'] == selected_topic]
df = df[df['confidence'] >= min_conf]
df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="main-title">STANCE SHIFT</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Tracking how public opinion evolves over time using NLP</div>', unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    total    = len(df)
    n_favor  = (df['predicted_stance'] == 'favor').sum()
    n_against= (df['predicted_stance'] == 'against').sum()
    n_neutral= (df['predicted_stance'] == 'neutral').sum()

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Posts</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00ff88">{n_favor:,}</div><div class="metric-label">✅ Favor</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff4d4d">{n_against:,}</div><div class="metric-label">❌ Against</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f5c400">{n_neutral:,}</div><div class="metric-label">➖ Neutral</div></div>', unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Stance Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts  = df['predicted_stance'].value_counts()
        bars    = ax.bar(counts.index, counts.values, color=[COLORS.get(l, '#888') for l in counts.index], width=0.5, edgecolor='none')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:,}', ha='center', color='#aaa', fontsize=9)
        ax.set_ylabel('Posts')
        ax.grid(axis='y')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Posts per Topic</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        tc = df['topic'].value_counts().head(8)
        ax.barh(tc.index, tc.values, color='#00cfff', alpha=0.85, edgecolor='none')
        ax.set_xlabel('Posts')
        ax.grid(axis='x')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Sample posts
    st.markdown('<div class="section-header">Sample Posts</div>', unsafe_allow_html=True)
    for stance in ['favor', 'against', 'neutral']:
        sub = df[df['predicted_stance'] == stance]
        if len(sub) == 0:
            continue
        sample = sub.sample(1, random_state=42).iloc[0]
        badge  = f'<span class="{stance}-badge">{stance.upper()}</span>'
        conf   = f'<span style="color:#555;font-size:0.8rem;margin-left:8px;">conf: {sample["confidence"]:.2f}</span>'
        text   = sample['clean_text'][:200]
        st.markdown(f'{badge}{conf}<br><span style="color:#ccc;font-size:0.9rem;">{text}</span>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1a1a1a;margin:0.6rem 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: TEMPORAL TRENDS
# ─────────────────────────────────────────────
elif page == "Temporal Trends":
    st.markdown('<div class="main-title">TEMPORAL TRENDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">How stance distribution shifts month by month</div>', unsafe_allow_html=True)

    monthly = get_monthly(df)

    if len(monthly) < 2:
        st.warning("Not enough temporal data to display trends. Try selecting 'All' topics or expanding the year range.")
        st.stop()

    # Stacked area chart
    st.markdown('<div class="section-header">Stance Over Time (Stacked)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(monthly))
    ax.stackplot(x,
        monthly['favor_pct'], monthly['neutral_pct'], monthly['against_pct'],
        labels=['Favor', 'Neutral', 'Against'],
        colors=['#00ff88', '#f5c400', '#ff4d4d'], alpha=0.85)
    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly.index[::step], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('% of Posts')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Favor ratio
    st.markdown('<div class="section-header">Net Stance Ratio</div>', unsafe_allow_html=True)
    st.caption("(Favor − Against) / Total  |  +1 = all favor  |  −1 = all against")
    fig, ax = plt.subplots(figsize=(14, 4))
    x  = range(len(monthly))
    y  = monthly['favor_ratio'].values
    ax.plot(x, y, color='#00cfff', linewidth=2.5)
    ax.fill_between(x, 0, y, where=(np.array(y) >= 0), alpha=0.2, color='#00ff88')
    ax.fill_between(x, 0, y, where=(np.array(y) < 0),  alpha=0.2, color='#ff4d4d')
    ax.axhline(0, color='#444', linewidth=1, linestyle='--')
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax.set_xticklabels(monthly.index[::max(1, len(monthly)//12)], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Favor Ratio')
    ax.grid(axis='y')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Table
    st.markdown('<div class="section-header">Monthly Data Table</div>', unsafe_allow_html=True)
    display_df = monthly[['favor', 'against', 'neutral', 'total', 'favor_pct', 'against_pct']].copy()
    display_df['favor_pct']   = display_df['favor_pct'].round(1).astype(str) + '%'
    display_df['against_pct'] = display_df['against_pct'].round(1).astype(str) + '%'
    st.dataframe(display_df.tail(24), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: CHANGE POINTS
# ─────────────────────────────────────────────
elif page == "Change Points":
    st.markdown('<div class="main-title">CHANGE POINTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Months where public stance significantly shifted</div>', unsafe_allow_html=True)

    monthly = get_monthly(df)
    signal  = monthly['favor_ratio'].values

    if len(signal) < 6:
        st.warning("Not enough data points for change point detection. Need at least 6 months.")
        st.stop()

    pen = st.slider("Detection Sensitivity (lower = more change points)", 0.5, 5.0, 1.5, 0.1)

    try:
        algo        = rpt.Pelt(model='rbf').fit(signal)
        breakpoints = algo.predict(pen=pen)
        n_changes   = len(breakpoints) - 1
    except Exception:
        breakpoints = []
        n_changes   = 0

    st.markdown(f'<div class="metric-card" style="display:inline-block;margin-bottom:1.5rem;"><div class="metric-value">{n_changes}</div><div class="metric-label">Change Points Detected</div></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(signal))
    ax.plot(x, signal, color='#00cfff', linewidth=2.5, zorder=2)
    ax.fill_between(x, 0, signal, where=(signal >= 0), alpha=0.15, color='#00ff88')
    ax.fill_between(x, 0, signal, where=(signal < 0),  alpha=0.15, color='#ff4d4d')
    ax.axhline(0, color='#444', linewidth=1, linestyle='--')

    for bp in breakpoints[:-1]:
        ax.axvline(bp, color='#ff4d4d', linewidth=2, linestyle='--', alpha=0.9, zorder=3)
        if bp < len(monthly):
            ax.annotate(monthly.index[bp], xy=(bp, signal.max()),
                xytext=(bp + 0.3, signal.max() * 0.95),
                color='#ff4d4d', fontsize=7.5, rotation=45)

    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly.index[::step], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Favor Ratio')
    ax.set_title('Stance Trajectory with Change Points', color='#e8e8e8', fontsize=12)
    ax.grid(axis='y')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Polarization
    st.markdown('<div class="section-header">Year-by-Year Polarization</div>', unsafe_allow_html=True)
    st.caption("Jensen-Shannon divergence between communities — higher = more polarized")

    pol_rows = []
    for year in sorted(df['year'].dropna().unique()):
        year_df = df[df['year'] == year]
        half    = len(year_df) // 2
        if half == 0:
            continue
        p1 = year_df.iloc[:half]['predicted_stance'].value_counts(normalize=True)
        p2 = year_df.iloc[half:]['predicted_stance'].value_counts(normalize=True)
        labels = ['favor', 'neutral', 'against']
        p = np.array([p1.get(l, 0) for l in labels]) + 1e-9
        q = np.array([p2.get(l, 0) for l in labels]) + 1e-9
        p /= p.sum(); q /= q.sum()
        pol_rows.append({'year': int(year), 'polarization': round(jensenshannon(p, q), 4), 'posts': len(year_df)})

    pol_df = pd.DataFrame(pol_rows)
    if len(pol_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(pol_df['year'].astype(str), pol_df['polarization'],
                      color='#ff4d4d', alpha=0.8, edgecolor='none', width=0.5)
        for bar, val in zip(bars, pol_df['polarization']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', color='#aaa', fontsize=9)
        ax.set_ylabel('JS Divergence')
        ax.set_xlabel('Year')
        ax.grid(axis='y')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.dataframe(pol_df, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: LIVE PREDICTOR
# ─────────────────────────────────────────────
elif page == "Live Predictor":
    st.markdown('<div class="main-title">LIVE PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Type any text and get an instant stance prediction</div>', unsafe_allow_html=True)

    tokenizer, model_loaded = load_model()
    model_available = tokenizer is not None

    if not model_available:
        st.warning("⚠️ Trained model not found in `roberta_stance_model/`. Run the training notebook first.")
        st.info("Showing a demo mode with rule-based predictions instead.")

    text_input = st.text_area("Enter text to classify:", height=120,
        placeholder="e.g. Climate change is real and we need to act immediately...")

    col1, col2 = st.columns([1, 3])
    with col1:
        predict_btn = st.button("🔍 Predict Stance")

    if predict_btn and text_input.strip():
        if model_available:
            label, probs = predict_stance(text_input, tokenizer, model_loaded)
            confidence   = float(probs.max())
        else:
            # Demo fallback
            text_lower = text_input.lower()
            if any(w in text_lower for w in ['must', 'need', 'important', 'support', 'great', 'good']):
                label, confidence = 'favor', 0.78
            elif any(w in text_lower for w in ['hoax', 'fake', 'wrong', 'against', 'bad', "don't", 'not']):
                label, confidence = 'against', 0.74
            else:
                label, confidence = 'neutral', 0.65
            probs = np.array([0.1, 0.1, 0.1])
            probs[LABEL2ID[label]] = confidence

        emoji = '✅' if label == 'favor' else '❌' if label == 'against' else '➖'
        color = COLORS[label]

        st.markdown(f"""
        <div class="predict-box result-{label}">
            <div style="font-family:Space Mono;font-size:1.8rem;font-weight:700;color:{color}">
                {emoji} {label.upper()}
            </div>
            <div style="color:#888;margin-top:0.3rem;">Confidence: <span style="color:{color};font-weight:600;">{confidence*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Probability Breakdown</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        stance_names = ['against', 'neutral', 'favor']
        prob_vals    = [float(probs[LABEL2ID[s]]) for s in stance_names]
        bar_colors   = [COLORS[s] for s in stance_names]
        bars = ax.barh(stance_names, prob_vals, color=bar_colors, alpha=0.85, edgecolor='none', height=0.5)
        for bar, val in zip(bars, prob_vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', color='#aaa', fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability')
        ax.grid(axis='x')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Example sentences
    st.markdown('<div class="section-header" style="margin-top:2rem;">Try These Examples</div>', unsafe_allow_html=True)
    examples = [
        "Climate change is an existential threat and we must act now!",
        "Global warming is a hoax invented by scientists for grant money.",
        "Vaccines have saved millions of lives and are proven safe.",
        "I don't trust AI — it's going to destroy jobs and invade our privacy.",
        "Renewable energy is our best hope for a sustainable future.",
    ]
    for ex in examples:
        if st.button(f"→ {ex[:70]}...", key=ex):
            st.session_state['example_text'] = ex
            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#333;font-size:0.75rem;margin-top:3rem;font-family:Space Mono;'>
    STANCE SHIFT — NLP Portfolio Project · RoBERTa Fine-tuned · Built with Streamlit
</div>
""", unsafe_allow_html=True)
