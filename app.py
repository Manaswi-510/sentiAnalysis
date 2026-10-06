import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis · SVM",
    page_icon=None,
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d1117;
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}

h1 {
    color: #e6edf3 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    font-size: 2rem !important;
}
h2 {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    margin-bottom: 0.6rem !important;
}
h3 {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

.badge {
    display: inline-block;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.5rem 2rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}
.badge-positive {
    background: rgba(56, 189, 142, 0.15);
    color: #3ddc97;
    border: 1px solid rgba(61, 220, 151, 0.35);
}
.badge-negative {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}
.badge-neutral {
    background: rgba(96, 165, 250, 0.12);
    color: #60a5fa;
    border: 1px solid rgba(96, 165, 250, 0.3);
}

.conf-bar-wrap { margin-top: 0.8rem; }
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #8b949e;
}
.conf-bar-bg {
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill-pos { background: #3ddc97; height: 8px; border-radius: 4px; }
.conf-bar-fill-neu { background: #60a5fa; height: 8px; border-radius: 4px; }
.conf-bar-fill-neg { background: #f87171; height: 8px; border-radius: 4px; }
.conf-val {
    width: 3rem;
    text-align: right;
    font-weight: 600;
    color: #c9d1d9;
    font-size: 0.88rem;
}

.stTextArea textarea {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 2px rgba(31,111,235,0.25) !important;
}

.stButton > button {
    background: #1f6feb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #388bfd !important;
    box-shadow: 0 4px 14px rgba(31,111,235,0.4) !important;
}

hr { border-color: #21262d !important; margin: 1rem 0 !important; }

.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #21262d !important;
}

.sub {
    color: #8b949e;
    font-size: 0.88rem;
    margin-top: -0.3rem;
    margin-bottom: 1.2rem;
}

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    color: #58a6ff;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.4rem;
    background: #0d1117;
    border: 1px solid #21262d;
    font-size: 0.88rem;
}
.hist-phrase {
    color: #c9d1d9;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-right: 0.8rem;
}
.hist-label {
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    flex-shrink: 0;
}
.hist-pos { color: #3ddc97; background: rgba(61,220,151,0.1); border: 1px solid rgba(61,220,151,0.25); }
.hist-neg { color: #f87171; background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.25); }
.hist-neu { color: #60a5fa; background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.25); }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, sans-serif", color="#c9d1d9"),
    margin=dict(t=40, b=40, l=40, r=20),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
)
COLORS = {
    "positive": "#3ddc97",
    "neutral":  "#60a5fa",
    "negative": "#f87171",
    "blue":     "#1f6feb",
    "grid":     "#21262d",
    "axis":     "#30363d",
}
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_NUM = {"negative": 0, "neutral": 1, "positive": 2}

# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training SVM model...")
def train_model():
    data = pd.read_csv("data/sentimentAnalysis.csv", encoding="latin1")
    data["sentiment_label"] = data["sentiment"].str.lower().str.strip()
    data["sentiment"] = data["sentiment_label"].map(LABEL_NUM)

    X = data["phrase"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = SVC(kernel="rbf", random_state=1, probability=True)
    model.fit(X_train_tfidf, y_train)

    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=["Negative", "Neutral", "Positive"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, accuracy, report, cm, data

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_single(phrase, model, vectorizer):
    vec  = vectorizer.transform([phrase])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, proba

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

STOPWORDS = set("""a an the is was were are be been being have has had do does
did will would could should may might shall can this that these those i me my
myself we our ours ourselves you your yours he him his she her hers it its
they them their what which who whom when where why how all any both each
few more most other some such no nor not only own same so than too very
just but and or if in of on to at as by for with about against between
into through during before after above below from up down out off over
under again further then once""".split())

def top_words(texts, n=20):
    words = []
    for t in texts:
        words.extend([w for w in clean_text(str(t)) if w not in STOPWORDS and len(w) > 2])
    return Counter(words).most_common(n)

# ── Chart builders ────────────────────────────────────────────────────────────
def chart_class_distribution(data):
    counts = data["sentiment_label"].value_counts()
    colors = [COLORS[k] for k in counts.index]
    fig = go.Figure(go.Bar(
        x=[c.capitalize() for c in counts.index],
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=13, family="Inter"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Class Distribution", font=dict(size=15, color="#e6edf3")),
        xaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        showlegend=False,
    )
    return fig

def chart_word_length(data):
    data = data.copy()
    data["word_count"] = data["phrase"].apply(lambda x: len(str(x).split()))
    fig = go.Figure()
    for label, color in [("positive", COLORS["positive"]), ("neutral", COLORS["neutral"]), ("negative", COLORS["negative"])]:
        subset = data[data["sentiment_label"] == label]["word_count"]
        fig.add_trace(go.Histogram(x=subset, name=label.capitalize(), marker_color=color, opacity=0.75, xbins=dict(size=1)))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Word Count Distribution by Sentiment", font=dict(size=15, color="#e6edf3")),
        xaxis=dict(title="Words per Phrase", gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        yaxis=dict(title="Count", gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        barmode="overlay",
    )
    return fig

def chart_precision_recall_f1(report):
    classes = ["Negative", "Neutral", "Positive"]
    bar_colors = [COLORS["negative"], COLORS["neutral"], COLORS["positive"]]
    fig = go.Figure()
    for cls, col in zip(classes, bar_colors):
        fig.add_trace(go.Bar(
            name=cls,
            x=["Precision", "Recall", "F1-Score"],
            y=[report[cls]["precision"], report[cls]["recall"], report[cls]["f1-score"]],
            marker_color=col,
            text=[f"{v:.2f}" for v in [report[cls]["precision"], report[cls]["recall"], report[cls]["f1-score"]]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=11),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Precision / Recall / F1 by Class", font=dict(size=15, color="#e6edf3")),
        xaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor=COLORS["grid"], range=[0, 1.15], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        barmode="group",
    )
    return fig

def chart_confusion_matrix(cm):
    labels = ["Negative", "Neutral", "Positive"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#0d1117"], [0.5, "#1f6feb"], [1, "#58a6ff"]],
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#e6edf3"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Confusion Matrix", font=dict(size=15, color="#e6edf3")),
        xaxis=dict(title="Predicted", tickfont=dict(color="#8b949e"), linecolor=COLORS["axis"]),
        yaxis=dict(title="Actual", tickfont=dict(color="#8b949e"), linecolor=COLORS["axis"], autorange="reversed"),
    )
    return fig

def chart_top_words(data, sentiment_label, n=15):
    subset = data[data["sentiment_label"] == sentiment_label]["phrase"]
    words  = top_words(subset, n)
    if not words:
        return None
    terms, counts = zip(*words)
    color = COLORS.get(sentiment_label, COLORS["blue"])
    fig = go.Figure(go.Bar(
        x=list(counts)[::-1], y=list(terms)[::-1],
        orientation="h", marker_color=color,
        text=list(counts)[::-1], textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Top Words — {sentiment_label.capitalize()}", font=dict(size=15, color="#e6edf3")),
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
        yaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#c9d1d9"), automargin=True),
        margin=dict(l=100, r=20, t=40, b=30),
        height=420,
    )
    return fig

def chart_confidence_radar(proba):
    labels = ["Negative", "Neutral", "Positive"]
    fig = go.Figure(go.Scatterpolar(
        r=[*proba, proba[0]],
        theta=[*labels, labels[0]],
        fill="toself",
        fillcolor="rgba(31,111,235,0.15)",
        line=dict(color="#1f6feb", width=2),
        marker=dict(size=6, color="#58a6ff"),
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        margin=dict(t=30, b=30, l=30, r=30),
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", tickfont=dict(color="#8b949e", size=9)),
            angularaxis=dict(tickfont=dict(color="#c9d1d9", size=12), linecolor="#30363d"),
        ),
        showlegend=False,
        height=260,
    )
    return fig

# ── History state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    model, vectorizer, accuracy, report, cm, data = train_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
        page = st.selectbox(
            "Go to",
            ["Predict", "Data Analysis", "Model Report", "Batch Analysis"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="color:#8b949e;font-size:0.82rem;line-height:1.9;">
  Algorithm: <span style="color:#c9d1d9;">SVC (RBF kernel)</span><br>
  Vectorizer: <span style="color:#c9d1d9;">TF-IDF</span><br>
  Train / Test: <span style="color:#c9d1d9;">80 / 20</span><br>
  Dataset size: <span style="color:#c9d1d9;">{len(data):,} phrases</span><br>
  Accuracy: <span style="color:#3ddc97;font-weight:600;">{accuracy:.2%}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-label">Prediction History</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for item in reversed(st.session_state.history[-8:]):
                cls_map = {"Positive": "hist-pos", "Negative": "hist-neg", "Neutral": "hist-neu"}
                c = cls_map[item["label"]]
                short = item["phrase"][:38] + ("..." if len(item["phrase"]) > 38 else "")
                st.markdown(
                    f'<div class="hist-item"><span class="hist-phrase">{short}</span>'
                    f'<span class="hist-label {c}">{item["label"]}</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.markdown('<p style="color:#484f58;font-size:0.82rem;">No predictions yet.</p>', unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# Sentiment Analysis")
    st.markdown('<p class="sub">Support Vector Machine with RBF kernel and TF-IDF vectorization</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",     f"{accuracy:.2%}")
    c2.metric("Neg F1",       f"{report['Negative']['f1-score']:.3f}")
    c3.metric("Neu F1",       f"{report['Neutral']['f1-score']:.3f}")
    c4.metric("Pos F1",       f"{report['Positive']['f1-score']:.3f}")
    c5.metric("Dataset Size", f"{len(data):,}")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    #  PAGE: PREDICT
    # ════════════════════════════════════════════════════════════
    if page == "Predict":
        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("## Analyse a Phrase")
            phrase = st.text_area(
                "phrase_input",
                placeholder="Type or paste a phrase here...",
                height=130,
                label_visibility="collapsed",
            )
            if st.button("Analyse Sentiment", key="btn_analyse"):
                if not phrase.strip():
                    st.warning("Please enter a phrase before analysing.")
                else:
                    pred, proba = predict_single(phrase, model, vectorizer)
                    label = LABEL_MAP[pred]
                    badge_cls = {"Negative": "badge-negative", "Neutral": "badge-neutral", "Positive": "badge-positive"}[label]
                    st.markdown(f'<div class="badge {badge_cls}">{label}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="conf-bar-wrap">', unsafe_allow_html=True)
                    for idx, (lbl, fill_cls) in enumerate([
                        ("Negative", "conf-bar-fill-neg"),
                        ("Neutral",  "conf-bar-fill-neu"),
                        ("Positive", "conf-bar-fill-pos"),
                    ]):
                        pct = proba[idx]
                        st.markdown(
                            f'<div class="conf-row">'
                            f'<span style="width:4.5rem;color:#8b949e;">{lbl}</span>'
                            f'<div class="conf-bar-bg"><div class="{fill_cls}" style="width:{pct*100:.1f}%;"></div></div>'
                            f'<span class="conf-val">{pct:.1%}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.session_state.history.append({"phrase": phrase.strip(), "label": label, "proba": proba})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card" style="height:100%">', unsafe_allow_html=True)
            st.markdown("## Confidence Radar")
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.plotly_chart(chart_confidence_radar(last["proba"]), use_container_width=True)
                excerpt = last["phrase"][:60] + ("..." if len(last["phrase"]) > 60 else "")
                st.markdown(f'<p style="color:#484f58;font-size:0.78rem;text-align:center;">Last: "{excerpt}"</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#484f58;font-size:0.9rem;margin-top:3rem;text-align:center;">Run a prediction to see the confidence radar chart.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  PAGE: DATA ANALYSIS
    # ════════════════════════════════════════════════════════════
    elif page == "Data Analysis":
        st.markdown("## Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Distribution", "Word Analysis", "Dataset Preview"])

        with tab1:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.plotly_chart(chart_class_distribution(data), use_container_width=True)
            with col2:
                st.plotly_chart(chart_word_length(data), use_container_width=True)

            # Character count
            data_c = data.copy()
            data_c["char_count"] = data_c["phrase"].apply(lambda x: len(str(x)))
            fig_char = go.Figure()
            for label, color in [("positive", COLORS["positive"]), ("neutral", COLORS["neutral"]), ("negative", COLORS["negative"])]:
                subset = data_c[data_c["sentiment_label"] == label]["char_count"]
                fig_char.add_trace(go.Histogram(x=subset, name=label.capitalize(), marker_color=color, opacity=0.75))
            fig_char.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Character Count Distribution by Sentiment", font=dict(size=15, color="#e6edf3")),
                xaxis=dict(title="Characters per Phrase", gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                yaxis=dict(title="Count", gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                barmode="overlay",
            )
            st.plotly_chart(fig_char, use_container_width=True)

            # Average word length
            data_c["avg_word_len"] = data_c["phrase"].apply(
                lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
            )
            avgs = data_c.groupby("sentiment_label")["avg_word_len"].mean()
            fig_avg = go.Figure(go.Bar(
                x=[c.capitalize() for c in avgs.index],
                y=avgs.values,
                marker_color=[COLORS.get(k, COLORS["blue"]) for k in avgs.index],
                text=[f"{v:.2f}" for v in avgs.values],
                textposition="outside",
                textfont=dict(color="#c9d1d9", size=12),
            ))
            fig_avg.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Average Word Length by Sentiment", font=dict(size=15, color="#e6edf3")),
                xaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                showlegend=False,
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        with tab2:
            sentiment_choice = st.selectbox(
                "Select sentiment class",
                ["positive", "neutral", "negative"],
                format_func=lambda x: x.capitalize(),
            )
            fig_words = chart_top_words(data, sentiment_choice)
            if fig_words:
                st.plotly_chart(fig_words, use_container_width=True)

            st.markdown("### Vocabulary size per class")
            vocab_data = {}
            for lbl in ["positive", "neutral", "negative"]:
                subset = data[data["sentiment_label"] == lbl]["phrase"]
                all_w = []
                for t in subset:
                    all_w.extend([w for w in clean_text(str(t)) if w not in STOPWORDS and len(w) > 2])
                vocab_data[lbl.capitalize()] = len(set(all_w))
            col_v1, col_v2, col_v3 = st.columns(3)
            for col, (cls, cnt) in zip([col_v1, col_v2, col_v3], vocab_data.items()):
                col.metric(f"{cls} Vocabulary", f"{cnt:,} words")

        with tab3:
            st.markdown("### Dataset Preview")
            display = data[["phrase", "sentiment_label"]].copy()
            display.columns = ["Phrase", "Sentiment"]
            display["Sentiment"] = display["Sentiment"].str.capitalize()
            filter_sent = st.selectbox("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
            if filter_sent != "All":
                display = display[display["Sentiment"] == filter_sent]
            n_rows = st.slider("Rows to show", 10, 100, 20)
            st.dataframe(display.head(n_rows), use_container_width=True)
            st.markdown(f'<p class="sub">{len(display):,} rows shown.</p>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  PAGE: MODEL REPORT
    # ════════════════════════════════════════════════════════════
    elif page == "Model Report":
        st.markdown("## Model Report")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(chart_precision_recall_f1(report), use_container_width=True)
        with col2:
            st.plotly_chart(chart_confusion_matrix(cm), use_container_width=True)

        st.markdown("### Per-class Support")
        sup_fig = go.Figure(go.Bar(
            x=["Negative", "Neutral", "Positive"],
            y=[int(report["Negative"]["support"]), int(report["Neutral"]["support"]), int(report["Positive"]["support"])],
            marker_color=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]],
            text=[int(report["Negative"]["support"]), int(report["Neutral"]["support"]), int(report["Positive"]["support"])],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=12),
        ))
        sup_fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
            yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
            showlegend=False, height=280,
        )
        st.plotly_chart(sup_fig, use_container_width=True)

        st.markdown("### Detailed Classification Report")
        rows = []
        for cls in ["Negative", "Neutral", "Positive"]:
            r = report[cls]
            rows.append({
                "Class": cls,
                "Precision": round(r["precision"], 4),
                "Recall":    round(r["recall"], 4),
                "F1-Score":  round(r["f1-score"], 4),
                "Support":   int(r["support"]),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

        wa = report["weighted avg"]
        st.markdown("### Weighted Averages")
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Weighted Precision", f"{wa['precision']:.4f}")
        wc2.metric("Weighted Recall",    f"{wa['recall']:.4f}")
        wc3.metric("Weighted F1",        f"{wa['f1-score']:.4f}")

    # ════════════════════════════════════════════════════════════
    #  PAGE: BATCH ANALYSIS
    # ════════════════════════════════════════════════════════════
    elif page == "Batch Analysis":
        st.markdown("## Batch Analysis")
        st.markdown('<p class="sub">Paste multiple phrases (one per line) or upload a CSV file.</p>', unsafe_allow_html=True)
        tab_text, tab_file = st.tabs(["Text Input", "CSV Upload"])

        with tab_text:
            bulk_text = st.text_area(
                "batch_text",
                placeholder="Enter one phrase per line...",
                height=200,
                label_visibility="collapsed",
            )
            if st.button("Analyse All", key="btn_batch"):
                lines = [l.strip() for l in bulk_text.strip().splitlines() if l.strip()]
                if not lines:
                    st.warning("Please enter at least one phrase.")
                else:
                    results = []
                    for ph in lines:
                        pred, proba = predict_single(ph, model, vectorizer)
                        label = LABEL_MAP[pred]
                        results.append({
                            "Phrase": ph,
                            "Sentiment": label,
                            "Neg Conf": f"{proba[0]:.1%}",
                            "Neu Conf": f"{proba[1]:.1%}",
                            "Pos Conf": f"{proba[2]:.1%}",
                        })
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    counts = df_results["Sentiment"].value_counts()
                    color_map = {"Positive": COLORS["positive"], "Neutral": COLORS["neutral"], "Negative": COLORS["negative"]}
                    batch_fig = go.Figure(go.Bar(
                        x=counts.index.tolist(),
                        y=counts.values.tolist(),
                        marker_color=[color_map.get(k, COLORS["blue"]) for k in counts.index],
                        text=counts.values.tolist(),
                        textposition="outside",
                        textfont=dict(color="#c9d1d9"),
                    ))
                    batch_fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(text="Batch Sentiment Distribution", font=dict(size=15, color="#e6edf3")),
                        xaxis=dict(showgrid=False, linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], tickfont=dict(color="#8b949e")),
                        showlegend=False, height=300,
                    )
                    st.plotly_chart(batch_fig, use_container_width=True)
                    st.download_button(
                        "Download Results as CSV",
                        df_results.to_csv(index=False),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

        with tab_file:
            uploaded = st.file_uploader("Upload a CSV with a 'phrase' column", type=["csv"])
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "phrase" not in df_up.columns:
                    st.error("CSV must contain a column named 'phrase'.")
                else:
                    results = []
                    for ph in df_up["phrase"].dropna().astype(str):
                        pred, proba = predict_single(ph, model, vectorizer)
                        label = LABEL_MAP[pred]
                        results.append({
                            "Phrase": ph,
                            "Sentiment": label,
                            "Neg Conf": f"{proba[0]:.1%}",
                            "Neu Conf": f"{proba[1]:.1%}",
                            "Pos Conf": f"{proba[2]:.1%}",
                        })
                    df_results = pd.DataFrame(results)
                    st.success(f"Analysed {len(df_results):,} phrases.")
                    st.dataframe(df_results, use_container_width=True)
                    counts = df_results["Sentiment"].value_counts()
                    color_map = {"Positive": COLORS["positive"], "Neutral": COLORS["neutral"], "Negative": COLORS["negative"]}
                    pie_fig = go.Figure(go.Pie(
                        labels=counts.index.tolist(),
                        values=counts.values.tolist(),
                        marker=dict(colors=[color_map.get(k, COLORS["blue"]) for k in counts.index]),
                        hole=0.5,
                        textfont=dict(color="#e6edf3", size=13),
                    ))
                    pie_fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(text="Sentiment Distribution (Uploaded File)", font=dict(size=15, color="#e6edf3")),
                        height=320,
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
                    st.download_button(
                        "Download Results as CSV",
                        df_results.to_csv(index=False),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="text-align:center;color:#484f58;font-size:0.75rem;margin-top:2.5rem;">'
        "SVC(kernel=&#39;rbf&#39;, probability=True) &middot; TF-IDF Vectorizer &middot; 80/20 split</p>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

