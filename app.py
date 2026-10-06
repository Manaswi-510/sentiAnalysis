import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis · SVM",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Card-style containers */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
}

/* Title */
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
h2, h3 {
    color: #e0e0ff !important;
    font-weight: 600 !important;
}

/* Metric labels */
[data-testid="metric-container"] label {
    color: #a0a0cc !important;
    font-size: 0.82rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* Sentiment result badges */
.badge-positive {
    display: inline-block;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    margin-top: 0.6rem;
    box-shadow: 0 4px 20px rgba(56,239,125,0.35);
}
.badge-negative {
    display: inline-block;
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    margin-top: 0.6rem;
    box-shadow: 0 4px 20px rgba(231,76,60,0.35);
}
.badge-neutral {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    margin-top: 0.6rem;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102,126,234,0.4) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.45) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* Dataframe / table */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Sub-text */
.sub {
    color: #a0a0cc;
    font-size: 0.9rem;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training SVM model…")
def train_model():
    data = pd.read_csv(
        "data/sentimentAnalysis.csv",
        encoding='latin1'
    )

    # --- same mapping as notebook ---
    data['sentiment'] = data['sentiment'].map({
        'negative': 0,
        'neutral':  1,
        'positive': 2
    })

    X = data['phrase']
    y = data['sentiment']

    # --- same split as notebook ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    # --- same vectorizer as notebook ---
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # --- EXACT same model as notebook (NOT CHANGED) ---
    model = SVC(kernel='rbf', random_state=1)
    model.fit(X_train_tfidf, y_train)

    # --- evaluation ---
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )

    return model, vectorizer, accuracy, report, data


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 🧠 Sentiment Analysis")
    st.markdown('<p class="sub">Powered by Support Vector Machine (SVC · RBF kernel) + TF-IDF</p>',
                unsafe_allow_html=True)

    model, vectorizer, accuracy, report, data = train_model()

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy:.2%}")
    c2.metric("Negative F1", f"{report['Negative']['f1-score']:.2f}")
    c3.metric("Neutral F1",  f"{report['Neutral']['f1-score']:.2f}")
    c4.metric("Positive F1", f"{report['Positive']['f1-score']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Prediction panel ─────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Predict Sentiment")

    phrase = st.text_area(
        "Enter a phrase below:",
        placeholder="e.g.  The flowers are very beautiful!",
        height=110,
        label_visibility="collapsed"
    )

    if st.button("Analyse Sentiment"):
        if not phrase.strip():
            st.warning("Please enter a phrase before analysing.")
        else:
            phrase_tfidf = vectorizer.transform([phrase])
            prediction   = model.predict(phrase_tfidf)[0]

            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            emoji_map = {0: "😞", 1: "😐", 2: "😊"}
            badge_map = {0: "badge-negative", 1: "badge-neutral", 2: "badge-positive"}

            label = label_map[prediction]
            emoji = emoji_map[prediction]
            badge = badge_map[prediction]

            st.markdown(
                f'<div class="{badge}">{emoji} {label}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Classification report table ──────────────────────────────────────────
    with st.expander("📋 Full Classification Report"):
        rows = []
        for cls in ['Negative', 'Neutral', 'Positive']:
            r = report[cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{r['precision']:.2f}",
                "Recall":    f"{r['recall']:.2f}",
                "F1-score":  f"{r['f1-score']:.2f}",
                "Support":   int(r['support'])
            })
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    # ── Dataset preview ──────────────────────────────────────────────────────
    with st.expander("🗂️ Dataset Preview (first 10 rows)"):
        display = data.copy()
        display['sentiment'] = display['sentiment'].map(
            {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        )
        st.dataframe(display.head(10), use_container_width=True)

    st.markdown(
        '<p style="text-align:center;color:#555;font-size:0.78rem;margin-top:2rem;">'
        "Model: SVC(kernel='rbf', random_state=1) · TF-IDF Vectorizer · 80/20 split</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
