import streamlit as st
import joblib
import spacy
import nltk
import re
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# PAGE CONFIG
st.set_page_config(
    page_title = "Sentiment AI",
    page_icon = "💬",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# COMPACT UI STYLING
st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5f7ff,
            #eef2ff,
            #f8fafc
        );
        color: #1f2937;
    }

    /* Reduce top spacing */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0;
        max-width: 1400px;
    }

    /* Sidebar background and spacing */
    [data-testid="stSidebar"] {
        background: #eef2ff;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }

    /* Sidebar information card */
    .sidebar-info {
        margin-top: 5rem;
        padding: 1rem;
        background: #ffffff;
        border: 1px solid #dbe3f5;
        border-radius: 12px;
        text-align: center;
        color: #475569;
        font-size: 0.9rem;
    }

    .sidebar-info h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }

    .sidebar-info p {
        margin: 0.35rem 0;
    }

    /* Main heading */
    .hero-title {
        text-align: center;
        color: #1f2937;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0 0.2rem 0;
        padding: 0.3rem 0;
        line-height: 1.3;
    }

    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }

    /* Compact cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        padding: 0.6rem;
    }

    /* Text input */
    [data-testid="stTextArea"] textarea {
        border-radius: 10px;
        background: #ffffff;
        font-size: 0.95rem;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 2.4rem;
        padding: 0.3rem 0.6rem;
    }

    div.stButton > button[kind="primary"] {
        background: #4f46e5;
        color: #ffffff;
        border: none;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #4338ca;
        color: #ffffff;
    }

    /* Sentiment result cards */
    .sentiment-positive {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
    }

    .sentiment-negative {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
    }

    .sentiment-neutral {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
    }

    /* Reduce vertical gaps between elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 0.3rem;
        padding: 0.3rem 0;
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }

    /* Fix sentiment result spacing */
    .sentiment-positive,
    .sentiment-negative,
    .sentiment-neutral {
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }

    [data-testid="stCaptionContainer"] {
        line-height: 1.4;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html = True
)

# File Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

# MODEL
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "sentiment_model.pkl was not found. "
            "Run Sentiment_Analysis.ipynb to generate the model first."
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba(). "
            "Regenerate sentiment_model.pkl using the calibrated "
            "model-training steps in Sentiment_Analysis.ipynb."
        )

    return loaded_model

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

# NLTK RESOURCES
@st.cache_resource
def load_stop_words():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download("stopwords", quiet = True)
        return set(stopwords.words('english'))

def ensure_tokenizer_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet = True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet = True)

# Load dependencies with a helpful error if something is missing
try:
    model = load_model()
    nlp = load_spacy()
    stop_words = load_stop_words()
    ensure_tokenizer_resources()
except Exception as error:
    st.error(f'Unable to load the application: {error}')
    st.info(
        "Install the requirements, download the spaCy English model, "
        "and generate sentiment_model.pkl from the notebook."
    )
    st.stop()

# NEGATIONS: keep these words during stopword removal
negation_words = {
    "no", "not", "nor", "neither", "never", "none", "cannot"
}
stop_words = stop_words - negation_words

# CONTRACTIONS
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not"
}

# TEXT CLEANING
# Keep this preprocessing identical to the notebook.
def clean_text(text):
    text = str(text).lower()

    # Normalize curly apostrophes
    text = text.replace("\u2019", "'").replace("\u2018", "'")

    # Expand known contractions
    for key, value in contractions.items():
        text = text.replace(key, value)

    # Handle other n't contractions
    text = re.sub(r"n't\b", " not", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Keep letters, numbers, and spaces
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    tokens = word_tokenize(text)

    # Remove stopwords while preserving negations
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if not token.is_space]

    return " ".join(lemmas)

# SESSION STATE
default = {
    'page': 'Home',
    'prediction': None,
    'probabilities': None,
    'cleaned_text': None,
    'review_input': '',
    'warning': None
}

for key, value in default.items():
    if key not in st.session_state:
        st.session_state[key] = value

def clear_all():
    st.session_state.prediction = None
    st.session_state.probabilities = None
    st.session_state.cleaned_text = None
    st.session_state.review_input = ""
    st.session_state.warning = None

# SIDEBAR
with st.sidebar:
    st.markdown("## 💬 Sentiment AI")
    st.caption("Customer Review Analyzer")
    if st.button("🏠 Home", use_container_width = True):
        st.session_state.page = "Home"
        st.rerun()
    if st.button("🔍 Analyze", use_container_width = True):
        st.session_state.page = "Analyze"
        st.rerun()
    if st.button("ⓘ About", use_container_width = True):
        st.session_state.page = "About"
        st.rerun()
    st.divider()
    st.caption("🤖 AI-powered NLP and Machine Learning")
    st.markdown("😊 Positive  ·  😐 Neutral  ·  😞 Negative")
    st.markdown(
    """
    <div class="sidebar-info">
        <h4>💡 Quick Guide</h4>
        <p>1. Enter your customer review.</p>
        <p>2. Click Analyze Sentiment.</p>
        <p>3. View your prediction and probabilities.</p>
        <hr>
        <p>Built with Python, NLP and Machine Learning.</p>
    </div>
    """,
    unsafe_allow_html = True
)

# ABOUT PAGE
if st.session_state.page == "About":
    st.markdown('<div class="hero-title">💬 About Sentiment AI</div>', unsafe_allow_html = True)
    st.markdown(
        '<div class="hero-subtitle">How the customer review analyzer works</div>',
        unsafe_allow_html = True
    )

    st.write(
        "Sentiment AI analyzes English mobile-phone customer reviews "
        "and predicts whether the review expresses Positive, Neutral, "
        "or Negative sentiment."
    )

    about1, about2 = st.columns(2)
    with about1:
        with st.container(border = True):
            st.subheader("🤖 Machine Learning")
            st.write(
                "The application uses a calibrated machine-learning "
                "classifier with TF-IDF text features."
            )
    with about2:
        with st.container(border = True):
            st.subheader("🧹 NLP Processing")
            st.write(
                "Reviews are cleaned, tokenized, filtered while "
                "preserving negations, and lemmatized before prediction."
            )

    st.info("Enter a customer review in the Analyze section to get a sentiment prediction.")
    st.divider()
    st.caption(
        "NLP Sentiment Analysis • Python • NLTK • spaCy • "
        "TF-IDF • Machine Learning • Streamlit"
    )

# ANALYZE / HOME PAGE
elif st.session_state.page in ["Home", "Analyze"]:
    st.markdown('<div class="hero-title">💬 Sentiment AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Analyze a customer review in seconds</div>', unsafe_allow_html=True)

    input_col, output_col = st.columns([1, 1], gap = "small")

    with input_col:
        with st.container(border = True):
            st.subheader("✍️ Enter Review")
            review = st.text_area(
                "Customer Review",
                height = 105,
                placeholder = "Type your customer review here...",
                max_chars = 500,
                key = "review_input",
                label_visibility = "collapsed"
            )
            st.caption(f"{len(review)}/500 characters")
            analyze_button = st.button(
                "✨ Analyze Sentiment", type = "primary", use_container_width = True
            )
            st.button("🗑️ Clear", use_container_width=True, on_click = clear_all)

    # ANALYZE
    if analyze_button:
        st.session_state.warning = None
        st.session_state.prediction = None
        st.session_state.probabilities = None
        st.session_state.cleaned_text = None

        if not review.strip():
            st.session_state.warning = "Please enter a customer review first."
        else:
            with st.spinner("Analyzing review..."):
                try:
                    cleaned_review = clean_text(review)
                    if not cleaned_review.strip():
                        st.session_state.warning = (
                            "Please enter a meaningful English review."
                        )
                    else:
                        probability_values = model.predict_proba([cleaned_review])[0]
                        classes = model.classes_
                        best_index = int(probability_values.argmax())
                        st.session_state.prediction = str(classes[best_index])
                        st.session_state.probabilities = {
                            str(label): float(value)
                            for label, value in zip(classes, probability_values)
                        }
                        st.session_state.cleaned_text = cleaned_review
                except Exception as error:
                    st.session_state.warning = f"Could not analyze review: {error}"

    with output_col:
        with st.container(border=True):
            st.subheader("📊 Sentiment Result")
            if st.session_state.warning:
                st.warning(st.session_state.warning)
            elif st.session_state.prediction is None:
                st.info("Your prediction and probabilities will appear here.")
            else:
                prediction = st.session_state.prediction
                probabilities = st.session_state.probabilities
                if prediction == "Positive":
                    emoji, message, sentiment_class = "😊", "Positive sentiment detected.", "sentiment-positive"
                elif prediction == "Negative":
                    emoji, message, sentiment_class = "😞", "Negative sentiment detected.", "sentiment-negative"
                else:
                    emoji, message, sentiment_class = "😐", "Neutral sentiment detected.", "sentiment-neutral"

                st.markdown(
                    f'<div class = "{sentiment_class}">{emoji} {prediction.upper()}</div>',
                    unsafe_allow_html = True
                )
                st.caption(message)
                st.markdown("**Class probabilities**")
                for label, emoji in [("Negative", "🔴"), ("Neutral", "🟡"), ("Positive", "🟢")]:
                    value = float(probabilities.get(label, 0.0))
                    st.write(f"{emoji} {label}: {value * 100:.1f}%")
                    st.progress(min(max(value, 0.0), 1.0))
                st.caption(f"Model probability for {prediction}: {probabilities.get(prediction, 0.0) * 100:.1f}%")
                with st.expander("View processed review"):
                    st.write(st.session_state.cleaned_text)

    st.markdown('<div class="app-footer">Python · NLTK · spaCy · TF-IDF · Machine Learning · Streamlit</div>', unsafe_allow_html=True)
