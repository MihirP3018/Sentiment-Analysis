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

# File Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'sentiment_model.pkl'

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
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_space
    ]

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
    st.title("💬 Sentiment AI")
    st.caption("Customer Review Analyzer")
    st.divider()

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
    st.subheader("🤖 AI Powered")
    st.caption("Using NLP & Machine Learning to understand your feedback.")

# ABOUT PAGE
if st.session_state.page == "About":
    st.title("About Sentiment AI")
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
    st.title("💬 Sentiment AI")
    st.caption("Analyze customer reviews using NLP and Machine Learning.")
    st.divider()

    # MAIN ONE-PAGE LAYOUT
    input_col, result_col, probability_col = st.columns([1.15, 0.9, 0.95])

    # INPUT
    with input_col:
        st.subheader("💬 Customer Review")
        st.caption("Enter a review and click Analyze.")
        review = st.text_area("Customer Review", height = 120,
            placeholder = (
                "Example: The product quality is excellent "
                "and I am very happy with my purchase."
            ),
            max_chars = 500,
            key = "review_input",
            label_visibility = "collapsed"
        )

        st.caption(f"{len(review)}/500 characters")
        analyze_col, clear_col = st.columns(2)

        with analyze_col:
            analyze_button = st.button("✨ Analyze", type = "primary",
                use_container_width = True)

        with clear_col:
            st.button(
                "🗑️ Clear", use_container_width = True, on_click = clear_all
            )
            
    # ANALYZE
    if analyze_button:
            st.session_state.warning = None

            # Clear old results before processing a new review
            st.session_state.prediction = None
            st.session_state.probabilities = None
            st.session_state.cleaned_text = None

            if not review.strip():
                st.session_state.warning = (
                    "Please enter a customer review first."
                )

            else:
                with st.spinner("🤖 Analyzing..."):
                    try:
                        cleaned_review = clean_text(review)
                        if not cleaned_review.strip():
                            st.session_state.warning = (
                                "Please enter a meaningful English review. "
                                "Text containing only symbols or unsupported "
                                "content cannot be analyzed."
                            )

                        else:
                            probability_values = model.predict_proba(
                                [cleaned_review]
                            )[0]

                            classes = model.classes_
                            best_index = int(probability_values.argmax())
                            prediction = str(classes[best_index])

                            probabilities = {
                                str(label): float(value)
                                for label, value in zip(
                                    classes, probability_values
                                )
                            }

                            st.session_state.prediction = prediction
                            st.session_state.probabilities = probabilities
                            st.session_state.cleaned_text = cleaned_review

                    except Exception as error:
                        st.session_state.warning = (
                            "The review could not be analyzed. "
                            "Please check the model and dependencies. "
                            f"Details: {error}"
                        )

    if st.session_state.warning:
        st.warning(st.session_state.warning)

    # RESULT
    with result_col:
        st.subheader("📊 Result")
        if st.session_state.prediction is None:
            st.info("Enter a review and click Analyze.")

        else:
            prediction = st.session_state.prediction
            probabilities = st.session_state.probabilities

            if prediction == "Positive":
                emoji = "😊"
                message = "Positive sentiment"
            elif prediction == "Negative":
                emoji = "😞"
                message = "Negative sentiment"
            else:
                emoji = "😐"
                message = "Neutral sentiment"

            st.markdown(f"# {emoji}")
            st.subheader(prediction.upper())
            st.caption(message)

            predicted_probability = float(probabilities.get(prediction, 0.0))
            confidence_percent = predicted_probability * 100

            st.write(f"🎯 Model Probability: {confidence_percent:.1f}%")
            st.progress(min(max(predicted_probability, 0.0), 1.0))

            if predicted_probability < 0.50:
                st.warning(
                    "Low model probability: the review may be mixed "
                    "or difficult to classify. Treat this prediction "
                    "with caution."
                )

            st.write("📄 Processed Review")
            st.caption(st.session_state.cleaned_text)

    # PROBABILITIES
    with probability_col:
        st.subheader("📈 Probabilities")
        if st.session_state.prediction is None:
            st.info("Results will appear here.")

        else:
            probabilities = st.session_state.probabilities

            for label, emoji in [
                ("Negative", "🔴"),
                ("Neutral", "🟡"),
                ("Positive", "🟢")
            ]:
                value = float(probabilities.get(label, 0.0))
                st.write(f"{emoji} {label}: {value * 100:.1f}%")
                st.progress(min(max(value, 0.0), 1.0))

    st.divider()
    st.caption(
        "NLP Sentiment Analysis • Python • NLTK • spaCy • "
        "TF-IDF • Machine Learning • Streamlit"
    )