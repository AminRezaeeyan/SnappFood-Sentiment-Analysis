import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go

# ------------------ Streamlit Page Config ------------------
st.set_page_config(
    page_title="SnappFood Sentiment Analysis",
    page_icon="🍽️",
    layout="wide"
)

# ------------------ Custom CSS for RTL and Dark Theme ------------------
st.markdown("""
<style>
:root {
    --snappfood-pink: #FF0057;
    --snappfood-light-pink: #FFE5EE;
    --snappfood-dark-pink: #CC0046;
    --dark-bg: #1E1E1E;
    --dark-secondary: #2D2D2D;
    --dark-text: #FFFFFF;
    --dark-text-secondary: #B3B3B3;
}

.stApp {
    background-color: var(--dark-bg);
    color: var(--dark-text);
}

section textarea {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
    background-color: var(--dark-secondary) !important;
    color: var(--dark-text) !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    border: 2px solid var(--dark-secondary) !important;
}

textarea::placeholder {
    direction: rtl !important;
    text-align: right !important;
    color: var(--dark-text-secondary) !important;
}

.stButton>button {
    width: 100%;
    height: 3rem;
    font-size: 1.1rem;
    background-color: var(--snappfood-pink);
    color: white;
    border: none;
    border-radius: 10px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: var(--snappfood-dark-pink);
    box-shadow: 0 4px 8px rgba(255, 0, 87, 0.2);
}

h1, h2, h3 {
    color: var(--snappfood-pink) !important;
    font-weight: bold !important;
}

.sentiment-text {
    font-size: 1.2rem;
    font-weight: 500;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model and Tokenizer ------------------
@st.cache_resource
def load_model():
    model_path = Path("models")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model, st.session_state.tokenizer = load_model()

# ------------------ Helper Functions ------------------
def get_sentiment_description(score):
    if score <= 0.2:
        return "بسیار مثبت", "😊"
    elif score <= 0.4:
        return "مثبت", "🙂"
    elif score <= 0.6:
        return "خنثی", "😐"
    elif score <= 0.8:
        return "منفی", "🙁"
    else:
        return "بسیار منفی", "😢"

def create_sentiment_gauge(score):
    display_score = (1 - score) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF0057"},
            'steps': [
                {'range': [0, 20], 'color': "#FFCCCC"},
                {'range': [20, 40], 'color': "#FFAAAA"},
                {'range': [40, 60], 'color': "#FF8888"},
                {'range': [60, 80], 'color': "#FF4444"},
                {'range': [80, 100], 'color': "#FF0000"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': display_score
            }
        },
        title={'text': "Sentiment Score", 'font': {'size': 20}}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=300
    )
    return fig

# ------------------ Page Content ------------------
st.title("🍽️ SnappFood Sentiment Analysis")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Your Review")
    review_text = st.text_area(
        " ",
        height=150,
        placeholder="تجربه خود را با اسنپ فود به اشتراک بگذارید...",
        key="review_input"
    )
    analyze_button = st.button("Analyze Sentiment", type="primary")

with col2:
    st.subheader("Analysis Results")
    if analyze_button and review_text.strip():
        with st.spinner("Analyzing..."):
            inputs = st.session_state.tokenizer(review_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = st.session_state.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiment_score = probs[0][1].item()

            sentiment_text, emoji = get_sentiment_description(sentiment_score)

            st.markdown(f"""
                <div class="sentiment-text">
                    {emoji} تحلیل احساسات: {sentiment_text}
                </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True)

            st.markdown("""
                <div style='direction: rtl; text-align: right;'>
                    <h4>راهنمای امتیاز:</h4>
                    <ul>
                        <li>نمره 80 تا 100: بسیار مثبت</li>
                        <li>نمره 60 تا 80: مثبت</li>
                        <li>نمره 40 تا 60: خنثی</li>
                        <li>نمره 20 تا 40: منفی</li>
                        <li>نمره 0 تا 20: بسیار منفی</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    elif analyze_button:
        st.warning("Please enter a review.")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app analyzes customer reviews for SnappFood using a fine-tuned sentiment analysis model.

    - Type your review in Persian  
    - Click the analyze button  
    - View the sentiment score and category  
    """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center' class='footer'>
    <p>© 2025 SnappFood Sentiment Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)
