import streamlit as st
import numpy as np
import re
import pickle
import os

# ── Page config ──
st.set_page_config(
    page_title="ADHD Detection from Social Media Text",
    page_icon="🧠",
    layout="centered"
)

# ── NLTK setup ──
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

# ── Text cleaning ──
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource
def load_models():
    from gensim.models import Word2Vec
    import keras
    svm_tfidf = pickle.load(open('svm_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    svm_w2v = pickle.load(open('svm_w2v_model.pkl', 'rb'))
    w2v_model = Word2Vec.load('w2v_model.bin')
    lstm_model = keras.models.load_model('lstm_model.h5')
    return svm_tfidf, vectorizer, svm_w2v, w2v_model, lstm_model

@st.cache_resource
def load_bert():
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    tokenizer = BertTokenizer.from_pretrained('svvayampattnaik/bert-adhd-detection')
    model = BertForSequenceClassification.from_pretrained('svvayampattnaik/bert-adhd-detection')
    model.eval()
    return tokenizer, model

def post_to_sequence(text, w2v_model, max_len=100, vector_size=100):
    words = text.split()[:max_len]
    sequence = []
    for word in words:
        if word in w2v_model.wv:
            sequence.append(w2v_model.wv[word])
        else:
            sequence.append(np.zeros(vector_size))
    while len(sequence) < max_len:
        sequence.append(np.zeros(vector_size))
    return np.array(sequence)

def get_lime_explanation(text, predict_fn, num_features=10, num_samples=300):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Non-ADHD', 'ADHD'])
    clean = clean_text(text)
    exp = explainer.explain_instance(clean, predict_fn,
                                     num_features=num_features,
                                     num_samples=num_samples)
    return exp.as_list()

st.title("🧠 ADHD Detection from Social Media Text")
st.markdown("**ITER CSE | Final Year Research Project | Group 09**")
st.markdown("Enter a social media post below. The system predicts ADHD indicators and explains which words influenced the prediction.")
st.divider()

model_choice = st.radio(
    "Select Model",
    ["SVM + TF-IDF", "SVM + Word2Vec", "LSTM + Word2Vec", "BERT"],
    horizontal=True
)

text_input = st.text_area("Enter text", placeholder="Paste a Reddit-style post here...", height=150)
show_lime = st.checkbox("Show LIME explanation", value=True)

if st.button("🔍 Predict", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Loading models and predicting..."):
            svm_tfidf, vectorizer, svm_w2v, w2v_model, lstm_model = load_models()
            clean = clean_text(text_input)

            if model_choice == "SVM + TF-IDF":
                tfidf = vectorizer.transform([clean])
                pred = svm_tfidf.predict(tfidf)[0]
                def predict_fn(texts):
                    cleaned = [clean_text(t) for t in texts]
                    tfidfs = vectorizer.transform(cleaned)
                    return svm_tfidf.decision_function(tfidfs).reshape(-1, 1) * [[-1, 1]]

            elif model_choice == "SVM + Word2Vec":
                ws = [w for w in clean.split() if w in w2v_model.wv]
                vec = np.mean([w2v_model.wv[w] for w in ws], axis=0) if ws else np.zeros(100)
                pred = svm_w2v.predict([vec])[0]
                def predict_fn(texts):
                    vecs = []
                    for t in texts:
                        c = clean_text(t)
                        words = [w for w in c.split() if w in w2v_model.wv]
                        v = np.mean([w2v_model.wv[w] for w in words], axis=0) if words else np.zeros(100)
                        vecs.append(v)
                    return svm_w2v.decision_function(vecs).reshape(-1, 1) * [[-1, 1]]

            elif model_choice == "LSTM + Word2Vec":
                seq = np.expand_dims(post_to_sequence(clean, w2v_model), axis=0)
                prob = lstm_model.predict(seq, verbose=0)[0][0]
                pred = int(prob > 0.5)
                def predict_fn(texts):
                    results = []
                    for t in texts:
                        c = clean_text(t)
                        s = np.expand_dims(post_to_sequence(c, w2v_model), axis=0)
                        p = lstm_model.predict(s, verbose=0)[0][0]
                        results.append([1 - float(p), float(p)])
                    return np.array(results)

            else:
                import torch
                bert_tokenizer, bert_model = load_bert()
                inputs = bert_tokenizer(text_input, return_tensors='pt',
                                        max_length=128, truncation=True,
                                        padding='max_length')
                with torch.no_grad():
                    logits = bert_model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
                def predict_fn(texts):
                    results = []
                    for t in texts:
                        inp = bert_tokenizer(t, return_tensors='pt',
                                             max_length=128, truncation=True,
                                             padding='max_length')
                        with torch.no_grad():
                            lg = bert_model(**inp).logits
                        probs = torch.softmax(lg, dim=1).numpy()[0]
                        results.append(probs.tolist())
                    return np.array(results)

        label = "🔴 ADHD" if pred == 1 else "🟢 Non-ADHD"
        color = "red" if pred == 1 else "green"

        st.divider()
        st.subheader("Prediction")
        st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
        st.caption(f"Model used: {model_choice}")

        if show_lime:
            with st.spinner("Running LIME explanation..."):
                explanation = get_lime_explanation(text_input, predict_fn)
            st.subheader("Top Influencing Words (LIME)")
            st.caption("🟠 = pushes toward ADHD | 🔵 = pushes toward Non-ADHD")
            for word, score in explanation:
                icon = "🟠" if score > 0 else "🔵"
                st.write(f"{icon} **{word}**: {score:.4f}")

st.divider()
st.markdown("""
**About**
- Dataset: 36,782 balanced Reddit posts
- SVM+TF-IDF: 96.56% | SVM+Word2Vec: 88.67% | LSTM: 96.94% | BERT: 98.11%
- Explainability: LIME
- ⚠️ Screening tool only — not a clinical diagnosis.
""")
