# ADHD Detection from Social Media Text Using NLP-Based ML and XAI

**Final Year Research Project 2026 | Group 09 | ITER CSE, SOA University**

## 🔴 Live Demo
👉 [https://huggingface.co/spaces/svvayampattnaik/adhd-detection-support](https://huggingface.co/spaces/svvayampattnaik/adhd-detection-support)

## 📌 About
An NLP-based system that detects ADHD indicators from social media text using machine learning and explainable AI. This is a **screening and awareness tool only — not a clinical diagnosis**.

## 🤖 Models & Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| SVM + TF-IDF | 96.56% | 0.965 |
| SVM + Word2Vec | 88.67% | 0.886 |
| LSTM + Word2Vec | 96.94% | 0.969 |
| **BERT (fine-tuned)** | **98.11%** | **0.981** |

## 📊 Dataset
- 36,782 balanced Reddit posts (r/ADHD + Normal class)
- 80/20 stratified train-test split

## 🔍 Explainability
LIME integrated with SVM+TF-IDF to highlight top 10 words influencing each prediction.

## 🧠 BERT Model on HuggingFace
👉 [https://huggingface.co/svvayampattnaik/bert-adhd-detection](https://huggingface.co/svvayampattnaik/bert-adhd-detection)

## 👥 Team
- Swayam Sibam Pattnaik (2241016362)
- Ravi Ranjan
- Aniket Kumar (2241011099)
- Aadarsh Kumar (2241011076)

**Supervisor:** Dr. Abhijit Pal
