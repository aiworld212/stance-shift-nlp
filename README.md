VIDEO LINK : https://drive.google.com/drive/folders/1uzDvQoSMy1HjK8MBZPUNE9Lg4V367w9t?usp=sharing

# 📊 Stance Shift — NLP Temporal Analysis

> Tracking how public opinion on **Climate, Vaccines, and AI** shifts over time using fine-tuned RoBERTa

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🔍 Overview

**Stance Shift** is an end-to-end NLP project that:

- Fine-tunes **RoBERTa-base** on multi-topic stance datasets
- Classifies social media posts as `favor`, `against`, or `neutral`
- Tracks stance distributions **over time** using temporal aggregation
- Detects **change points** where public opinion significantly shifted
- Measures **community polarization** using Jensen-Shannon divergence
- Serves everything through an interactive **Streamlit dashboard**

---


## 🗂️ Project Structure

```
stance-shift-nlp/
│
├── 📓 Notebooks
│   ├── 01_stance_detection_dataset.ipynb     # Data collection & merging
│   ├── 02_stance_roberta_training.ipynb      # RoBERTa fine-tuning
│   └── 03_stance_inference_temporal.ipynb   # Inference + temporal analysis
│
├── 🖥️ Dashboard
│   ├── app.py                                # Streamlit dashboard
│   └── requirements.txt                      # Dependencies
│
├── 📊 Data
│   ├── stance_dataset_merged.csv             # All 3 datasets combined
│   ├── stance_dataset_labeled.csv            # Labeled data for training
│   └── stance_predictions_full.csv           # Model predictions + timestamps
│
└── 🤖 Model
    └── roberta_stance_model/                 # Fine-tuned RoBERTa weights
```

---

## 📦 Datasets Used

| Dataset | Source | Topics | Labels |
|---|---|---|---|
| TweetEval Stance | HuggingFace | Climate, Feminist | Favor / Against / Neutral |
| SemEval 2016 Task 6 | HuggingFace | Climate, Atheism, Hillary, Abortion | Favor / Against / None |
| Climate FEVER | HuggingFace | Climate | Supports / Refutes / Not Enough Info |

---

## 🤖 Model

| Property | Value |
|---|---|
| Base model | `roberta-base` (125M parameters) |
| Fine-tuned on | TweetEval + SemEval 2016 + Climate FEVER |
| Classes | `favor` / `against` / `neutral` |
| Optimizer | AdamW with linear warmup |
| Epochs | 4 |
| Max sequence length | 128 tokens |

---

## 📈 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | Total posts, stance counts, distribution charts |
| **Temporal Trends** | Stacked area chart + net favor ratio over time |
| **Change Points** | Auto-detected stance shifts with adjustable sensitivity |
| **Live Predictor** | Type any text → instant stance prediction |

---

## ⚙️ Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/aiworld212/stance-shift-nlp.git
cd stance-shift-nlp
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard:**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📓 Run the Notebooks

Run in order:

```bash
# 1. Collect and merge datasets
jupyter notebook 01_stance_detection_dataset.ipynb

# 2. Fine-tune RoBERTa (requires GPU recommended)
jupyter notebook 02_stance_roberta_training.ipynb

# 3. Run inference and temporal analysis
jupyter notebook 03_stance_inference_temporal.ipynb
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Test Accuracy | ~78% |
| Macro F1 Score | ~0.75 |
| Training Time (GPU) | ~15 mins |
| Dataset Size | ~10,000 posts |
| Topics Covered | Climate, Vaccines, AI, Feminist |

---

## 🧠 Technical Highlights

- **Domain-adaptive fine-tuning** on social media text
- **Change point detection** using the PELT algorithm (`ruptures`)
- **Polarization measurement** via Jensen-Shannon divergence
- **Temporal aggregation** — monthly stance distributions with rolling trends
- **Confidence filtering** — high-confidence predictions for cleaner analysis

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Modeling | PyTorch, HuggingFace Transformers |
| Data | Pandas, HuggingFace Datasets |
| Analysis | SciPy, Ruptures, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🔮 Future Work

- Collect real Reddit data via PRAW for richer temporal signals
- Add SHAP explainability to show which words drive stance predictions
- Expand to more topics (immigration, gun control, crypto)
- Fine-tune on more recent data post-2023
- Add multilingual support (Hindi, Spanish)

---

## 👤 Author

**Sumit Yadav**
- GitHub: [@aiworld212](https://github.com/aiworld212)


