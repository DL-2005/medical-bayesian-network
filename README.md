# 🏥 Medical Bayesian Network — Respiratory Disease Diagnosis

> **Advanced Artificial Intelligence — Final Project**  
> Bayesian Networks + Variable Elimination

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![pgmpy](https://img.shields.io/badge/pgmpy-verified-green)](https://pgmpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Overview

A probabilistic medical diagnostic system that computes the posterior probability distribution **P(Disease | Symptoms)** using a hand-crafted Bayesian Network and Variable Elimination implemented from scratch in Python.

Given a partial set of observed symptoms as evidence, the system returns a ranked differential diagnosis with full step-by-step reasoning trace — making the AI decision process fully transparent and auditable.

---

## 🦠 Supported Diseases (8)

| Disease | Prior P(D) | Data Source |
|---|---|---|
| Influenza (Flu) | 0.25 | CDC FluView Clinical Surveillance |
| Common Cold | 0.22 | Kaggle itachi9604 (MLE, n=120) |
| COVID-19 | 0.15 | Israeli MoH — Zoabi et al., npj Digital Medicine (2021) |
| Pneumonia | 0.12 | Kaggle itachi9604 (MLE, n=120) |
| Asthma | 0.10 | Kaggle itachi9604 (MLE, n=120) |
| Tuberculosis | 0.06 | Kaggle itachi9604 (MLE, n=120) |
| Allergy | 0.07 | Kaggle itachi9604 (MLE, n=120) |
| Bronchitis | 0.03 | WHO/CDC Expert Knowledge |

## 🩺 Symptoms (15)

Fever · Cough · Fatigue · Sore Throat · Dyspnea · Runny Nose · Chest Pain · Chills · Muscle Aches · Headache · Phlegm · Weight Loss · Loss of Appetite · Night Sweats · Watering Eyes

---

## ✨ Features

- **Variable Elimination from scratch** — not using pgmpy's inference engine; fully custom implementation with step-by-step reasoning trace
- **MPE (Most Probable Explanation)** — Max-Product VE to find the single most likely diagnosis
- **What-if Analysis** — add one symptom at a time and observe how the posterior shifts
- **Sensitivity Analysis** — ranks symptoms by informativeness (how much they shift the posterior)
- **Interactive BN Graph** — Plotly-based DAG with hover-to-see CPT values, color-coded evidence nodes
- **CPT Heatmap** — visual overview of all 15×8 conditional probability values
- **pgmpy Verification** — VE output cross-validated against pgmpy's VariableElimination (deviation = 0.0000)

---

## 📊 Evaluation Results

| Metric | Value |
|---|---|
| Top-1 Accuracy | 50.0% (10/20) |
| Top-2 Accuracy | 85.0% (17/20) |
| Top-3 Accuracy | 85.0% (17/20) |
| Brier Score | 0.4564 |
| Pneumonia Top-1 | 100% (5/5) |
| Cold Top-1 | 100% (5/5) |
| VE vs pgmpy deviation | 0.0000 |

> Top-3 = Top-2 indicates that misclassified cases fall below rank 3 entirely — a characteristic of Naive Bayes inference with CPT imbalance, not marginal rank-swaps.

---

## 🗂️ Project Structure

```
medical-bayesian-network/
├── data/
│   └── cpt_values.py          # CPT parameters — 3 data sources, fully cited
├── model/
│   ├── bn_model.py            # BN structure via pgmpy (DiscreteBayesianNetwork)
│   └── visualize.py           # Static DAG visualization (networkx + matplotlib)
├── inference/
│   ├── engine.py              # VE engine from scratch + MPE + explain + compare
│   └── evaluation.py          # Top-1/2/3 accuracy, Brier Score, what-if, sensitivity
├── app.py                     # Streamlit web UI (interactive)
├── main.py                    # CLI test runner
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone repo

```bash
git clone https://github.com/DL-2005/medical-bayesian-network.git
cd medical-bayesian-network
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run web app

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run CLI tests

```bash
python main.py
```

---

## 📦 Requirements

```
pgmpy
networkx
pandas
streamlit
plotly
matplotlib
```

---

## 🗃️ Data Sources & CPT Estimation

CPT parameters P(Symptom | Disease) were estimated using **Maximum Likelihood Estimation (MLE) with Laplace smoothing (α=1)**:

```
P(S=True | D) = (count(S=True, D) + 1) / (count(D) + 2)
```

| Disease | Source | n |
|---|---|---|
| Flu | [CDC FluView](https://www.cdc.gov/flu/weekly/fluviewinteractive.htm) | Clinical surveillance |
| Cold, Pneumonia, Asthma, Tuberculosis, Allergy | [Kaggle itachi9604](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | 120/disease |
| COVID-19 | [Israeli MoH via Zoabi et al. (2021)](https://github.com/nshomron/covidpred) | 96,433 symptomatic |
| Bronchitis | WHO/CDC symptom guidelines | Expert knowledge |

---

## 🧠 How VE Works (Quick Summary)

For each disease **d**, compute an unnormalized score:

```
score(d) = P(d) × ∏ P(symptom_i = v_i | d)   [only observed symptoms]
```

Unobserved symptoms are dropped because `Σ P(S|d) = 1` → they marginalize to 1 and contribute no information.

Then normalize:

```
P(d | evidence) = score(d) / Σ score(d')
```

The Reasoning Trace in the UI shows every multiplication step explicitly.

---

## ⚠️ Limitations

- **Top-1 accuracy 50%** — Flu/COVID confusion due to CPT quality of the itachi9604 dataset (Cold has anomalously high symptom prevalence). Pneumonia and Cold achieve 100% Top-1 accuracy.
- **Naive Bayes assumption** — symptoms are assumed conditionally independent given disease, which is a simplification. Correlated symptoms (e.g., Fever + Chills) are treated independently.
- **Test set coverage** — evaluation covers 4 of 8 diseases. Asthma, Tuberculosis, Allergy, Bronchitis test cases are left for future work.
- **No FastAPI** — Streamlit calls the inference engine directly as a library. A REST API wrapper can be added without modifying the core logic.

---

## 📚 References

1. Zoabi Y., Deri-Rozov S., Shomron N. (2021). Machine learning-based prediction of COVID-19 diagnosis based on symptoms. *npj Digital Medicine*, 4, 21.
2. itachi9604. (2020). Disease Symptom Prediction [Dataset]. Kaggle.
3. Israeli Ministry of Health. (2020). COVID-19 Tested Individuals Dataset. data.gov.il.
4. CDC. (2024). FluView Interactive — Clinical Data. cdc.gov/flu/weekly/fluviewinteractive.htm
5. Koller D., Friedman N. (2009). *Probabilistic Graphical Models*. MIT Press.
6. Ankan A., Panda A. (2015). pgmpy: Probabilistic Graphical Models using Python. SciPy 2015.

---



*Advanced Artificial Intelligence — May 2026*
