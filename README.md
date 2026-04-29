# Medical Bayesian Network — Respiratory Disease Diagnosis

Hệ thống chẩn đoán bệnh hô hấp dựa trên Bayesian Network và Variable Elimination.

## Bệnh được hỗ trợ
- Influenza (Flu)
- Common Cold
- COVID-19
- Pneumonia

## Triệu chứng
Fever, Cough, Fatigue, Sore Throat, Dyspnea, Runny Nose

## Cài đặt

```bash
pip install pgmpy networkx pandas streamlit plotly matplotlib
```

## Chạy app

```bash
streamlit run app.py
```

## Nguồn dữ liệu CPT
- **Flu**: CDC FluView Clinical Surveillance
- **Cold, Pneumonia**: Kaggle Disease Symptom Prediction (itachi9604, 2020)
- **COVID**: Israeli Ministry of Health — Zoabi et al., npj Digital Medicine (2021)

## Cấu trúc project

```
medical_bn/
├── data/
│   └── cpt_values.py       # CPT từ 3 nguồn dataset
├── model/
│   ├── bn_model.py         # BN structure với pgmpy
│   └── visualize.py        # DAG visualization
├── inference/
│   ├── engine.py           # VE engine tự viết
│   └── evaluation.py       # Evaluation & what-if
├── app.py                  # Streamlit UI
└── main.py                 # CLI testing
```