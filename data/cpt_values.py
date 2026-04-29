# ================================================================
# CPT VALUES — Multi-source Dataset (3 nguồn)
#
# Flu       → CDC FluView Clinical Surveillance Data
#             cdc.gov/flu/weekly/fluviewinteractive.htm
#
# Cold      → Kaggle 'Disease Symptom Prediction' (itachi9604, 2020)
#             kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
#             MLE + Laplace smoothing, n=120
#
# COVID     → Israeli Ministry of Health COVID-19 Dataset
#             Zoabi Y. et al., npj Digital Medicine 4, 21 (2021)
#             github.com/nshomron/covidpred
#             n=96,433 symptomatic COVID-positive patients
#             Fatigue, RunnyNose: WHO symptom prevalence data
#
# Pneumonia → Kaggle 'Disease Symptom Prediction' (itachi9604, 2020)
#             MLE + Laplace smoothing, n=120
# ================================================================

DISEASE_PRIOR = {
    'Flu': 0.35,
    'Cold': 0.3,
    'COVID': 0.2,
    'Pneumonia': 0.15,
}

SYMPTOM_CPT = {
    'Fever': {
        'Flu': 0.87,
        'Cold': 0.9426,
        'COVID': 0.5232,
        'Pneumonia': 0.9426,
    },
    'Cough': {
        'Flu': 0.8,
        'Cold': 0.9426,
        'COVID': 0.4408,
        'Pneumonia': 0.9426,
    },
    'Fatigue': {
        'Flu': 0.9,
        'Cold': 0.9918,
        'COVID': 0.7,
        'Pneumonia': 0.9918,
    },
    'SoreThroat': {
        'Flu': 0.5,
        'Cold': 0.9918,
        'COVID': 0.2124,
        'Pneumonia': 0.0082,
    },
    'Dyspnea': {
        'Flu': 0.2,
        'Cold': 0.0082,
        'COVID': 0.0773,
        'Pneumonia': 0.9426,
    },
    'RunnyNose': {
        'Flu': 0.6,
        'Cold': 0.9918,
        'COVID': 0.2,
        'Pneumonia': 0.0082,
    },
}

DISEASES = ['Flu', 'Cold', 'COVID', 'Pneumonia']
SYMPTOMS = ['Fever', 'Cough', 'Fatigue', 'SoreThroat', 'Dyspnea', 'RunnyNose']