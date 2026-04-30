# ================================================================
# CPT VALUES — Multi-source Dataset (3 nguồn) — 8 bệnh, 15 triệu chứng
#
# Flu         → CDC FluView Clinical Surveillance
# Cold        → Kaggle itachi9604 (MLE + Laplace, n=120)
# COVID       → Israeli MoH, Zoabi et al. npj Digital Medicine (2021)
#               n=96,433 symptomatic; Fatigue/RunnyNose: WHO
# Pneumonia   → Kaggle itachi9604 (MLE + Laplace, n=120)
# Asthma      → Kaggle itachi9604 (MLE + Laplace, n=120)
# Tuberculosis→ Kaggle itachi9604 (MLE + Laplace, n=120)
# Allergy     → Kaggle itachi9604 (MLE + Laplace, n=120)
# Bronchitis  → WHO/CDC symptom guidelines (expert knowledge)
# ================================================================

DISEASE_PRIOR = {
    'Flu': 0.25,
    'Cold': 0.22,
    'COVID': 0.15,
    'Pneumonia': 0.12,
    'Asthma': 0.1,
    'Tuberculosis': 0.06,
    'Allergy': 0.07,
    'Bronchitis': 0.03,
}

SYMPTOM_CPT = {
    'Fever': {
        'Flu': 0.87,
        'Cold': 0.9426,
        'COVID': 0.5232,
        'Pneumonia': 0.9426,
        'Asthma': 0.9426,
        'Tuberculosis': 0.9918,
        'Allergy': 0.0082,
        'Bronchitis': 0.4,
    },
    'Cough': {
        'Flu': 0.8,
        'Cold': 0.9426,
        'COVID': 0.4408,
        'Pneumonia': 0.9426,
        'Asthma': 0.8934,
        'Tuberculosis': 0.9426,
        'Allergy': 0.0082,
        'Bronchitis': 0.95,
    },
    'Fatigue': {
        'Flu': 0.9,
        'Cold': 0.9918,
        'COVID': 0.7,
        'Pneumonia': 0.9918,
        'Asthma': 0.8934,
        'Tuberculosis': 0.9918,
        'Allergy': 0.0082,
        'Bronchitis': 0.6,
    },
    'SoreThroat': {
        'Flu': 0.5,
        'Cold': 0.9918,
        'COVID': 0.2124,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.0082,
        'Allergy': 0.0082,
        'Bronchitis': 0.35,
    },
    'Dyspnea': {
        'Flu': 0.2,
        'Cold': 0.0082,
        'COVID': 0.0773,
        'Pneumonia': 0.9426,
        'Asthma': 0.9426,
        'Tuberculosis': 0.9426,
        'Allergy': 0.0082,
        'Bronchitis': 0.7,
    },
    'RunnyNose': {
        'Flu': 0.6,
        'Cold': 0.9918,
        'COVID': 0.2,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.0082,
        'Allergy': 0.8934,
        'Bronchitis': 0.1,
    },
    'ChestPain': {
        'Flu': 0.1,
        'Cold': 0.9918,
        'COVID': 0.15,
        'Pneumonia': 0.9918,
        'Asthma': 0.0082,
        'Tuberculosis': 0.9918,
        'Allergy': 0.0082,
        'Bronchitis': 0.5,
    },
    'Chills': {
        'Flu': 0.75,
        'Cold': 0.9426,
        'COVID': 0.4,
        'Pneumonia': 0.9426,
        'Asthma': 0.0082,
        'Tuberculosis': 0.9426,
        'Allergy': 0.9918,
        'Bronchitis': 0.3,
    },
    'MuscleAches': {
        'Flu': 0.8,
        'Cold': 0.9918,
        'COVID': 0.35,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.0082,
        'Allergy': 0.0082,
        'Bronchitis': 0.2,
    },
    'Headache': {
        'Flu': 0.7,
        'Cold': 0.9426,
        'COVID': 0.4313,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.0082,
        'Allergy': 0.0082,
        'Bronchitis': 0.15,
    },
    'Phlegm': {
        'Flu': 0.3,
        'Cold': 0.9918,
        'COVID': 0.2,
        'Pneumonia': 0.9426,
        'Asthma': 0.9426,
        'Tuberculosis': 0.9918,
        'Allergy': 0.0082,
        'Bronchitis': 0.9,
    },
    'WeightLoss': {
        'Flu': 0.05,
        'Cold': 0.0082,
        'COVID': 0.1,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.9426,
        'Allergy': 0.0082,
        'Bronchitis': 0.1,
    },
    'LossOfAppetite': {
        'Flu': 0.5,
        'Cold': 0.0082,
        'COVID': 0.3,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.9918,
        'Allergy': 0.0082,
        'Bronchitis': 0.2,
    },
    'NightSweats': {
        'Flu': 0.1,
        'Cold': 0.0082,
        'COVID': 0.1,
        'Pneumonia': 0.9426,
        'Asthma': 0.0082,
        'Tuberculosis': 0.9426,
        'Allergy': 0.0082,
        'Bronchitis': 0.05,
    },
    'WateringEyes': {
        'Flu': 0.1,
        'Cold': 0.0082,
        'COVID': 0.05,
        'Pneumonia': 0.0082,
        'Asthma': 0.0082,
        'Tuberculosis': 0.0082,
        'Allergy': 0.8934,
        'Bronchitis': 0.05,
    },
}

DISEASES = ['Flu', 'Cold', 'COVID', 'Pneumonia', 'Asthma', 'Tuberculosis', 'Allergy', 'Bronchitis']
SYMPTOMS = ['Fever', 'Cough', 'Fatigue', 'SoreThroat', 'Dyspnea', 'RunnyNose', 'ChestPain', 'Chills', 'MuscleAches', 'Headache', 'Phlegm', 'WeightLoss', 'LossOfAppetite', 'NightSweats', 'WateringEyes']
