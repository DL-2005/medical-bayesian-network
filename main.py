import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model.bn_model import build_model
from inference.engine import (
    run_inference, find_mpe,
    explain_disease, compare_diseases, full_report
)
from inference.evaluation import evaluate, what_if_analysis, sensitivity_analysis

def main():
    # Khởi tạo model (chỉ cần 1 lần)
    build_model()

    # ── TEST 1: Inference cơ bản ──
    print("\n\n" + "="*60)
    print("DEMO 1 — Inference: Sốt + Ho + Khó thở")
    run_inference({
        'Fever':   True,
        'Cough':   True,
        'Dyspnea': True
    })

    # ── TEST 2: Explain chiều ngược ──
    print("\n\n" + "="*60)
    print("DEMO 2 — Explain: Nếu bệnh nhân mắc COVID thì sao?")
    explain_disease('COVID')

    print("\n\n" + "="*60)
    print("DEMO 3 — Explain: Nếu bệnh nhân mắc Pneumonia thì sao?")
    explain_disease('Pneumonia')

    # ── TEST 3: Compare symptom ──
    print("\n\n" + "="*60)
    print("DEMO 4 — Compare: Khó thở đặc trưng cho bệnh nào nhất?")
    compare_diseases('Dyspnea')

    print("\n\n" + "="*60)
    print("DEMO 5 — Compare: Đau họng đặc trưng cho bệnh nào?")
    compare_diseases('SoreThroat')

    # ── TEST 4: Full report ──
    print("\n\n")
    full_report({
        'Fever':   True,
        'Cough':   True,
        'Dyspnea': True
    })


if __name__ == '__main__':
    main()
    print("\n\n>>> EVALUATION & ANALYSIS:")
    evaluate()
    what_if_analysis({'Fever': True, 'Cough': True})
    sensitivity_analysis()

