import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.engine import run_inference
from data.cpt_values import DISEASES, SYMPTOMS, SYMPTOM_CPT, DISEASE_PRIOR

# ================================================================
# TEST SET — ground truth cases để evaluate
# Mỗi case: (evidence, true_disease)
# ================================================================
TEST_CASES = [
    # --- Pneumonia cases ---
    ({'Fever': True,  'Cough': True,  'Dyspnea': True,  'Fatigue': True,  'SoreThroat': False, 'RunnyNose': False}, 'Pneumonia'),
    ({'Fever': True,  'Cough': True,  'Dyspnea': True,  'Fatigue': True }, 'Pneumonia'),
    ({'Fever': True,  'Dyspnea': True, 'Cough': True  }, 'Pneumonia'),
    ({'Fever': True,  'Cough': True,  'Dyspnea': True,  'RunnyNose': False}, 'Pneumonia'),
    ({'Fever': True,  'Fatigue': True, 'Dyspnea': True  }, 'Pneumonia'),

    # --- Flu cases ---
    ({'Fever': True,  'Cough': True,  'Fatigue': True,  'SoreThroat': True,  'RunnyNose': True }, 'Flu'),
    ({'Fever': True,  'Fatigue': True, 'Cough': True,   'Dyspnea': False }, 'Flu'),
    ({'Fever': True,  'Cough': True,  'Fatigue': True,  'RunnyNose': True  }, 'Flu'),
    ({'Fever': True,  'Fatigue': True, 'RunnyNose': True }, 'Flu'),
    ({'Fever': True,  'Cough': True,  'Fatigue': True  }, 'Flu'),

    # --- Cold cases ---
    ({'SoreThroat': True,  'RunnyNose': True,  'Fever': False, 'Dyspnea': False}, 'Cold'),
    ({'RunnyNose': True,   'SoreThroat': True, 'Cough': True,  'Fever': False  }, 'Cold'),
    ({'SoreThroat': True,  'RunnyNose': True,  'Fatigue': True, 'Fever': False  }, 'Cold'),
    ({'RunnyNose': True,   'SoreThroat': True }, 'Cold'),
    ({'SoreThroat': True,  'RunnyNose': True,  'Cough': True  }, 'Cold'),

    # --- COVID cases ---
    ({'Fever': True,  'Fatigue': True, 'SoreThroat': True, 'RunnyNose': False, 'Dyspnea': False}, 'COVID'),
    ({'Fever': True,  'Fatigue': True, 'Cough': True,  'SoreThroat': False }, 'COVID'),
    ({'Fatigue': True, 'Fever': True,  'Cough': True   }, 'COVID'),
    ({'Fever': True,  'Fatigue': True  }, 'COVID'),
    ({'Fatigue': True, 'Cough': True,  'Fever': True,  'Dyspnea': False, 'RunnyNose': False}, 'COVID'),

    # --- Tuberculosis ---
    ({'Fever': True, 'Cough': True, 'WeightLoss': True, 'NightSweats': True, 'LossOfAppetite': True}, 'Tuberculosis'),
    ({'Cough': True, 'WeightLoss': True, 'Fatigue': True, 'NightSweats': True}, 'Tuberculosis'),

    # --- Asthma ---
    ({'Dyspnea': True, 'Cough': True, 'Fever': False, 'WeightLoss': False}, 'Asthma'),
    ({'Dyspnea': True, 'Phlegm': True, 'Cough': True}, 'Asthma'),

    # --- Allergy ---
    ({'WateringEyes': True, 'RunnyNose': True, 'Fever': False}, 'Allergy'),
    ({'WateringEyes': True, 'Chills': True, 'RunnyNose': True}, 'Allergy'),

    # --- Bronchitis ---
    ({'Cough': True, 'Phlegm': True, 'Dyspnea': True, 'Fever': False}, 'Bronchitis'),
    ({'Cough': True, 'Phlegm': True, 'Fatigue': True}, 'Bronchitis'),
]


def evaluate(verbose=True):
    """Đánh giá model trên test set."""

    results = []
    for evidence, true_disease in TEST_CASES:
        posterior = run_inference(evidence, verbose=False)
        ranked    = sorted(posterior.items(), key=lambda x: -x[1])
        pred_top1 = ranked[0][0]
        pred_top2 = ranked[1][0] if len(ranked) > 1 else None
        correct_p = posterior[true_disease]

        results.append({
            'evidence':     evidence,
            'true':         true_disease,
            'pred_top1':    pred_top1,
            'pred_top2':    pred_top2,
            'correct_p':    correct_p,
            'top1_correct': pred_top1 == true_disease,
            'top2_correct': pred_top1 == true_disease or pred_top2 == true_disease,
            'ranked':       ranked,
        })

    # ── Metrics ──────────────────────────────────────────
    n          = len(results)
    top1_acc   = sum(r['top1_correct'] for r in results) / n
    top2_acc   = sum(r['top2_correct'] for r in results) / n
    brier      = sum((1 - r['correct_p'])**2 for r in results) / n

    # Per-disease accuracy
    per_disease = {}
    for d in DISEASES:
        subset = [r for r in results if r['true'] == d]
        if subset:
            acc = sum(r['top1_correct'] for r in subset) / len(subset)
            per_disease[d] = (acc, len(subset))

    if verbose:
        print("\n" + "="*60)
        print("   MODEL EVALUATION — Test Set Results")
        print("="*60)

        # Chi tiết từng case
        print(f"\n{'#':>3}  {'True':12s} {'Pred':12s} {'P(true)':>8}  Result")
        print("-"*55)
        for i, r in enumerate(results):
            icon = "✅" if r['top1_correct'] else "❌"
            ev_short = ', '.join([f"{k[:3]}={'T' if v else 'F'}"
                                  for k,v in r['evidence'].items()])
            print(f"{i+1:>3}  {r['true']:12s} {r['pred_top1']:12s} "
                  f"{r['correct_p']:>8.4f}  {icon}")
            print(f"     Evidence: {ev_short}")

        # Confusion matrix
        print(f"\n{'':15s}", end='')
        for d in DISEASES:
            print(f"{d[:6]:>8s}", end='')
        print("  (predicted →)")
        print("-"*50)
        for true_d in DISEASES:
            print(f"{true_d:12s}  ", end='')
            for pred_d in DISEASES:
                count = sum(1 for r in results
                            if r['true'] == true_d and r['pred_top1'] == pred_d)
                marker = "██" if true_d == pred_d else "  "
                print(f"{count:>6}{marker}", end='')
            print()
        print("(true ↓)")

        # Summary metrics
        print(f"\n{'='*60}")
        print(f"  Top-1 Accuracy : {top1_acc:.1%}  ({sum(r['top1_correct'] for r in results)}/{n})")
        print(f"  Top-2 Accuracy : {top2_acc:.1%}  ({sum(r['top2_correct'] for r in results)}/{n})")
        print(f"  Brier Score    : {brier:.4f}  (thấp hơn = tốt hơn, best=0)")
        print(f"\n  Per-disease Top-1 Accuracy:")
        for d, (acc, cnt) in per_disease.items():
            bar = '█' * int(acc * 20)
            print(f"    {d:12s}: {acc:.1%}  {bar}  ({cnt} cases)")

    return {
        'top1_acc': top1_acc,
        'top2_acc': top2_acc,
        'brier':    brier,
        'per_disease': per_disease,
        'results':  results,
    }


def what_if_analysis(base_evidence: dict, verbose=True):
    """
    Phân tích what-if: thêm từng triệu chứng vào evidence,
    xem xác suất thay đổi thế nào.
    """
    if verbose:
        print("\n" + "="*60)
        print("   WHAT-IF ANALYSIS")
        print("="*60)
        print(f"   Base evidence: {base_evidence}")

    base_posterior = run_inference(base_evidence, verbose=False)
    base_winner    = max(base_posterior, key=base_posterior.get)

    if verbose:
        print(f"\n   Base result:")
        for d, p in sorted(base_posterior.items(), key=lambda x: -x[1]):
            bar = '█' * int(p * 25)
            print(f"     {d:12s}: {p:.4f}  {bar}")

    # Thêm từng symptom chưa có trong evidence
    remaining = [s for s in SYMPTOMS if s not in base_evidence]

    if verbose:
        print(f"\n   Nếu thêm từng triệu chứng (= True):")
        print(f"   {'Symptom':14s}", end='')
        for d in DISEASES:
            print(f"  {d:10s}", end='')
        print("  → Winner")
        print("   " + "-"*65)

    what_if_results = {}
    for sym in remaining:
        new_evidence = {**base_evidence, sym: True}
        new_posterior = run_inference(new_evidence, verbose=False)
        new_winner    = max(new_posterior, key=new_posterior.get)
        delta         = {d: new_posterior[d] - base_posterior[d] for d in DISEASES}
        what_if_results[sym] = {
            'posterior': new_posterior,
            'winner':    new_winner,
            'delta':     delta,
        }

        if verbose:
            print(f"   +{sym:13s}", end='')
            for d in DISEASES:
                diff = delta[d]
                sign = '+' if diff >= 0 else ''
                print(f"  {sign}{diff:+.3f}    ", end='')
            changed = " ← winner đổi!" if new_winner != base_winner else ""
            print(f"  {new_winner}{changed}")

    return what_if_results


def sensitivity_analysis(verbose=True):
    """
    Phân tích sensitivity: symptom nào ảnh hưởng nhiều nhất
    đến posterior khi thêm vào từ prior.
    """
    if verbose:
        print("\n" + "="*60)
        print("   SENSITIVITY ANALYSIS")
        print("   Symptom nào informative nhất khi chẩn đoán?")
        print("="*60)

    prior_posterior = {d: p for d, p in DISEASE_PRIOR.items()}
    sensitivity     = {}

    for sym in SYMPTOMS:
        # Thêm symptom=True từ no evidence
        post_true  = run_inference({sym: True},  verbose=False)
        post_false = run_inference({sym: False}, verbose=False)

        # Measure: max absolute change across diseases
        max_change_true  = max(abs(post_true[d]  - prior_posterior[d]) for d in DISEASES)
        max_change_false = max(abs(post_false[d] - prior_posterior[d]) for d in DISEASES)
        sensitivity[sym] = round((max_change_true + max_change_false) / 2, 4)

    ranked = sorted(sensitivity.items(), key=lambda x: -x[1])

    if verbose:
        print(f"\n   Ranked by informativeness:")
        for i, (sym, score) in enumerate(ranked, 1):
            bar = '█' * int(score * 60)
            print(f"   {i}. {sym:14s}: {score:.4f}  {bar}")
        print(f"\n   → {ranked[0][0]} là triệu chứng informative nhất")
        print(f"   → {ranked[-1][0]} là triệu chứng ít informative nhất")

    return sensitivity


if __name__ == '__main__':
    # 1. Evaluation
    evaluate()

    # 2. What-if
    what_if_analysis({'Fever': True, 'Cough': True})

    # 3. Sensitivity
    sensitivity_analysis()
