from data.cpt_values import DISEASE_PRIOR, SYMPTOM_CPT, DISEASES, SYMPTOMS


def run_inference(evidence: dict, verbose: bool = True) -> dict:
    """
    Variable Elimination tay cho Naive Bayes BN.

    Args:
        evidence: dict triệu chứng đã observe.
                  Ví dụ: {'Fever': True, 'Cough': True, 'Dyspnea': False}
        verbose:  In reasoning trace hay không

    Returns:
        dict {disease: posterior_probability}
    """

    if verbose:
        _print_header(evidence)

    # ── BƯỚC 1: Instantiate evidence ──
    # Chỉ lấy các symptom được observe (có trong evidence)
    observed = {s: v for s, v in evidence.items() if s in SYMPTOM_CPT}

    if verbose:
        print("\n📋 BƯỚC 1 — Instantiate evidence")
        print(f"   Triệu chứng được quan sát: {observed}")
        missing = [s for s in SYMPTOM_CPT if s not in observed]
        print(f"   Triệu chứng không quan sát (bỏ qua): {missing}")

    # ── BƯỚC 2: Tính factor cho mỗi disease ──
    # score(d) = P(d) × ∏ P(symptom=value | d)
    if verbose:
        print("\n📊 BƯỚC 2 — Tính factor product cho từng disease")
        print(f"   Công thức: score(d) = P(d) × ∏ P(symptom|d)")
        print()

    scores = {}
    for disease in DISEASES:
        prior = DISEASE_PRIOR[disease]
        score = prior
        factors = [f"P({disease})={prior:.4f}"]

        for symptom, value in observed.items():
            p_true = SYMPTOM_CPT[symptom][disease]
            p_symptom = p_true if value else (1 - p_true)
            label = f"P({symptom}={'T' if value else 'F'}|{disease})={p_symptom:.4f}"
            factors.append(label)
            score *= p_symptom

        scores[disease] = score

        if verbose:
            print(f"   {disease}:")
            print(f"     = {' × '.join(factors)}")
            print(f"     = {score:.6f}")
            print()

    # ── BƯỚC 3: Normalize ──
    total = sum(scores.values())

    if verbose:
        print(f"📐 BƯỚC 3 — Normalize")
        print(f"   Tổng scores = {total:.6f}")
        print()

    posterior = {}
    for disease, score in scores.items():
        posterior[disease] = score / total
        if verbose:
            print(f"   P({disease} | evidence) = {score:.6f} / {total:.6f} = {posterior[disease]:.4f}")

    # ── BƯỚC 4: Kết quả ──
    if verbose:
        _print_result(posterior)

    return posterior


def find_mpe(evidence: dict, verbose: bool = True) -> str:
    """
    Most Probable Explanation — tìm disease có xác suất cao nhất.
    """
    posterior = run_inference(evidence, verbose=False)
    best = max(posterior, key=posterior.get)

    if verbose:
        print("\n🏆 MPE (Most Probable Explanation):")
        for d, p in sorted(posterior.items(), key=lambda x: -x[1]):
            bar   = "█" * int(p * 30)
            arrow = " ← best" if d == best else ""
            print(f"   {d:12s} {p:.4f} {bar}{arrow}")

    return best


def _print_header(evidence):
    print("\n" + "="*60)
    print("   BAYESIAN NETWORK INFERENCE — Variable Elimination")
    print("="*60)
    print(f"   Query:    P(Disease | evidence)")
    print(f"   Evidence: {evidence}")
    print("="*60)


def _print_result(posterior):
    print("\n✅ KẾT QUẢ CUỐI:")
    print("-"*40)
    ranked = sorted(posterior.items(), key=lambda x: -x[1])
    for disease, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"   {disease:12s}: {prob:.4f}  {bar}")
    print("-"*40)
    winner = ranked[0]
    print(f"   → Chẩn đoán: {winner[0]} ({winner[1]*100:.1f}%)")


def explain_disease(disease: str, verbose: bool = True) -> dict:
    """
    Chiều ngược: Biết disease, giải thích xác suất từng triệu chứng.

    Args:
        disease: tên bệnh, ví dụ 'COVID', 'Flu'
        verbose: in trace hay không
    Returns:
        dict {symptom: probability}
    """
    if disease not in DISEASES:
        print(f"❌ Bệnh '{disease}' không có trong hệ thống.")
        print(f"   Bệnh hợp lệ: {DISEASES}")
        return {}

    if verbose:
        print("\n" + "="*60)
        print(f"   EXPLAIN — P(Symptom | Disease={disease})")
        print("="*60)
        print(f"\n   Nếu bệnh nhân mắc {disease}, khả năng xuất hiện")
        print(f"   từng triệu chứng là:\n")

    result = {}
    for symptom in SYMPTOMS:
        p = SYMPTOM_CPT[symptom][disease]
        result[symptom] = p

        if verbose:
            bar   = "█" * int(p * 30)
            level = _risk_level(p)
            print(f"   {symptom:12s}: {p:.0%}  {bar}  {level}")

    if verbose:
        print()
        _explain_summary(disease, result)

    return result


def compare_diseases(symptom: str, verbose: bool = True) -> dict:
    """
    So sánh xác suất 1 triệu chứng across tất cả diseases.
    Hữu ích để hiểu tại sao model phân biệt được bệnh.
    """
    if symptom not in SYMPTOMS:
        print(f"❌ Triệu chứng '{symptom}' không có trong hệ thống.")
        return {}

    if verbose:
        print("\n" + "="*60)
        print(f"   COMPARE — P({symptom}=T | Disease)")
        print("="*60)
        print(f"\n   {symptom} xuất hiện với xác suất:\n")

    result = {}
    ranked = sorted(DISEASES, key=lambda d: SYMPTOM_CPT[symptom][d], reverse=True)

    for disease in ranked:
        p = SYMPTOM_CPT[symptom][disease]
        result[disease] = p
        if verbose:
            bar   = "█" * int(p * 30)
            level = _risk_level(p)
            print(f"   {disease:12s}: {p:.0%}  {bar}  {level}")

    if verbose:
        best  = ranked[0]
        worst = ranked[-1]
        print(f"\n   → {symptom} đặc trưng nhất cho {best} ({SYMPTOM_CPT[symptom][best]:.0%})")
        print(f"   → Ít đặc trưng nhất ở {worst} ({SYMPTOM_CPT[symptom][worst]:.0%})")

    return result


def full_report(evidence: dict) -> None:
    """
    Full demo report: inference + explain winning disease + key symptom compare.
    Dùng cho demo UI hoặc in ra báo cáo.
    """
    print("\n" + "█"*60)
    print("   MEDICAL BAYESIAN NETWORK — FULL DIAGNOSTIC REPORT")
    print("█"*60)

    # Bước 1: Inference
    posterior = run_inference(evidence, verbose=True)

    # Bước 2: MPE
    best = max(posterior, key=posterior.get)

    # Bước 3: Explain disease thắng
    print("\n" + "="*60)
    print(f"   EXPLAIN WINNING DIAGNOSIS: {best}")
    print("="*60)
    explain_disease(best, verbose=True)

    # Bước 4: So sánh symptom quan trọng nhất
    # Tìm symptom có evidence=True và phân biệt tốt nhất
    observed_true = [s for s, v in evidence.items() if v is True and s in SYMPTOMS]
    if observed_true:
        # Chọn symptom có variance cao nhất giữa các diseases
        best_symptom = max(
            observed_true,
            key=lambda s: max(SYMPTOM_CPT[s].values()) - min(SYMPTOM_CPT[s].values())
        )
        print("\n" + "="*60)
        print(f"   KEY SYMPTOM ANALYSIS: {best_symptom}")
        print("="*60)
        compare_diseases(best_symptom, verbose=True)

    print("\n" + "█"*60)


# ── Helper functions ──

def _risk_level(p: float) -> str:
    if p >= 0.80:
        return "🔴 rất cao"
    elif p >= 0.55:
        return "🟠 cao"
    elif p >= 0.30:
        return "🟡 trung bình"
    else:
        return "🟢 thấp"


def _explain_summary(disease: str, result: dict):
    high    = [s for s, p in result.items() if p >= 0.70]
    medium  = [s for s, p in result.items() if 0.40 <= p < 0.70]
    low     = [s for s, p in result.items() if p < 0.40]

    print(f"   📌 Tóm tắt cho {disease}:")
    if high:
        print(f"   🔴 Rất hay gặp  : {', '.join(high)}")
    if medium:
        print(f"   🟡 Có thể gặp   : {', '.join(medium)}")
    if low:
        print(f"   🟢 Ít gặp       : {', '.join(low)}")