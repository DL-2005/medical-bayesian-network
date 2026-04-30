import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from inference.engine import run_inference, find_mpe, explain_disease, compare_diseases
from inference.evaluation import what_if_analysis, sensitivity_analysis
from data.cpt_values import DISEASES, SYMPTOMS, SYMPTOM_CPT, DISEASE_PRIOR

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Medical BN Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem; font-weight: 700;
    color: #1a1a2e; text-align: center; margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem; color: #666;
    text-align: center; margin-bottom: 2rem;
}
.disease-card {
    padding: 1rem; border-radius: 10px;
    border: 1px solid #e0e0e0; margin: 0.5rem 0;
    background: #f9f9f9;
}
.highlight { background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.5rem 1rem; }
.trace-box { background: #1e1e1e; color: #d4d4d4; padding: 1rem;
             border-radius: 8px; font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── HELPER: Vẽ BN graph ──────────────────────────────────
def draw_bn_graph(evidence):
    G = nx.DiGraph()
    G.add_node('Disease')
    for s in SYMPTOMS:
        G.add_node(s)
        G.add_edge('Disease', s)

    pos = {'Disease': (0, 1)}
    n = len(SYMPTOMS)
    for i, s in enumerate(SYMPTOMS):
        pos[s] = ((i - (n-1)/2) * 1.8, 0)

    colors = []
    sizes  = []
    for node in G.nodes():
        if node == 'Disease':
            colors.append('#4C72B0'); sizes.append(3500)
        elif evidence.get(node) is True:
            colors.append('#55A868'); sizes.append(2200)
        elif evidence.get(node) is False:
            colors.append('#C44E52'); sizes.append(2200)
        else:
            colors.append('#8172B2'); sizes.append(2000)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#aaaaaa',
                           arrows=True, arrowsize=18, width=1.5,
                           min_source_margin=28, min_target_margin=22)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5,
                            font_color='white', font_weight='bold')

    for s in SYMPTOMS:
        x, y = pos[s]
        lines = [f"{d[:3]}: {SYMPTOM_CPT[s][d]:.2f}" for d in DISEASES]
        ax.text(x, y-0.18, '\n'.join(lines), ha='center', va='top',
                fontsize=6, color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.85))

    legend = [
        mpatches.Patch(color='#4C72B0', label='Disease (query)'),
        mpatches.Patch(color='#55A868', label='Observed = True'),
        mpatches.Patch(color='#C44E52', label='Observed = False'),
        mpatches.Patch(color='#8172B2', label='Unobserved'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=7.5)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ── HELPER: Build reasoning trace string ─────────────────
def build_trace(evidence, posterior):
    observed = {s: v for s, v in evidence.items() if s in SYMPTOMS}
    missing  = [s for s in SYMPTOMS if s not in observed]
    total    = sum(posterior.values())

    lines = []
    lines.append("═"*55)
    lines.append("  VARIABLE ELIMINATION — Reasoning Trace")
    lines.append("═"*55)
    lines.append(f"\n📋 BƯỚC 1 — Instantiate Evidence")
    lines.append(f"   Quan sát  : {observed}")
    lines.append(f"   Bỏ qua    : {missing} (marginalize → = 1)")

    lines.append(f"\n📊 BƯỚC 2 — Factor Product")
    lines.append(f"   score(d) = P(d) × ∏ P(symptom|d)\n")

    scores = {}
    for d in DISEASES:
        prior = DISEASE_PRIOR[d]
        score = prior
        factors = [f"P({d})={prior:.4f}"]
        for s, v in observed.items():
            p_true = SYMPTOM_CPT[s][d]
            p = p_true if v else (1 - p_true)
            label = f"P({s}={'T' if v else 'F'}|{d})={p:.4f}"
            factors.append(label)
            score *= p
        scores[d] = score
        lines.append(f"   {d}:")
        lines.append(f"     = {' × '.join(factors)}")
        lines.append(f"     = {score:.6f}\n")

    total_score = sum(scores.values())
    lines.append(f"📐 BƯỚC 3 — Normalize")
    lines.append(f"   Tổng = {total_score:.6f}\n")
    for d, score in scores.items():
        p = score / total_score
        lines.append(f"   P({d}|evi) = {score:.6f}/{total_score:.6f} = {p:.4f}")

    return '\n'.join(lines)


# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🏥 Medical Bayesian Network</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống chẩn đoán bệnh hô hấp dựa trên Bayesian Network & Variable Elimination</div>', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Chẩn đoán",
    "📊 BN Graph",
    "🔬 What-if Analysis",
    "📈 Sensitivity"
])

# ════════════════════════════════
# TAB 1: CHẨN ĐOÁN
# ════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📋 Nhập triệu chứng")
        st.caption("Chọn triệu chứng bệnh nhân đang có")

        symptom_labels = {
    'Fever':          '🌡️ Sốt (Fever)',
    'Cough':          '😮‍💨 Ho (Cough)',
    'Fatigue':        '😴 Mệt mỏi (Fatigue)',
    'SoreThroat':     '🤒 Đau họng (Sore Throat)',
    'Dyspnea':        '🫁 Khó thở (Dyspnea)',
    'RunnyNose':      '🤧 Sổ mũi (Runny Nose)',
    'ChestPain':      '💔 Đau ngực (Chest Pain)',
    'Chills':         '🥶 Ớn lạnh (Chills)',
    'MuscleAches':    '💪 Đau cơ (Muscle Aches)',
    'Headache':       '🤕 Đau đầu (Headache)',
    'Phlegm':         '🫧 Đờm (Phlegm)',
    'WeightLoss':     '⚖️ Sụt cân (Weight Loss)',
    'LossOfAppetite': '🍽️ Chán ăn (Loss of Appetite)',
    'NightSweats':    '💧 Đổ mồ hôi đêm (Night Sweats)',
    'WateringEyes':   '👁️ Chảy nước mắt (Watering Eyes)',
}

        evidence = {}
        for sym, label in symptom_labels.items():
            val = st.selectbox(
                label,
                options=['— Không chọn —', 'Có ✅', 'Không ❌'],
                key=f"sym_{sym}"
            )
            if val == 'Có ✅':
                evidence[sym] = True
            elif val == 'Không ❌':
                evidence[sym] = False

        diagnose_btn = st.button("🔍 Chẩn đoán", type="primary", use_container_width=True)

    with col_right:
        if diagnose_btn and evidence:
            st.subheader("📊 Kết quả chẩn đoán")

            posterior = run_inference(evidence, verbose=False)
            ranked    = sorted(posterior.items(), key=lambda x: -x[1])
            winner    = ranked[0][0]

            # ── Bar chart kết quả ──
            colors_bar = ['#2ecc71' if d == winner else '#3498db' for d, _ in ranked]
            fig_bar = go.Figure(go.Bar(
                x=[p*100 for _, p in ranked],
                y=[d for d, _ in ranked],
                orientation='h',
                marker_color=colors_bar,
                text=[f"{p*100:.1f}%" for _, p in ranked],
                textposition='outside',
            ))
            fig_bar.update_layout(
                xaxis_title="Xác suất (%)",
                xaxis_range=[0, 110],
                height=280,
                margin=dict(l=10, r=40, t=10, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Winner card ──
            st.success(f"🏆 Chẩn đoán: **{winner}** ({posterior[winner]*100:.1f}%)")

            # ── Explain winning disease ──
            with st.expander(f"📖 Giải thích triệu chứng của {winner}", expanded=True):
                exp = explain_disease(winner, verbose=False)
                cols = st.columns(3)
                for i, (sym, p) in enumerate(exp.items()):
                    with cols[i % 3]:
                        color = "🔴" if p >= 0.80 else "🟠" if p >= 0.55 else "🟡" if p >= 0.30 else "🟢"
                        st.metric(label=sym, value=f"{p*100:.0f}%", delta=color)

            # ── Reasoning Trace ──
            with st.expander("🧠 Reasoning Trace — Variable Elimination từng bước"):
                trace = build_trace(evidence, posterior)
                st.code(trace, language=None)

        elif diagnose_btn and not evidence:
            st.warning("⚠️ Vui lòng chọn ít nhất 1 triệu chứng!")
        else:
            st.info("👈 Chọn triệu chứng và bấm **Chẩn đoán**")


# ════════════════════════════════
# TAB 2: BN GRAPH
# ════════════════════════════════
with tab2:
    st.subheader("🕸️ Bayesian Network Graph")
    st.caption("Node xanh lá = evidence True | Node đỏ = evidence False | Node tím = unobserved")

    # Lấy evidence từ tab 1 nếu có
    current_evidence = {
        sym: (True if st.session_state.get(f"sym_{sym}") == 'Có ✅'
              else False if st.session_state.get(f"sym_{sym}") == 'Không ❌'
              else None)
        for sym in SYMPTOMS
    }
    current_evidence = {k: v for k, v in current_evidence.items() if v is not None}

    fig_graph = draw_bn_graph(current_evidence)
    st.pyplot(fig_graph)

    st.markdown("**CPT — P(Symptom=True | Disease):**")
    import pandas as pd
    cpt_df = pd.DataFrame(SYMPTOM_CPT).T
    cpt_df.columns = DISEASES
    st.dataframe(cpt_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1),
                 use_container_width=True)


# ════════════════════════════════
# TAB 3: WHAT-IF
# ════════════════════════════════
with tab3:
    st.subheader("🔬 What-if Analysis")
    st.caption("Thêm 1 triệu chứng vào evidence → xác suất thay đổi thế nào?")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        base_syms = st.multiselect(
            "Base evidence (có triệu chứng):",
            options=SYMPTOMS,
            default=['Fever', 'Cough']
        )
        run_whatif = st.button("▶️ Phân tích", type="primary")

    with col_b:
        if run_whatif and base_syms:
            base_ev = {s: True for s in base_syms}
            base_post = run_inference(base_ev, verbose=False)

            st.markdown("**Base result:**")
            base_fig = go.Figure(go.Bar(
                x=list(base_post.values()),
                y=list(base_post.keys()),
                orientation='h',
                marker_color='#3498db',
                text=[f"{p*100:.1f}%" for p in base_post.values()],
                textposition='outside',
            ))
            base_fig.update_layout(height=200, margin=dict(l=5,r=40,t=5,b=5),
                                   xaxis_range=[0,1.1])
            st.plotly_chart(base_fig, use_container_width=True)

            wi_results = what_if_analysis(base_ev, verbose=False)

            st.markdown("**Nếu thêm từng triệu chứng = True:**")
            for sym, res in wi_results.items():
                delta   = res['delta']
                winner  = res['winner']
                base_winner = max(base_post, key=base_post.get)
                changed = " 🔄 **Winner đổi!**" if winner != base_winner else ""
                with st.expander(f"+ {sym} → **{winner}**{changed}"):
                    delta_fig = go.Figure(go.Bar(
                        x=list(delta.values()),
                        y=list(delta.keys()),
                        orientation='h',
                        marker_color=['#2ecc71' if v >= 0 else '#e74c3c'
                                      for v in delta.values()],
                        text=[f"{v:+.3f}" for v in delta.values()],
                        textposition='outside',
                    ))
                    delta_fig.update_layout(
                        height=180, margin=dict(l=5,r=60,t=5,b=5),
                        xaxis_title="Δ probability",
                    )
                    st.plotly_chart(delta_fig, use_container_width=True)


# ════════════════════════════════
# TAB 4: SENSITIVITY
# ════════════════════════════════
with tab4:
    st.subheader("📈 Sensitivity Analysis")
    st.caption("Triệu chứng nào informative nhất khi chẩn đoán?")

    sens = sensitivity_analysis(verbose=False)
    ranked_sens = sorted(sens.items(), key=lambda x: -x[1])

    sens_fig = go.Figure(go.Bar(
        x=[s[1] for s in ranked_sens],
        y=[s[0] for s in ranked_sens],
        orientation='h',
        marker_color='#9b59b6',
        text=[f"{s[1]:.4f}" for s in ranked_sens],
        textposition='outside',
    ))
    sens_fig.update_layout(
        xaxis_title="Informativeness score",
        height=350,
        margin=dict(l=10, r=80, t=10, b=30),
    )
    st.plotly_chart(sens_fig, use_container_width=True)

    st.info(f"🏆 **{ranked_sens[0][0]}** là triệu chứng informative nhất "
            f"(score={ranked_sens[0][1]:.4f}) — "
            f"thêm triệu chứng này vào evidence sẽ thay đổi chẩn đoán nhiều nhất.")

    # Per-symptom detail
    st.markdown("**Chi tiết: P(Disease | symptom=True) vs prior:**")
    for sym, score in ranked_sens:
        with st.expander(f"{sym} (score={score:.4f})"):
            post_t = run_inference({sym: True},  verbose=False)
            post_f = run_inference({sym: False}, verbose=False)
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Bar(
                name='Prior', x=DISEASES,
                y=[DISEASE_PRIOR[d] for d in DISEASES],
                marker_color='#95a5a6'
            ))
            comp_fig.add_trace(go.Bar(
                name=f'{sym}=True', x=DISEASES,
                y=[post_t[d] for d in DISEASES],
                marker_color='#2ecc71'
            ))
            comp_fig.add_trace(go.Bar(
                name=f'{sym}=False', x=DISEASES,
                y=[post_f[d] for d in DISEASES],
                marker_color='#e74c3c'
            ))
            comp_fig.update_layout(
                barmode='group', height=250,
                margin=dict(l=5, r=5, t=5, b=30),
            )
            st.plotly_chart(comp_fig, use_container_width=True)
