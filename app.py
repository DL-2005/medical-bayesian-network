import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from inference.engine import run_inference, explain_disease
from inference.evaluation import what_if_analysis, sensitivity_analysis
from data.cpt_values import DISEASES, SYMPTOMS, SYMPTOM_CPT, DISEASE_PRIOR

st.set_page_config(page_title="Medical BN Diagnosis", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #1a3a6e, #2980b9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem; padding-top: 1rem;
}
.sub-title { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
.winner-card {
    background: linear-gradient(135deg, #1a3a6e 0%, #2980b9 100%);
    border-radius: 16px; padding: 1.5rem 2rem; color: white;
    text-align: center; margin: 1rem 0; box-shadow: 0 8px 32px rgba(26,58,110,0.25);
}
.winner-card h2 { font-size: 1.8rem; margin: 0; font-weight: 700; }
.winner-card p  { font-size: 1rem; opacity: 0.85; margin: 0.3rem 0 0 0; }
.metric-card {
    background: #f8faff; border-radius: 12px; border: 1px solid #e0e8ff;
    padding: 0.8rem 1rem; text-align: center; margin: 0.3rem 0;
}
.metric-card .val { font-size: 1.4rem; font-weight: 700; color: #1a3a6e; }
.metric-card .lbl { font-size: 0.75rem; color: #888; margin-top: 0.1rem; }
.section-header {
    font-size: 1rem; font-weight: 600; color: #1a3a6e;
    border-left: 3px solid #2980b9; padding-left: 0.7rem; margin: 1rem 0 0.5rem 0;
}
.empty-state {
    background:#f0f4ff; border-radius:14px; padding:2rem;
    text-align:center; margin-top:2rem; border:1px dashed #c0d0f0;
}
</style>
""", unsafe_allow_html=True)

DISEASE_COLORS = {
    'Flu':'#e74c3c','Cold':'#3498db','COVID':'#e67e22','Pneumonia':'#8e44ad',
    'Asthma':'#27ae60','Tuberculosis':'#c0392b','Allergy':'#16a085','Bronchitis':'#d35400',
}
DISEASE_EMOJI = {
    'Flu':'🤧','Cold':'🥶','COVID':'🦠','Pneumonia':'🫁',
    'Asthma':'💨','Tuberculosis':'🧬','Allergy':'🌿','Bronchitis':'😮\u200d💨'
}

def draw_bn_graph_plotly(evidence):
    row1 = SYMPTOMS[:8]
    row2 = SYMPTOMS[8:]
    pos = {'Disease': (0, 2)}
    for i, s in enumerate(row1):
        pos[s] = ((i - (len(row1)-1)/2) * 1.6, 1)
    for i, s in enumerate(row2):
        pos[s] = ((i - (len(row2)-1)/2) * 1.6, 0)

    edge_x, edge_y = [], []
    for s in SYMPTOMS:
        x0,y0 = pos['Disease']
        x1,y1 = pos[s]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.2, color='#b0bec5'), hoverinfo='none')

    nx_list, ny_list, nt_list, nc_list, ns_list, nh_list = [],[],[],[],[],[]
    for node in ['Disease'] + SYMPTOMS:
        x,y = pos[node]
        nx_list.append(x); ny_list.append(y)
        if node == 'Disease':
            nc_list.append('#1a3a6e'); ns_list.append(44)
            nt_list.append('<b>Disease</b>')
            nh_list.append('<b>Disease Node</b><br>8 possible states<br>This is the query variable')
        elif evidence.get(node) is True:
            nc_list.append('#27ae60'); ns_list.append(34)
            nt_list.append(f'<b>{node}</b>')
            vals = '<br>'.join([f'{d}: {SYMPTOM_CPT[node][d]:.3f}' for d in DISEASES])
            nh_list.append(f'<b>{node}</b><br>Evidence: TRUE ✅<br><br>P({node}=T|D):<br>{vals}')
        elif evidence.get(node) is False:
            nc_list.append('#e74c3c'); ns_list.append(34)
            nt_list.append(f'<b>{node}</b>')
            vals = '<br>'.join([f'{d}: {SYMPTOM_CPT[node][d]:.3f}' for d in DISEASES])
            nh_list.append(f'<b>{node}</b><br>Evidence: FALSE ❌<br><br>P({node}=T|D):<br>{vals}')
        else:
            nc_list.append('#7f8c8d'); ns_list.append(30)
            nt_list.append(node)
            vals = '<br>'.join([f'{d}: {SYMPTOM_CPT[node][d]:.3f}' for d in DISEASES])
            nh_list.append(f'<b>{node}</b><br>Unobserved<br><br>P({node}=T|D):<br>{vals}')

    node_trace = go.Scatter(
        x=nx_list, y=ny_list, mode='markers+text',
        hoverinfo='text', hovertext=nh_list,
        text=nt_list, textposition='middle center',
        textfont=dict(size=8.5, color='white', family='Inter'),
        marker=dict(size=ns_list, color=nc_list,
                    line=dict(width=2, color='white'), symbol='circle')
    )

    annots = [
        dict(x=0.01,y=0.99,text='🔵 Disease (query)',showarrow=False,xref='paper',yref='paper',font=dict(size=10),align='left'),
        dict(x=0.01,y=0.93,text='🟢 Observed = True',showarrow=False,xref='paper',yref='paper',font=dict(size=10),align='left'),
        dict(x=0.01,y=0.87,text='🔴 Observed = False',showarrow=False,xref='paper',yref='paper',font=dict(size=10),align='left'),
        dict(x=0.01,y=0.81,text='⚫ Unobserved',showarrow=False,xref='paper',yref='paper',font=dict(size=10),align='left'),
    ]

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, hovermode='closest', annotations=annots,
        margin=dict(l=10,r=10,t=10,b=10), height=500,
        paper_bgcolor='#f8faff', plot_bgcolor='#f8faff',
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
    )
    return fig

def build_trace(evidence, posterior):
    observed = {s: v for s, v in evidence.items() if s in SYMPTOMS}
    missing  = [s for s in SYMPTOMS if s not in observed]
    lines = []
    lines.append("="*60)
    lines.append("  VARIABLE ELIMINATION — Reasoning Trace")
    lines.append("="*60)
    lines.append(f"\nBUOC 1 — Instantiate Evidence")
    lines.append(f"  Quan sat : {list(observed.keys())}")
    lines.append(f"  Bo qua   : {missing}")
    lines.append(f"  (Sum P(S|d) = 1 -> marginalize out -> no effect on inference)")
    lines.append(f"\nBUOC 2 — Factor Product")
    lines.append(f"  score(d) = P(d) x prod P(symptom_i=v_i | d)\n")
    scores = {}
    for d in DISEASES:
        prior = DISEASE_PRIOR[d]; score = prior
        factors = [f"P({d})={prior:.4f}"]
        for s, v in observed.items():
            p_t = SYMPTOM_CPT[s][d]; p = p_t if v else (1-p_t)
            factors.append(f"P({s}={'T' if v else 'F'}|{d})={p:.4f}")
            score *= p
        scores[d] = score
        lines.append(f"  {d}:"); lines.append(f"    = {' x '.join(factors)}")
        lines.append(f"    = {score:.8f}\n")
    Z = sum(scores.values())
    lines.append(f"BUOC 3 — Normalize  (Z = {Z:.8f})\n")
    for d, sc in sorted(scores.items(), key=lambda x: -x[1]):
        p = sc/Z; bar = '|' * int(p*30)
        lines.append(f"  {d:14s}: {p:.4f}  {bar}")
    lines.append(f"\nMPE = {max(scores,key=scores.get)} ({max(scores.values())/Z*100:.1f}%)")
    return '\n'.join(lines)

# ════ MAIN APP ════
st.markdown('<div class="main-title">🏥 Medical Bayesian Network</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống chẩn đoán bệnh hô hấp · Bayesian Network + Variable Elimination</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Chẩn đoán","🕸️ BN Graph","🔬 What-if","📈 Sensitivity"])

# ══ TAB 1 ══
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")
    with col_left:
        st.markdown('<div class="section-header">Nhập triệu chứng</div>', unsafe_allow_html=True)
        symptom_labels = {
            'Fever':'🌡️ Sốt','Cough':'😮\u200d💨 Ho','Fatigue':'😴 Mệt mỏi',
            'SoreThroat':'🤒 Đau họng','Dyspnea':'🫁 Khó thở','RunnyNose':'🤧 Sổ mũi',
            'ChestPain':'💔 Đau ngực','Chills':'🥶 Ớn lạnh','MuscleAches':'💪 Đau cơ',
            'Headache':'🤕 Đau đầu','Phlegm':'🫧 Đờm','WeightLoss':'⚖️ Sụt cân',
            'LossOfAppetite':'🍽️ Chán ăn','NightSweats':'💧 Đổ mồ hôi đêm','WateringEyes':'👁️ Chảy nước mắt',
        }
        evidence = {}
        for sym, label in symptom_labels.items():
            val = st.select_slider(label, options=['❌','—','✅'], value='—', key=f"sym_{sym}")
            if val == '✅': evidence[sym] = True
            elif val == '❌': evidence[sym] = False
        st.markdown("")
        diagnose_btn = st.button("🔍 Chẩn đoán ngay", type="primary", use_container_width=True)

    with col_right:
        if diagnose_btn and evidence:
            posterior = run_inference(evidence, verbose=False)
            ranked = sorted(posterior.items(), key=lambda x: -x[1])
            winner = ranked[0][0]
            pct = posterior[winner]*100
            st.markdown(f'<div class="winner-card"><p>Chẩn đoán hàng đầu</p><h2>{DISEASE_EMOJI.get(winner,"")} {winner}</h2><p>Xác suất posterior: <b>{pct:.1f}%</b></p></div>', unsafe_allow_html=True)

            fig_bar = go.Figure(go.Bar(
                x=[p*100 for _,p in ranked],
                y=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d,_ in ranked],
                orientation='h',
                marker=dict(color=[DISEASE_COLORS.get(d,'#95a5a6') for d,_ in ranked],
                            opacity=[1.0 if d==winner else 0.55 for d,_ in ranked]),
                text=[f"{p*100:.1f}%" for _,p in ranked], textposition='outside',
                textfont=dict(size=12),
            ))
            fig_bar.update_layout(
                xaxis=dict(title="Xác suất (%)", range=[0,118], gridcolor='#f0f0f0'),
                yaxis=dict(autorange='reversed'), height=320,
                margin=dict(l=10,r=60,t=10,b=30),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander(f"📖 Profile triệu chứng của {winner}", expanded=True):
                exp = explain_disease(winner, verbose=False)
                c1,c2,c3 = st.columns(3)
                cols_c = [c1,c2,c3]
                for i,(sym,p) in enumerate(exp.items()):
                    em = '🔴' if p>=0.80 else '🟠' if p>=0.55 else '🟡' if p>=0.30 else '🟢'
                    with cols_c[i%3]:
                        st.markdown(f'<div class="metric-card"><div class="val">{p*100:.0f}%</div><div class="lbl">{em} {sym}</div></div>', unsafe_allow_html=True)

            with st.expander("🧠 Reasoning Trace — VE từng bước"):
                st.code(build_trace(evidence, posterior), language=None)

        elif diagnose_btn:
            st.warning("⚠️ Vui lòng chọn ít nhất 1 triệu chứng!")
        else:
            st.markdown('<div class="empty-state"><div style="font-size:2.5rem">🩺</div><div style="font-size:1.1rem;color:#1a3a6e;font-weight:600;margin-top:0.5rem">Sẵn sàng chẩn đoán</div><div style="color:#888;margin-top:0.3rem">Chọn triệu chứng bên trái và bấm <b>Chẩn đoán ngay</b></div></div>', unsafe_allow_html=True)

# ══ TAB 2 ══
with tab2:
    st.markdown('<div class="section-header">Bayesian Network — DAG Structure</div>', unsafe_allow_html=True)
    st.caption("Hover vào node để xem CPT values · Màu node phản ánh evidence từ Tab Chẩn đoán")
    cur_ev = {s:(True if st.session_state.get(f"sym_{s}")=='✅' else False if st.session_state.get(f"sym_{s}")=='❌' else None) for s in SYMPTOMS}
    cur_ev = {k:v for k,v in cur_ev.items() if v is not None}
    st.plotly_chart(draw_bn_graph_plotly(cur_ev), use_container_width=True)

    st.markdown('<div class="section-header">CPT Heatmap — P(Symptom=True | Disease)</div>', unsafe_allow_html=True)
    cpt_df = pd.DataFrame(SYMPTOM_CPT).T
    cpt_df.columns = DISEASES
    fig_heat = go.Figure(go.Heatmap(
        z=cpt_df.values, x=DISEASES, y=list(cpt_df.index),
        colorscale='RdYlGn', zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in cpt_df.values],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="P(S=T|D)", thickness=15),
    ))
    fig_heat.update_layout(
        height=520, margin=dict(l=130,r=20,t=20,b=60),
        xaxis=dict(side='top', tickfont=dict(size=11)),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
        paper_bgcolor='#f8faff', plot_bgcolor='#f8faff',
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══ TAB 3 ══
with tab3:
    st.markdown('<div class="section-header">What-if Analysis</div>', unsafe_allow_html=True)
    st.caption("Thêm 1 triệu chứng vào evidence → xác suất thay đổi thế nào?")
    col_a, col_b = st.columns([1,2], gap="large")
    with col_a:
        base_syms = st.multiselect("Base evidence:", options=SYMPTOMS, default=['Fever','Cough'])
        run_whatif = st.button("▶️ Phân tích", type="primary", use_container_width=True)
    with col_b:
        if run_whatif and base_syms:
            base_ev = {s: True for s in base_syms}
            base_post = run_inference(base_ev, verbose=False)
            base_winner = max(base_post, key=base_post.get)
            fig_base = go.Figure(go.Bar(
                x=list(base_post.values()),
                y=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in base_post],
                orientation='h',
                marker_color=[DISEASE_COLORS.get(d,'#95a5a6') for d in base_post],
                text=[f"{p*100:.1f}%" for p in base_post.values()], textposition='outside',
            ))
            fig_base.update_layout(
                title=f"Base: {base_syms} → winner: {base_winner}",
                height=260, xaxis_range=[0,1.15], margin=dict(l=5,r=60,t=40,b=5),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(fig_base, use_container_width=True)

            wi_results = what_if_analysis(base_ev, verbose=False)
            changed   = [(s,r) for s,r in wi_results.items() if r['winner']!=base_winner]
            unchanged = [(s,r) for s,r in wi_results.items() if r['winner']==base_winner]

            if changed:
                st.markdown("**🔄 Làm đổi winner:**")
                for sym, res in changed:
                    with st.expander(f"✚ {sym} → **{res['winner']}** 🔄", expanded=True):
                        delta = res['delta']
                        fig_d = go.Figure(go.Bar(
                            x=list(delta.values()),
                            y=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in delta],
                            orientation='h',
                            marker_color=['#27ae60' if v>=0 else '#e74c3c' for v in delta.values()],
                            text=[f"{v:+.3f}" for v in delta.values()], textposition='outside',
                        ))
                        fig_d.update_layout(height=230, margin=dict(l=5,r=80,t=5,b=5),
                            xaxis_title="Δ probability", paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(autorange='reversed'))
                        st.plotly_chart(fig_d, use_container_width=True)

            st.markdown("**Không đổi winner:**")
            for sym, res in unchanged:
                with st.expander(f"✚ {sym} → {res['winner']}"):
                    delta = res['delta']
                    fig_d = go.Figure(go.Bar(
                        x=list(delta.values()),
                        y=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in delta],
                        orientation='h',
                        marker_color=['#27ae60' if v>=0 else '#e74c3c' for v in delta.values()],
                        text=[f"{v:+.3f}" for v in delta.values()], textposition='outside',
                    ))
                    fig_d.update_layout(height=230, margin=dict(l=5,r=80,t=5,b=5),
                        xaxis_title="Δ probability", paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(autorange='reversed'))
                    st.plotly_chart(fig_d, use_container_width=True)
        elif run_whatif:
            st.warning("Chọn ít nhất 1 triệu chứng!")

# ══ TAB 4 ══
with tab4:
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.caption("Triệu chứng nào informative nhất khi chẩn đoán?")
    sens = sensitivity_analysis(verbose=False)
    ranked_sens = sorted(sens.items(), key=lambda x: -x[1])
    n_s = len(ranked_sens)
    colors_s = [f'rgba(41,128,185,{max(1.0-i*0.055,0.3):.2f})' for i in range(n_s)]
    fig_sens = go.Figure(go.Bar(
        x=[s[1] for s in ranked_sens], y=[s[0] for s in ranked_sens],
        orientation='h', marker_color=colors_s,
        text=[f"{s[1]:.4f}" for s in ranked_sens], textposition='outside',
    ))
    fig_sens.update_layout(
        xaxis=dict(title="Informativeness Score", gridcolor='#f0f0f0', range=[0,0.38]),
        height=440, margin=dict(l=10,r=80,t=10,b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange='reversed'), font=dict(family='Inter'),
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1: st.success(f"🏆 Informative nhất: **{ranked_sens[0][0]}** ({ranked_sens[0][1]:.4f})")
    with c2: st.info(f"📉 Ít nhất: **{ranked_sens[-1][0]}** ({ranked_sens[-1][1]:.4f})")

    st.markdown('<div class="section-header">Prior vs Posterior khi observe từng triệu chứng</div>', unsafe_allow_html=True)
    for sym, score in ranked_sens:
        with st.expander(f"{sym}  —  score {score:.4f}"):
            pt = run_inference({sym:True}, verbose=False)
            pf = run_inference({sym:False}, verbose=False)
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(name='Prior', x=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in DISEASES], y=[DISEASE_PRIOR[d] for d in DISEASES], marker_color='#bdc3c7', opacity=0.8))
            fig_c.add_trace(go.Bar(name=f'{sym}=True', x=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in DISEASES], y=[pt[d] for d in DISEASES], marker_color='#27ae60'))
            fig_c.add_trace(go.Bar(name=f'{sym}=False', x=[f"{DISEASE_EMOJI.get(d,'')} {d}" for d in DISEASES], y=[pf[d] for d in DISEASES], marker_color='#e74c3c'))
            fig_c.update_layout(barmode='group', height=260, margin=dict(l=5,r=5,t=5,b=30),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02), font=dict(family='Inter',size=11))
            st.plotly_chart(fig_c, use_container_width=True)
