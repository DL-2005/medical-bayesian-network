import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.cpt_values import DISEASES, SYMPTOMS, SYMPTOM_CPT
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def plot_bn(evidence: dict = None, save_path: str = None):
    """
    Vẽ DAG của Bayesian Network.
    evidence: dict các symptom đã observe, ví dụ {'Fever': True, 'Cough': True}
    """
    G = nx.DiGraph()

    # ── Nodes ──
    G.add_node('Disease')
    for s in SYMPTOMS:
        G.add_node(s)
        G.add_edge('Disease', s)

    # ── Layout: Disease ở trên, symptoms ở dưới ──
    pos = {}
    pos['Disease'] = (0, 1)
    n = len(SYMPTOMS)
    for i, s in enumerate(SYMPTOMS):
        x = (i - (n - 1) / 2) * 1.8
        pos[s] = (x, 0)

    # ── Màu sắc node ──
    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        if node == 'Disease':
            node_colors.append('#4C72B0')   # xanh đậm
            node_sizes.append(3500)
        elif evidence and node in evidence:
            if evidence[node] is True:
                node_colors.append('#55A868')   # xanh lá — observed True
            else:
                node_colors.append('#C44E52')   # đỏ — observed False
            node_sizes.append(2200)
        else:
            node_colors.append('#8172B2')       # tím — unobserved
            node_sizes.append(2000)

    # ── Vẽ graph ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_facecolor('#F8F8F8')
    fig.patch.set_facecolor('#F8F8F8')

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#888888',
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        width=1.5,
        connectionstyle='arc3,rad=0.0',
        min_source_margin=30,
        min_target_margin=25,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=9,
        font_color='white',
        font_weight='bold',
    )

    # ── CPT annotations dưới mỗi symptom node ──
    for s in SYMPTOMS:
        x, y = pos[s]
        lines = []
        for d in DISEASES:
            p = SYMPTOM_CPT[s][d]
            lines.append(f"{d[:3]}: {p:.2f}")
        annotation = '\n'.join(lines)
        ax.text(
            x, y - 0.18,
            annotation,
            ha='center', va='top',
            fontsize=6.5,
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85)
        )

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(color='#4C72B0', label='Disease node (query)'),
        mpatches.Patch(color='#55A868', label='Symptom = True (evidence)'),
        mpatches.Patch(color='#C44E52', label='Symptom = False (evidence)'),
        mpatches.Patch(color='#8172B2', label='Symptom (unobserved)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=8, framealpha=0.9)

    # ── Title ──
    if evidence:
        obs = [f"{k}={'T' if v else 'F'}" for k, v in evidence.items()]
        title = f"Medical BN — Evidence: {', '.join(obs)}"
    else:
        title = "Medical Bayesian Network — DAG Structure"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == '__main__':
    # Test 1: Không có evidence
    print("Vẽ BN graph không có evidence...")
    plot_bn(save_path='bn_graph_empty.png')

    # Test 2: Có evidence
    print("Vẽ BN graph với evidence...")
    plot_bn(
        evidence={'Fever': True, 'Cough': True, 'Dyspnea': True},
        save_path='bn_graph_evidence.png'
    )
    print("Done!")