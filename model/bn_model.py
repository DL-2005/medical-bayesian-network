import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from data.cpt_values import DISEASE_PRIOR, SYMPTOM_CPT, DISEASES, SYMPTOMS


def build_model():
    edges = [('Disease', symptom) for symptom in SYMPTOMS]
    model = DiscreteBayesianNetwork(edges)

    disease_cpd = TabularCPD(
        variable='Disease',
        variable_card=len(DISEASES),
        values=[[p] for p in DISEASE_PRIOR.values()],
        state_names={'Disease': DISEASES}
    )

    symptom_cpds = []
    for symptom in SYMPTOMS:
        cpt = SYMPTOM_CPT[symptom]
        values = [
            [1 - cpt[d] for d in DISEASES],
            [cpt[d]     for d in DISEASES]
        ]
        cpd = TabularCPD(
            variable=symptom,
            variable_card=2,
            values=values,
            evidence=['Disease'],
            evidence_card=[len(DISEASES)],
            state_names={
                symptom:   [False, True],
                'Disease': DISEASES
            }
        )
        symptom_cpds.append(cpd)

    model.add_cpds(disease_cpd, *symptom_cpds)
    assert model.check_model(), "Model không hợp lệ!"
    print("✅ Model hợp lệ")
    return model
