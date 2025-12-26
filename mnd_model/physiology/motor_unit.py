from mnd_model.core.nodes import (
    Node,
    cortex_forward,
    neuron_forward,
    lmn_forward,
    nmj_forward,
    muscle_forward
)
from mnd_model.core.edges import Edge


def build_motor_unit():
    cortex = Node("Cortex", {"drive_intensity": 1.0})
    umn = Node("UMN", {
        "health": 1.0,
        "signal_gain": 1.0,
        "degeneration_burden": 0.0
    })
    lmn = Node("LMN", {
        "health": 1.0,
        "signal_gain": 1.0,
        "degeneration_burden": 0.0,
        "excitotoxicity_factor": 0.0
    })
    nmj = Node("NMJ", {"transmission_efficiency": 1.0})
    muscle = Node("Muscle", {
        "activation_level": 0.0,
        "max_force": 1.0
    })

    edges = [
        Edge(cortex, umn),
        Edge(umn, lmn),
        Edge(lmn, nmj),
        Edge(nmj, muscle)
    ]

    return cortex, umn, lmn, nmj, muscle, edges


def forward_pass(cortex, umn, lmn, nmj, muscle, edges):
    signal = cortex_forward(cortex)
    signal = neuron_forward(umn, edges[0].transmit())
    signal = lmn_forward(lmn, edges[1].transmit())
    signal = nmj_forward(nmj, edges[2].transmit())
    force = muscle_forward(muscle, edges[3].transmit())
    return force
