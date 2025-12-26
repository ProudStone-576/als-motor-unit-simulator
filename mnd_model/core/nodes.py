import numpy as np

class Node:
    def __init__(self, name, state):
        self.name = name
        self.state = state
        self.input_signal = 0.0
        self.output_signal = 0.0


def cortex_forward(node):
    output = node.state["drive_intensity"]
    node.output_signal = output
    return output


def neuron_forward(node, signal):
    effective_health = np.clip(
        node.state["health"] * (1.0 - node.state.get("degeneration_burden", 0.0)),
        0.0,
        1.0
    )
    output = signal * effective_health * node.state["signal_gain"]
    node.output_signal = output
    return output


def lmn_forward(node, signal):
    stress_threshold = 0.8
    stress = max(0.0, signal - stress_threshold)

    excitotoxic_penalty = stress * node.state.get("excitotoxicity_factor", 0.0)

    effective_health = np.clip(
        node.state["health"] *
        (1.0 - node.state.get("degeneration_burden", 0.0) - excitotoxic_penalty),
        0.0,
        1.0
    )

    output = signal * effective_health * node.state["signal_gain"]
    node.output_signal = output
    return output


def nmj_forward(node, signal):
    output = signal * node.state["transmission_efficiency"]
    node.output_signal = output
    return output


def muscle_forward(node, signal):
    activation = np.clip(signal, 0, 1)
    force = activation * node.state["max_force"]
    node.state["activation_level"] = activation
    node.output_signal = force
    return force
