from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA, SPSA, POWELL, SLSQP
from qiskit.circuit.library import (
    TwoLocal,
    PauliTwoDesign,
    RealAmplitudes,
    EfficientSU2,
    ExcitationPreserving,
    XXPlusYYGate,
    SwapGate,
)

optimizers = {
    "spsa": SPSA(),
    "powell": POWELL(),
    "cobyla": COBYLA(),
    "slsqp": SLSQP(),
}

DEPTH = 5


class Ansatz:
    def __init__(self):
        self.ansatz_list = [
            "two_local",
            "pauli_two_design",
            "real_amplitudes",
            "efficientSU2",
            # "excitation_preserving"
        ]

    def get_ansatz(self, n_qubits: int, ansatz: str, depth: int) -> object:
        if ansatz not in self.ansatz_list:
            raise ValueError(f"Invalid ansatz: {ansatz}")
        if ansatz == "two_local":
            return TwoLocal(
                n_qubits, "ry", "cz", "full", reps=depth, name=f"two_local_d_{depth}"
            )
        if ansatz == "pauli_two_design":
            return PauliTwoDesign(
                n_qubits, reps=depth, name=f"pauli_two_design_d_{depth}"
            )
        if ansatz == "real_amplitudes":
            return RealAmplitudes(
                n_qubits, reps=depth, name=f"real_amplitudes_d_{depth}"
            )
        if ansatz == "efficientSU2":
            return EfficientSU2(n_qubits, reps=depth, name=f"efficientSU2_d_{depth}")
        if ansatz == "excitation_preserving":
            return ExcitationPreserving(
                n_qubits, reps=depth, name=f"excitation_preserving_d_{depth}"
            )


def _ring_mixer(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc.append(XXPlusYYGate(0), [i, i + 1])
    qc.append(XXPlusYYGate(0), [n_qubits - 1, 0])

    return qc


def _swap_ring_mixer(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc.append(SwapGate(), [i, i + 1])
    qc.append(SwapGate(), [n_qubits - 1, 0])

    return qc


def _parity_ring_mixer(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    is_even = n_qubits % 2 == 0
    for i in range(0, n_qubits - 1, 2):
        qc.append(XXPlusYYGate(0), [i, i + 1])
    if not is_even:
        qc.append(XXPlusYYGate(0), [n_qubits - 1, 0])
    for i in range(1, n_qubits - 1, 2):
        qc.append(XXPlusYYGate(0), [i, i + 1])
    if is_even:
        qc.append(XXPlusYYGate(0), [n_qubits - 1, 0])

    return qc


def _full_mixer(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    is_even = n_qubits % 2 == 0
    n = n_qubits - 1 if is_even else n_qubits

    subsets = [[] for _ in range(n)]

    used_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = (i, j)
            if set(pair) in used_pairs:
                continue

            # In the original logic we have i,j = {1,2,3...} so we need to take this into account
            k = ((i + 1) + (j + 1)) % n
            subsets[k - 1].append(pair)

            used_pairs.append(set(pair))

    if is_even:
        for subset in subsets:
            miss_pair = [i for i in range(n_qubits)]
            for pair in subset:
                miss_pair.remove(pair[0])
                miss_pair.remove(pair[1])
            subset.append(tuple(miss_pair))

    for subset in subsets:
        for pair in subset:
            qc.append(XXPlusYYGate(0), list(pair))

    return qc


def _row_swap_mixer(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    added = []
    for i in range(n_qubits - 1):
        for j in range(n_qubits):
            for t in range(n_qubits):
                if i != t:
                    qc.append(XXPlusYYGate(0), [i, t])
                    added.append((i, t))
                if j != t:
                    qc.append(XXPlusYYGate(0), [j, t])
                    added.append((j, t))

    return qc


class Mixers:
    def __init__(self):
        self.mixer_list = [
            "ring_mixer",
            "swap_ring_mixer",
            "parity_ring_mixer",
            "full_mixer",
            "row_swap_mixer",
        ]

    def get_mixer(self, n_qubits: int, mixer: str) -> QuantumCircuit:
        if mixer not in self.mixer_list:
            raise ValueError(f"Invalid mixer: {mixer}")
        if mixer == "ring_mixer":
            return _ring_mixer(n_qubits)
        if mixer == "swap_ring_mixer":
            return _swap_ring_mixer(n_qubits)
        if mixer == "parity_ring_mixer":
            return _parity_ring_mixer(n_qubits)
        if mixer == "full_mixer":
            return _full_mixer(n_qubits)
        if mixer == "row_swap_mixer":
            return _row_swap_mixer(n_qubits)
