import pennylane as qml
import torch
import math




# Angle Embedding
def get_ang_entangling_qnode(dev,n_qubits,diff_method):
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def ang_entangling_qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return qml.expval(qml.PauliZ(0))
    return ang_entangling_qnode


# Amplitude Embedding
def get_amp_entangling_qnode(dev,n_qubits,diff_method):
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def amp_entangling_qnode(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return qml.expval(qml.PauliZ(0))
    return amp_entangling_qnode


def get_qcnn_qnode(dev, n_qubits, diff_method):
    assert n_qubits >= 2 and (n_qubits & (n_qubits - 1)) == 0

    n_levels = int(math.log2(n_qubits))

    weight_shapes = {}
    active = n_qubits
    for lvl in range(n_levels):
        n_pairs = active // 2
        weight_shapes[f"conv_{lvl}"] = (n_pairs, 4)
        if lvl < n_levels - 1:
            weight_shapes[f"pool_{lvl}"] = (n_pairs, 2)
        active = active // 2

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def qcnn_circuit(inputs, **weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

        active_wires = list(range(n_qubits))
        for lvl in range(n_levels):
            n_pairs = len(active_wires) // 2
            conv_p = weights[f"conv_{lvl}"]
            for i in range(n_pairs):
                w0, w1 = active_wires[2 * i], active_wires[2 * i + 1]
                qml.RY(conv_p[i, 0], wires=w0)
                qml.RY(conv_p[i, 1], wires=w1)
                qml.CNOT(wires=[w0, w1])
                qml.RY(conv_p[i, 2], wires=w0)
                qml.RY(conv_p[i, 3], wires=w1)

            if lvl < n_levels - 1:
                pool_p = weights[f"pool_{lvl}"]
                src = [active_wires[2 * i + 1] for i in range(n_pairs)]
                snk = [active_wires[2 * i]     for i in range(n_pairs)]
                for i in range(n_pairs):
                    qml.RY(pool_p[i, 0], wires=src[i])
                    qml.CNOT(wires=[src[i], snk[i]])
                    qml.RY(pool_p[i, 1], wires=snk[i])
                active_wires = snk

        return qml.expval(qml.PauliZ(active_wires[0]))

    return qcnn_circuit, weight_shapes