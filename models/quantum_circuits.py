import pennylane as qml
import torch




# Angle Embedding
def get_ang_entangling_qnode(dev,n_qubits,diff_method="backprop"):
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def ang_entangling_qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return ang_entangling_qnode


# Amplitude Embedding
def get_amp_entangling_qnode(dev,n_qubits,diff_method="backprop"):
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def amp_entangling_qnode(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return amp_entangling_qnode