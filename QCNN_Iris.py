import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# QCNN for Iris dataset (2 classes, 4 features -> 4 qubits)
def load_and_process_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    mask = y != 2
    X = X[mask]
    y = y[mask]

    y = np.where(y == 0, -1.0, 1.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_process_data()

n_qubits = 4
dev = qml.device("lightning.gpu", wires=n_qubits) #If GPU is not available, change to "default.qubit"

# 2 qubit convolution
def conv_circuit(params, wires):
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[1])

# pool
def pool_circuit(params, wires):
    qml.CRZ(params[0], wires=wires)
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=wires)


@qml.qnode(dev)
def qcnn_qnode(inputs, params):

    qml.AngleEmbedding(inputs, wires=range(n_qubits))# Embedding



    conv_circuit(params[0], wires=[0, 1])
    conv_circuit(params[1], wires=[2, 3])

    conv_circuit(params[2], wires=[1, 2])
    conv_circuit(params[3], wires=[3, 0])

    pool_circuit(params[4], wires=[0, 1])
    pool_circuit(params[5], wires=[2, 3])

    conv_circuit(params[6], wires=[0, 2])

    pool_circuit(params[7], wires=[0, 2])

    return qml.expval(qml.PauliZ(0))


weight_shapes = {
    "params": (8, 12)# 8 blocks, each with 12 parameters, pool use first 2
}

# model
class QCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qcnn_layer = qml.qnn.TorchLayer(qcnn_qnode, weight_shapes)
        
    def forward(self, x):
        return self.qcnn_layer(x)


# traub
def train_model(model, X, y, epochs=50, lr=0.01):
    criterion = nn.MSELoss()
    b1 = 0.9
    b2 = 0.999
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(b1, b2))
    
    loss_history = []
    
    print("Start training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(X)
        
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    return loss_history

# evaluation
def evaluate_model(model, X, y):
    with torch.no_grad():
        predictions = model(X)
        

        predicted_classes = torch.sign(predictions)# If > 0, predict 1; otherwise -1
        
        correct = (predicted_classes == y).float().sum()
        accuracy = correct / y.shape[0]
        
    print(f"\nTest accuracy: {accuracy:.4f} ({int(correct)}/{y.shape[0]})")
    
    print("\n--- Example predictions ---")
    for i in range(5):
        print(f"True label: {y[i].item():.0f}, Predicted value: {predictions[i].item():.4f}, Predicted class: {predicted_classes[i].item():.0f}")


if __name__ == "__main__":
    model = QCNNModel()
    
    train_model(model, X_train, y_train, epochs=60, lr=0.05)
    
    evaluate_model(model, X_test, y_test)