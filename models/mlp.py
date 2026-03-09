import torch
import torch.nn as nn
import pennylane as qml
from .quantum_circuits import get_ang_entangling_qnode
import numpy as np



class HybridMLP(nn.Module):
    def __init__(self, n_qubits, n_layers, input_dim,dev,diff_method):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(get_ang_entangling_qnode(dev, n_qubits,diff_method), weight_shapes)
        self.linear = nn.Linear(n_qubits, 1)
        
    def forward(self, x):
        x = self.pre_net(x)
        x = x * np.pi 
        q_out = self.qlayer(x)
        # return self.linear(q_out)
        return q_out.unsqueeze(1)
    

class ClassicalMLP(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)
