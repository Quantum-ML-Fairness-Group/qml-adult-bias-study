import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import os
import time
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import dataloader.adult_dataloader as adult_dataloader
import InputEmbeddings
import bias_calculation
from seed import set_seed
import models.quantum_circuits as quantum_circuits
import models.models as models


# Hyperparameters and configuration

# Tips: For #qubits < 20, use CPU with 'default.qubit' device. For >20 qubits, consider using GPU with 'lightning.qubit' device.

enable_wandb = False # Set to False/True to disable/enable wandb logging and run without it

epochs = 1
n_qubits = 7
n_layers = 3 
dev = qml.device('default.qubit', wires=n_qubits) # If using GPU, change to 'lightning.qubit'
diff_method = "backprop"
learning_rate = 0.005
BATCH_SIZE = 64
run_name = f"Qubits-{n_qubits}_Ang_TrnEbding"
seed = 42
device = 'cpu' # If using GPU, set to 'cuda'


#ensure reproducibility
set_seed(seed)


cpu_count = os.cpu_count()
num_workers = 4 if cpu_count > 4 else 0

# Logs
print(f"Using device: {device}")
print(f"Number of CPU cores: {cpu_count}, Setting num_workers to: {num_workers}")
print(f"Hyperparameters: epochs={epochs}, n_qubits={n_qubits}, n_layers={n_layers}, learning_rate={learning_rate}, batch_size={BATCH_SIZE}diff_method={diff_method}")
print(f"Run name: {run_name}")


# Hybrid QNN
class HybridQNN(nn.Module):
    def __init__(self, n_qubits, n_layers,input_dim):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(quantum_circuits.get_ang_entangling_qnode(dev,n_qubits), weight_shapes)
        self.linear = nn.Linear(n_qubits,1)
        
    def forward(self, x):
        x = self.pre_net(x)
        x = x * np.pi 
        q_out = self.qlayer(x)
        return self.linear(q_out)


# Main
if __name__ == "__main__":
    if enable_wandb:
        wandb.init( # Change these if needed
            project="QNN-Adult-MaleFemale-Classification",
            name=run_name,
            config={
                "epochs": epochs,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "learning_rate": learning_rate,
                "batch_size": BATCH_SIZE,
                "diff_method": diff_method,
                "device": device,
                "model": "HybridQNN"
            }
        )
        wandb.define_metric("epoch") 
        wandb.define_metric("*", step_metric="epoch")
        print(f"W&B initialized with run name: {run_name}")

    X_train, X_test, y_train, y_test, sens_train, sens_test = adult_dataloader.load_and_preprocess_data()
    # X_train, X_test = InputEmbeddings.apply_pca(X_train, X_test, n_qubits) # PCA 
    # X_train, X_test, n_qubits = prepare_amplitude_encoding(X_train, X_test) # Amplitude Encoding

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    sens_test_t = sens_test 


    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    model = HybridQNN(n_qubits=n_qubits, n_layers=n_layers, input_dim=X_train.shape[1])

    if enable_wandb:
        wandb.watch(model, log="all", log_freq=10)

        # Log model architecture as text in W&B
        str_model = str(model)
        wandb.log({"model_architecture_text": wandb.Html(f"<pre>{str_model}</pre>")})

        # log quantum circuit
        dummy_inputs = torch.zeros((n_qubits,), dtype=torch.float32)
        dummy_weights = torch.zeros((n_layers, n_qubits, 3), dtype=torch.float32)
        fig, ax = qml.draw_mpl(qnode, expansion_strategy="device")(dummy_inputs, dummy_weights)
        wandb.log({"quantum_circuit": wandb.Image(fig)})


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    
    
    print(f"\nStarting mini-batch training (Batch Size: {BATCH_SIZE}, Qubits: {n_qubits})")
    print(f"Training device: {device}")

    start_train = time.time()

    # Train

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch_X, batch_y in pbar:

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if enable_wandb:
                    wandb.log({"batch_loss": loss.item(), "epoch": epoch})
                pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        if enable_wandb:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    

    # Testing and evaluation
        model.eval()
        print("\nTesting and evaluating bias...")
        with torch.no_grad():
            test_outputs = model(X_test_t)
            predictions = (torch.sigmoid(test_outputs) > 0.5).float().numpy().flatten()
            y_test_numpy = y_test_t.numpy().flatten().astype(int)

        acc = accuracy_score(y_test_numpy, predictions)

        dpd,prob_female,prob_male = bias_calculation.demographic_parity(predictions, sens_test_t)
        eod, tpr_diff, fpr_diff = bias_calculation.equalized_odds(predictions, y_test_numpy, sens_test_t)


        if enable_wandb:
            wandb.log({
                "test_accuracy": acc,
                "prob_female": prob_female,
                "prob_male": prob_male,
                "dpd": dpd,
                "tpr_diff": tpr_diff,
                "fpr_diff": fpr_diff,
                "equalized_odds_diff": eod,
                "epoch": epoch
            })

        print("-" * 30)
        print(f"Model accuracy (Accuracy): {acc:.4f}")
        print("\n[Bias analysis (based on sex)]")
        print(f"Probability of being predicted >50K for female: {prob_female:.4%}")
        print(f"Probability of being predicted >50K for male: {prob_male:.4%}")
        print(f"Demographic parity difference (DPD): {dpd:.4f}")
        print(f"True Positive Rate difference (TPR diff): {tpr_diff:.4f}")
        print(f"False Positive Rate difference (FPR diff): {fpr_diff:.4f}")
        print(f"Equalized Odds difference (EOD): {eod:.4f}")
        print("-" * 30)

    print(f"Training time: {time.time() - start_train:.2f}s")


    if enable_wandb:
        wandb.finish()