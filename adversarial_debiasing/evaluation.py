import torch
import torch.nn as nn
import pandas as pd
from dataset import get_train_test_split
from model import Predictor, load_model

X_train, X_test, y_train, y_test, A_train, A_test = get_train_test_split()

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
A_test = torch.tensor(A_test, dtype=torch.float32).unsqueeze(1)

input_dim = X_test.shape[1]
predictor = Predictor(input_dim=input_dim, hidden_dim=32)

load_model(predictor, "predictor.pt")
predictor.eval()

with torch.no_grad():
    y_hat, _ = predictor(X_test)
    y_pred = (y_hat >= 0.5).float()

accuracy = (y_pred == y_test).sum().item() / y_test.size(0)
print(f"Accuracy: {accuracy:.4f}")

y_pred_male = y_pred[A_test.squeeze() == 1]
y_pred_female = y_pred[A_test.squeeze() == 0]

p_male = y_pred_male.mean().item()
p_female = y_pred_female.mean().item()

demographic_parity_diff = abs(p_male - p_female)
print(f"Demographic Parity Difference: {demographic_parity_diff:.4f}")

y_test_male = y_test[A_test.squeeze() == 1]
y_test_female = y_test[A_test.squeeze() == 0]

tp_male = ((y_pred_male == 1) & (y_test_male == 1)).sum().item() / y_test_male.sum().item()
tp_female = ((y_pred_female == 1) & (y_test_female == 1)).sum().item() / y_test_female.sum().item()

equal_opportunity_diff = abs(tp_male - tp_female)
print(f"Equal Opportunity Difference: {equal_opportunity_diff:.4f}")

metrics = {
    "accuracy": [accuracy],
    "demographic_parity_diff": [demographic_parity_diff],
    "equal_opportunity_diff": [equal_opportunity_diff]
}

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("results/metrics.csv", index=False)
print("Metrics saved to results/metrics.csv")