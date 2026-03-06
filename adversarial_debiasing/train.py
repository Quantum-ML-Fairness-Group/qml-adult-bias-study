import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_train_test_split
from model import Predictor, Adversary, save_model

X_train, X_test, y_train, y_test, A_train, A_test = get_train_test_split()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("A_train shape:", A_train.shape)

input_dim = X_train.shape[1]
hidden_dim = 32
adv_hidden = 16
lr = 0.001
epochs = 50
batch_size = 64
lambda_fair = 0.5

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
A_train = torch.tensor(A_train, dtype=torch.float32).unsqueeze(1)
A_test = torch.tensor(A_test, dtype=torch.float32).unsqueeze(1)

predictor = Predictor(input_dim=input_dim, hidden_dim=hidden_dim)
adversary = Adversary(hidden_dim=hidden_dim, adv_hidden=adv_hidden)

bce_loss = nn.BCELoss()

predictor_optimizer = optim.Adam(predictor.parameters(), lr=lr)
adversary_optimizer = optim.Adam(adversary.parameters(), lr=lr)

for epoch in range(epochs):
    predictor.train()
    adversary.train()
    
    y_hat, hidden = predictor(X_train)
    
    a_hat = adversary(hidden.detach())
    adv_loss = bce_loss(a_hat, A_train)
    
    adversary_optimizer.zero_grad()
    adv_loss.backward()
    adversary_optimizer.step()
    
    y_hat, hidden = predictor(X_train)
    a_hat = adversary(hidden)
    pred_loss = bce_loss(y_hat, y_train)
    
    total_loss = pred_loss - lambda_fair * bce_loss(a_hat, A_train)
    
    predictor_optimizer.zero_grad()
    total_loss.backward()
    predictor_optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, Adv Loss: {adv_loss.item():.4f}")

save_model(predictor, "predictor.pt")
save_model(adversary, "adversary.pt")