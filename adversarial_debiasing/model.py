import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(Predictor, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        hidden_repr = self.relu(self.hidden(x))
        y_hat = self.sigmoid(self.output(hidden_repr))
        return y_hat, hidden_repr
    
class Adversary(nn.Module):
    def __init__(self, hidden_dim=32, adv_hidden=16):
        super(Adversary, self).__init__()
        self.hidden = nn.Linear(hidden_dim, adv_hidden)
        self.relu = nn.ReLU()
        self.output = nn.Linear(adv_hidden, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_repr):
        adv_hidden = self.relu(self.hidden(hidden_repr))
        a_hat = self.sigmoid(self.output(adv_hidden))
        return a_hat
    
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()