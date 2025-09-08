import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaDQNAgent(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(MetaDQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

# Example usage:
if __name__ == "__main__":
    model = MetaDQNAgent(input_dim=3, num_actions=6)
    sample_state = torch.tensor([[0, 10, 3]], dtype=torch.float32)  # batch of 1
    q_vals = model(sample_state)
    print("Q-values:", q_vals)
