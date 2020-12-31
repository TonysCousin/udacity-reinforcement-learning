import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        ### John's code:
        fc2_size = 8*state_size
        fc3_size = 8*state_size
        
        self.fc1 = nn.Linear(state_size, fc2_size)
        self.fc2 = nn.Linear(fc2_size, fc3_size)
        self.fc3 = nn.Linear(fc3_size, action_size)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        ### John's code
        #x = self.dropout(state)
        x = state
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        
        #x = self.dropout(x)
        x = self.fc3(x) #no activation function
        #x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        
        return x
