import flax.linen as nn 
from typing import Sequence

class MLP(nn.Module):
    hidden_layers: Sequence[int]
    out_dims:int 

    @nn.compact
    def __call__(self, x):
        for layer in self.hidden_layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dims)(x) 
        return x