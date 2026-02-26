import torch
import torch.nn as nn


class LogicLayer(nn.Module):
    def __init__(self, num_neurons, input_size):
        super().__init__()
        # each neuron has 16 weights for the 16 possible gates
        self.w = nn.Parameter(torch.randn(num_neurons, 16))

        # fixed random connections: each neuron picks 2 inputs
        # these stay fixed and are NOT parameters
        self.register_buffer('conn_indices', torch.randint(0, input_size, (num_neurons, 2)))

    def forward(self, x):
        # x shape: (batch_size, input_size)
        # 1. fetch the two inputs (a1, a2) for every neuron based on fixed wiring
        a1 = x[:, self.conn_indices[:, 0]]
        a2 = x[:, self.conn_indices[:, 1]]

        # 2. compute all 16 related logic operations
        # we compute these as tensors so it's fast

        ops = []
        ops.append(torch.zeros_like(a1))          # False
        ops.append(a1 * a2)                       # AND
        ops.append(a1 - a1 * a2)                  # A and not B
        ops.append(a1)                            # A
        ops.append(a2 - a1 * a2)                  # not A and B
        ops.append(a2)                            # B
        ops.append(a1 + a2 - 2 * a1 * a2)         # XOR
        ops.append(a1 + a2 - a1 * a2)             # OR
        ops.append(1 - (a1 + a2 - a1 * a2))       # NOR
        ops.append(1 - (a1 + a2 - 2 * a1 * a2))   # XNOR
        ops.append(1 - a2)                        # NOT B
        ops.append(1 - a2 + a1 * a2)              # B implies A
        ops.append(1 - a1)                        # NOT A
        ops.append(1 - a1 + a1 * a2)              # A implies B
        ops.append(1 - a1 * a2)                   # NAND
        ops.append(torch.ones_like(a1))           # True

        #stack to shape: (16, batch_size, num_neurons)
        all_ops = torch.stack(ops, dim=0) # dim tells the function which axis of the tensor to operate on

        # 3. Apply Softmax to weights to get gate probs
        probs = torch.softmax(self.w, dim=1) # (num_neurons, 16)

        # 4. weighted sum (equation 2)
        # reshape probs for broadcasting: (16, 1, num_neurons)
        probs = probs.t().unsqueeze(1) # t() does transpose on 2d tensor
        return torch.sum(all_ops * probs, dim=0)