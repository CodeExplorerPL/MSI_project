from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from .logicLayer import LogicLayer
except ImportError:
    from logicLayer import LogicLayer

class TankLogicValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, num_layers=3, tau=10.0):
        """
        Aproksymator wartości dla agenta czołgu.
        
        :param input_dim: Rozmiar zbinaryzowanego wektora stanu gry.
        :param output_dim: 1 (dla State Value - ocena stanu) lub liczba akcji (dla Q-values).
        """
        super().__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tau = tau

        # Upewniamy się, że warstwa ukryta dzieli się równo przez liczbę wyjść
        assert hidden_size % output_dim == 0, "hidden_size musi być podzielne przez output_dim"
        self.neurons_per_output = self.hidden_size // self.output_size

        # Dynamiczne budowanie warstw (zamiast powtarzania kodu)
        layers = []
        layers.append(LogicLayer(self.hidden_size, self.input_size))
        for _ in range(self.num_layers - 1):
            layers.append(LogicLayer(self.hidden_size, self.hidden_size))
        
        self.logic_layers = nn.Sequential(*layers)

        # NOWOŚĆ: Warstwa skalująca wyjścia na pełne liczby rzeczywiste (w tym ujemne!)
        # Posiada tylko 8 wag i 8 biasów, więc nie obciąży obliczeń, a odblokuje potencjał sieci.
        self.q_value_scaler = nn.Linear(self.output_size, self.output_size)

    def forward(self, x, apply_scaler: bool = True):
        # Spłaszczenie wejścia (wielkość batcha, reszta)
        x = x.view(x.size(0), -1)
        
        # UWAGA: Środowisko musi podać tu stan już zbinaryzowany (np. grid z wartościami 0.0 i 1.0).
        # Jeśli podasz tu surowe wartości (np. pozycję X = 15.5), poniższa linia 
        # bezsensownie zamieni to na "1", tracąc informację o położeniu.
        x = (x > 0.5).float()

        # Przepływ przez bramki logiczne
        x = self.logic_layers(x)

        # Agregacja do konkretnej liczby wartości wyjściowych (akcji lub wartości stanu)
        x = x.view(x.size(0), self.output_size, self.neurons_per_output)
        values = torch.sum(x, dim=2) / self.tau
        
        # Przepuszczenie przez warstwę skalującą dla Q-learningu
        q_values = self.q_value_scaler(values)

        return q_values if apply_scaler else values


class MultiHeadTankDQN(nn.Module):
    """
    Shared encoder + independent Q-heads for multi-discrete action spaces.

    Head names are user-defined (e.g. move/hull/barrel/ammo).
    """

    def __init__(self, input_dim: int, head_dims: Dict[str, int], hidden_size: int = 768, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.head_dims = dict(head_dims)
        self.hidden_size = hidden_size

        layers = [LogicLayer(hidden_size, input_dim)]
        for _ in range(num_layers - 1):
            layers.append(LogicLayer(hidden_size, hidden_size))
        self.encoder = nn.Sequential(*layers)

        self.heads = nn.ModuleDict({
            head_name: nn.Linear(hidden_size, head_size)
            for head_name, head_size in self.head_dims.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.view(x.size(0), -1)
        x = (x > 0.5).float()
        z = self.encoder(x)
        return {head_name: head_layer(z) for head_name, head_layer in self.heads.items()}


class ANFISFireController(nn.Module):
    """
    Differentiable first-order Sugeno ANFIS for binary fire decisions.
    Input is expected in [0, 1] range with shape (batch, input_dim).
    """

    def __init__(self, input_dim: int = 4, num_rules: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        self.centers = nn.Parameter(torch.rand(num_rules, input_dim))
        self.log_sigmas = nn.Parameter(torch.zeros(num_rules, input_dim))
        self.consequents = nn.Parameter(torch.randn(num_rules, input_dim + 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_expanded = x.unsqueeze(1)
        centers = self.centers.unsqueeze(0)
        sigmas = torch.exp(self.log_sigmas).unsqueeze(0) + 1e-6

        membership = torch.exp(-0.5 * ((x_expanded - centers) / sigmas) ** 2)
        firing_strength = torch.prod(membership, dim=2)
        norm_strength = firing_strength / (torch.sum(firing_strength, dim=1, keepdim=True) + 1e-6)

        x_bias = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)
        consequents = torch.matmul(x_bias, self.consequents.t())
        output = torch.sum(norm_strength * consequents, dim=1, keepdim=True)

        return torch.sigmoid(output)


class TankMLPValueNetwork(nn.Module):
    """
    Simpler dense baseline for value approximation.
    Useful when logic gates collapse to repetitive actions.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...] = (512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = (x > 0.5).float()
        return self.net(x)
