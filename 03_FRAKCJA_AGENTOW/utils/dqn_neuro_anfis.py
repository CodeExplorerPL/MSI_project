from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


class ANFISHeadTorch(nn.Module):
    def __init__(self, input_dim: int, n_rules: int = 12):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_rules = int(n_rules)

        self.centers = nn.Parameter(torch.empty(self.n_rules, self.input_dim))
        self.log_sigmas = nn.Parameter(torch.empty(self.n_rules, self.input_dim))
        self.rule_w = nn.Parameter(torch.empty(self.n_rules, self.input_dim))
        self.rule_b = nn.Parameter(torch.zeros(self.n_rules))
        self.bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.centers, -1.0, 1.0)
        nn.init.uniform_(self.log_sigmas, -0.3, 0.3)
        nn.init.uniform_(self.rule_w, -0.15, 0.15)
        nn.init.uniform_(self.rule_b, -0.05, 0.05)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-5.0, 5.0)
        centers = torch.nan_to_num(self.centers, nan=0.0, posinf=3.0, neginf=-3.0)
        log_sigmas = torch.nan_to_num(self.log_sigmas, nan=0.0, posinf=4.0, neginf=-4.0)
        rule_w = torch.nan_to_num(self.rule_w, nan=0.0, posinf=3.0, neginf=-3.0)
        rule_b = torch.nan_to_num(self.rule_b, nan=0.0, posinf=3.0, neginf=-3.0)
        bias = torch.nan_to_num(self.bias, nan=0.0, posinf=3.0, neginf=-3.0)

        sigmas = F.softplus(log_sigmas).clamp(min=0.05, max=3.0)
        diff = (x.unsqueeze(1) - centers.unsqueeze(0)) / sigmas.unsqueeze(0)
        log_strength = (-0.5 * torch.sum(diff * diff, dim=-1)).clamp(-60.0, 10.0)
        phi = torch.softmax(log_strength, dim=1)
        phi = torch.nan_to_num(phi, nan=1.0 / self.n_rules, posinf=1.0, neginf=0.0)
        phi = phi / phi.sum(dim=1, keepdim=True).clamp(min=1e-6)

        rule_out = (torch.matmul(x, rule_w.t()) + rule_b.unsqueeze(0)).clamp(-30.0, 30.0)
        y = bias + torch.sum(phi * rule_out, dim=1)
        return torch.nan_to_num(y, nan=0.0, posinf=30.0, neginf=-30.0)


class TurretQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_rules: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList(
            [ANFISHeadTorch(input_dim=int(latent_dim), n_rules=int(n_rules)) for _ in range(int(n_actions))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        q = torch.stack([h(z) for h in self.action_heads], dim=1)
        return torch.nan_to_num(q, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)


@dataclass
class TurretActionDiag:
    q_selected: float
    q_mean: float
    q_std: float
    epsilon: float


class DQNTurretNeuroANFIS:
    def __init__(
        self,
        network: TurretQNetwork,
        action_bins: Sequence[float],
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.action_bins = [float(v) for v in action_bins]
        self.rng = random.Random(int(seed))
        self.state_dim = int(network.backbone[0].in_features)

    def select_action(self, features: Sequence[float], epsilon: float = 0.0) -> Tuple[int, float, Dict[str, float]]:
        eps = max(0.0, min(1.0, float(epsilon)))
        x = torch.tensor(list(features), dtype=torch.float32, device=self.device).view(1, -1)
        if x.shape[1] != self.state_dim:
            if x.shape[1] < self.state_dim:
                pad = torch.zeros((1, self.state_dim - x.shape[1]), dtype=torch.float32, device=self.device)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, : self.state_dim]

        with torch.no_grad():
            q_values = self.network(x)[0]

        if self.rng.random() < eps:
            idx = self.rng.randrange(len(self.action_bins))
        else:
            idx = int(torch.argmax(q_values).item())

        diag = TurretActionDiag(
            q_selected=float(q_values[idx].item()),
            q_mean=float(q_values.mean().item()),
            q_std=float(q_values.std(unbiased=False).item()),
            epsilon=eps,
        )
        return idx, float(self.action_bins[idx]), diag.__dict__

    @classmethod
    def load(cls, path: str, device: str = "auto") -> Tuple["DQNTurretNeuroANFIS", Dict[str, Any]]:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported turret checkpoint payload type: {type(payload)}")

        state_dim = int(payload.get("state_dim", 81))
        action_bins = payload.get("barrel_action_bins", [-1.0, 0.0, 1.0])
        if not isinstance(action_bins, list) or len(action_bins) <= 0:
            raise ValueError("Checkpoint is missing valid barrel_action_bins")

        meta = payload.get("meta", {})
        train_cfg = meta.get("train_config", {}) if isinstance(meta, dict) else {}
        hidden_dim = int(train_cfg.get("hidden_dim", 64))
        latent_dim = int(train_cfg.get("latent_dim", 32))
        n_rules = int(train_cfg.get("n_rules", 12))
        seed = int(train_cfg.get("seed", 42))

        net = TurretQNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_rules=n_rules,
            n_actions=len(action_bins),
        )
        online_state = payload.get("online_state", None)
        if not isinstance(online_state, dict):
            raise ValueError("Checkpoint is missing online_state")
        net.load_state_dict(online_state, strict=True)

        d = _resolve_device(device)
        agent = cls(network=net, action_bins=action_bins, device=d, seed=seed)
        return agent, payload

