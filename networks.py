import torch
import torch.nn as nn
import torch.nn.functional as F

# Input dimensionality breakdown:
#   178  = game state (vectorizer.py: 52+52+6+6+11+40+7+4)
#           Added: hand_strength, is_suited, connectedness,
#                  flush_possible, straight_poss, board_paired, is_monotone
#   169  = opponent range belief (range_encoder.py, 169 hand classes)
#     8  = tell features (tell_detector.py, 8-dim behavioral vector)
# ─────
#   355  = NEXUS_GTO_Net total input
GAME_STATE_DIM = 178
RANGE_DIM      = 169
TELL_DIM       = 8
INPUT_DIM      = GAME_STATE_DIM + RANGE_DIM + TELL_DIM  # 355
N_ACTIONS      = 7


class ResidualBlock(nn.Module):
    """Pre-activation ResNet block with layer norm for stable deep training."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2   = nn.Linear(dim, dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation pattern (He et al.) — more stable than post-activation
        out = self.fc1(F.leaky_relu(self.norm1(x)))
        out = self.drop(out)
        out = self.fc2(F.leaky_relu(self.norm2(out)))
        return out + x  # Skip connection


class NEXUS_GTO_Net(nn.Module):
    """
    NEXUS core network.

    Architecture: 348 → 512 → ResBlock×6 → {Advantage, Value, RangePred}

    Three output heads:
      advantage (7)  : Per-action regret estimates for CFR matching
      value (1)      : Hand value (EV in normalised chip units)
      range_pred (169): Predicts opponent's range (self-supervised auxiliary task)

    Learned scalars:
      log_alpha : controls RAG blend weight (α = sigmoid(log_alpha), init ≈ 0.9)
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 512,
                 num_blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Input projection with normalisation
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # Deep reasoning trunk — 6 residual blocks (vs 4 in old architecture)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Head 1: Advantage / regret estimates for CFR
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, N_ACTIONS),
        )

        # Head 2: State value (EV)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            # Note: NO tanh — output must be unbounded (chips are unbounded)
        )

        # Head 3: Opponent range prediction (auxiliary, self-supervised)
        # Predicts opponent hand class distribution given observed board + actions.
        self.range_pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, RANGE_DIM),
            nn.Softmax(dim=-1),   # Output is a probability distribution over 169 classes
        )

        # Learned RAG blend weight: α = sigmoid(log_alpha)
        # Initialised so α ≈ 0.9 (mostly trust network, 10% RAG prior).
        # Learned during training; backprop adjusts how much to weight retrieved priors.
        self.log_alpha = nn.Parameter(torch.tensor(2.2))  # sigmoid(2.2) ≈ 0.9

        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 348] input tensor

        Returns:
            advantage : [batch, 7]
            value     : [batch, 1]
            range_pred: [batch, 169]
        """
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)

        advantage  = self.advantage_head(feat)
        value      = self.value_head(feat)
        range_pred = self.range_pred_head(feat)

        return advantage, value, range_pred

    def get_alpha(self) -> float:
        """Returns current RAG blend weight α ∈ (0, 1)."""
        return float(torch.sigmoid(self.log_alpha).item())

    def get_strategy(self, x: torch.Tensor,
                     rag_prior: torch.Tensor | None = None,
                     legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Full inference path: network output blended with RAG prior, then masked.

        Args:
            x          : [1, 348] state tensor
            rag_prior  : [1, 7]  retrieved strategy prior (or None)
            legal_mask : [1, 7]  boolean mask of legal moves (or None)

        Returns:
            strategy: [7] probability distribution over actions
        """
        with torch.no_grad():
            adv, _, _ = self.forward(x)

        # Regret matching: clip negatives, normalise
        regrets = adv.squeeze(0)
        regrets = torch.clamp(regrets, min=0.0)

        # Blend with RAG prior if available
        if rag_prior is not None:
            alpha = torch.sigmoid(self.log_alpha)
            rag   = rag_prior.squeeze(0) if rag_prior.dim() > 1 else rag_prior
            regrets = alpha * regrets + (1.0 - alpha) * rag

        # Apply legal move mask
        if legal_mask is not None:
            mask = legal_mask.squeeze(0).float()
            regrets = regrets * mask

        total = regrets.sum()
        if total > 1e-8:
            return regrets / total
        else:
            # Uniform over legal moves
            if legal_mask is not None:
                mask = legal_mask.squeeze(0).float()
                return mask / mask.sum().clamp(min=1e-8)
            return torch.ones(N_ACTIONS) / N_ACTIONS

    def _init_weights(self) -> None:
        """Xavier uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Backward-compatibility alias — old code importing DeepCFR_Network still works.
DeepCFR_Network = NEXUS_GTO_Net
