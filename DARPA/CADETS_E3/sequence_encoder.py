"""Sequence encoder and consensus fusion modules for the Kairos extension.

This module introduces a lightweight Transformer-style encoder that turns the
hashed command/path token sequences produced during preprocessing into per-event
embeddings.  A small classification head predicts edge types from the sequence
view alone, and a consensus fusion head blends the logits from the graph branch
with the sequence branch so the system can reason about structural and
contextual cues jointly.

The design intentionally keeps the number of attention layers and hidden
dimensions modest to respect the computational constraints highlighted during
planning.  The encoder falls back to zero vectors whenever an event lacks valid
context tokens so the existing graph-only pathway continues to operate without
penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from config import (
    context_max_seq_len,
    context_vocab_size,
    enable_sequence_branch,
    fusion_dropout,
    fusion_hidden_dim,
    seq_dropout,
    seq_embedding_dim,
    seq_feedforward_dim,
    seq_hidden_dim,
    seq_num_heads,
    seq_num_layers,
)


class TokenTransformerEncoder(nn.Module):
    """Encode a batch of token sequences with a light Transformer stack."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            context_vocab_size,
            seq_embedding_dim,
            padding_idx=0,
        )
        self.position = nn.Parameter(
            torch.zeros(1, context_max_seq_len, seq_embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_embedding_dim,
            nhead=seq_num_heads,
            dim_feedforward=seq_feedforward_dim,
            dropout=seq_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=seq_num_layers)
        self.layer_norm = nn.LayerNorm(seq_embedding_dim)
        self.dropout = nn.Dropout(seq_dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return pooled representations for each token sequence in ``tokens``."""

        if tokens.size(1) > context_max_seq_len:
            raise ValueError(
                "Token sequence exceeds configured maximum length; "
                "re-run preprocessing with a larger context_max_seq_len."
            )

        position = self.position[:, : tokens.size(1)].to(tokens.device)
        embedded = self.embedding(tokens) + position
        key_padding_mask = ~mask
        encoded = self.encoder(embedded, src_key_padding_mask=key_padding_mask)
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)

        mask_float = mask.unsqueeze(-1).float()
        pooled = (encoded * mask_float).sum(dim=1)
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        pooled = pooled / denom

        no_tokens = denom.squeeze(-1) == 0
        if no_tokens.any():
            pooled = pooled.masked_fill(no_tokens.unsqueeze(-1), 0.0)

        return pooled


class CommandPathSequenceEncoder(nn.Module):
    """Combine command and path encodings into a single representation."""

    def __init__(self) -> None:
        super().__init__()
        self.cmd_encoder = TokenTransformerEncoder()
        self.path_encoder = self.cmd_encoder  # share weights for stability

        combined_dim = seq_embedding_dim * 4
        self.combine = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(seq_dropout),
            nn.Linear(combined_dim, seq_hidden_dim),
            nn.GELU(),
            nn.Dropout(seq_dropout),
            nn.Linear(seq_hidden_dim, seq_hidden_dim),
        )

    def forward(
        self,
        cmd_tokens: torch.Tensor,
        cmd_mask: torch.Tensor,
        path_tokens: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode command/path tokens and return the fused representation."""

        cmd_feat = self.cmd_encoder(cmd_tokens, cmd_mask)
        path_feat = self.path_encoder(path_tokens, path_mask)
        combined = torch.cat(
            [cmd_feat, path_feat, torch.abs(cmd_feat - path_feat), cmd_feat * path_feat],
            dim=-1,
        )
        fused = self.combine(combined)
        return fused, cmd_feat, path_feat


class SequenceClassifier(nn.Module):
    """Predict edge logits from sequence representations alone."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(seq_hidden_dim),
            nn.Dropout(seq_dropout),
            nn.Linear(seq_hidden_dim, seq_hidden_dim),
            nn.GELU(),
            nn.Dropout(seq_dropout),
            nn.Linear(seq_hidden_dim, num_classes),
        )

    def forward(self, seq_repr: torch.Tensor) -> torch.Tensor:
        return self.net(seq_repr)


class ConsensusFusionHead(nn.Module):
    """Blend graph and sequence logits via a learnable per-class gate."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
            nn.Sigmoid(),
        )

    def forward(
        self,
        graph_logits: torch.Tensor,
        seq_logits: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return fused logits and the gating weights."""

        if seq_logits is None:
            return graph_logits, None

        gate_input = torch.cat([graph_logits, seq_logits], dim=-1)
        gate = self.gate(gate_input)

        if context_mask is not None:
            context_mask = context_mask.unsqueeze(-1).float()
            gate = gate * context_mask + (1.0 - context_mask)

        fused = gate * graph_logits + (1.0 - gate) * seq_logits
        return fused, gate


@dataclass
class SequenceBranchBundle:
    """Container to pass sequence branch modules around conveniently."""

    encoder: CommandPathSequenceEncoder
    classifier: SequenceClassifier
    fusion: ConsensusFusionHead

    def to(self, device: torch.device) -> "SequenceBranchBundle":
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)
        self.fusion = self.fusion.to(device)
        return self

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.classifier.parameters()
        yield from self.fusion.parameters()

    @property
    def enabled(self) -> bool:
        return enable_sequence_branch


__all__ = [
    "CommandPathSequenceEncoder",
    "ConsensusFusionHead",
    "SequenceBranchBundle",
    "SequenceClassifier",
]
