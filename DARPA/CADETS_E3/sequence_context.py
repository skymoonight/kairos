"""Utilities for handling per-event sequence context tensors.

Stage 1 of the Kairos extension pipeline augments every ``TemporalData``
instance with hashed command/path token sequences so that a future
Transformer branch can share the exact same event ordering as the
graph/TGN branch.  Stage 2 requires the training and inference loops to
surface those tensors in a structured way without yet modifying the
existing graph model.

This module provides light-weight helpers to check for the presence of
the new tensors, wrap them in a dedicated dataclass, validate their
alignment, and (optionally) move them across devices.  Keeping the logic
here avoids scattering attribute checks throughout the training and
testing code and prepares a single integration point for the upcoming
sequence encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

REQUIRED_SEQUENCE_FIELDS = (
    "cmd_tokens",
    "cmd_mask",
    "path_tokens",
    "path_mask",
)


def sequence_context_available(obj) -> bool:
    """Return ``True`` if *obj* exposes all required sequence fields."""

    return all(hasattr(obj, field) for field in REQUIRED_SEQUENCE_FIELDS)


@dataclass(frozen=True)
class SequenceContextBatch:
    """Container for the per-event sequence context tensors.

    All tensors are expected to have ``batch_size`` as their first
    dimension.  ``event_index`` is optional and is used purely for
    debugging/traceability.
    """

    cmd_tokens: torch.Tensor
    cmd_mask: torch.Tensor
    path_tokens: torch.Tensor
    path_mask: torch.Tensor
    event_index: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "SequenceContextBatch":
        """Move all tensors to ``device`` and return a new instance."""

        return SequenceContextBatch(
            cmd_tokens=self.cmd_tokens.to(device=device),
            cmd_mask=self.cmd_mask.to(device=device),
            path_tokens=self.path_tokens.to(device=device),
            path_mask=self.path_mask.to(device=device),
            event_index=None if self.event_index is None else self.event_index.to(device=device),
        )

    def validate(self, expected_length: int) -> None:
        """Ensure every tensor has ``expected_length`` entries.

        The Stage 1 preprocessing enforces this alignment; the extra check
        guards against inadvertent regressions before the sequence branch
        starts consuming these tensors.
        """

        tensors: Iterable[torch.Tensor] = (
            self.cmd_tokens,
            self.cmd_mask,
            self.path_tokens,
            self.path_mask,
        )

        for tensor in tensors:
            if tensor.size(0) != expected_length:
                raise ValueError(
                    "Sequence context alignment failure: expected length "
                    f"{expected_length}, got {tensor.size(0)}"
                )

        if self.event_index is not None and self.event_index.size(0) != expected_length:
            raise ValueError(
                "Sequence context index misalignment: expected length "
                f"{expected_length}, got {self.event_index.size(0)}"
            )

    def has_observed_tokens(self) -> bool:
        """Return ``True`` if any context token/mask contains real data."""

        return bool(self.cmd_mask.any() or self.path_mask.any())

    def event_mask(self) -> torch.Tensor:
        """Return a boolean mask indicating events with observed tokens."""

        return self.cmd_mask.any(dim=1) | self.path_mask.any(dim=1)


def sequence_context_from_batch(batch) -> Optional[SequenceContextBatch]:
    """Extract :class:`SequenceContextBatch` from a ``TemporalData`` batch.

    ``None`` is returned if the batch does not expose the augmented
    context attributes (for example when older preprocessed graphs are
    loaded).
    """

    if not sequence_context_available(batch):
        return None

    return SequenceContextBatch(
        cmd_tokens=batch.cmd_tokens,
        cmd_mask=batch.cmd_mask,
        path_tokens=batch.path_tokens,
        path_mask=batch.path_mask,
        event_index=getattr(batch, "context_event_index", None),
    )

