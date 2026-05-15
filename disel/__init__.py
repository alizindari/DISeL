"""DISeL — per-rank input-dependent sigmoid gates on top of LoRA.

Public API:

* :class:`DiselConfig` — subclass of :class:`peft.LoraConfig` with gate fields.
* :func:`enable_disel` — attach gates to a model produced by ``get_peft_model``.
* :func:`build_optimizer` — AdamW with the separate-LR, no-WD gate group.
* :class:`RankGate`, :class:`LightRankGate` — the gate modules themselves,
  exposed for reuse outside PEFT.
* :class:`DiselLinearVariant` — PEFT ``LoraVariant`` (shaped like the in-tree
  ``DoraLinearVariant``, ready for upstreaming).
"""

from .config import DiselConfig
from .integration import GATE_PARAM_KEY, build_optimizer, enable_disel
from .layer import LightRankGate, RankGate
from .variant import DiselLinearVariant

__all__ = [
    "DiselConfig",
    "DiselLinearVariant",
    "GATE_PARAM_KEY",
    "LightRankGate",
    "RankGate",
    "build_optimizer",
    "enable_disel",
]

__version__ = "0.1.0"
