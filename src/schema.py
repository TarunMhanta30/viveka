"""
VIVEKA — data schemas.

Defines every entity in the system. No logic here, only shape.
See Section 9 for the design reasoning behind each field.

Money is ALWAYS stored as integer paise, never float rupees.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------
# Enums — closed sets of allowed values
# ---------------------------------------------------------------

class SpendProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RhythmProfile(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    IRREGULAR = "irregular"


class AgentType(str, Enum):
    GROCERY = "grocery"
    TRAVEL = "travel"
    SUBSCRIPTION = "subscription"


class MerchantCategory(str, Enum):
    GROCERY = "grocery"
    FOOD = "food"
    TRAVEL = "travel"
    ELECTRONICS = "electronics"
    OTHER = "other"


class VolumeBand(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Channel(str, Enum):
    AGENTIC = "agentic"
    MANUAL = "manual"


class InstructionSource(str, Enum):
    """Where the agent's instruction came from.

    NOT a label. Benign transactions use EXTERNAL_CONTENT too —
    an agent legitimately reads a product page and buys.
    See Section 9.4.5.
    """
    USER_DIRECT = "user_direct"
    SCHEDULED = "scheduled"
    AGENT_AUTONOMOUS = "agent_autonomous"
    EXTERNAL_CONTENT = "external_content"


class MandateStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AttackRoute(str, Enum):
    A_INJECTION = "A"
    B_DELEGATION_ABUSE = "B"
    C_COUNTERFEIT_MERCHANT = "C"


# ---------------------------------------------------------------
# Entities
# ---------------------------------------------------------------

@dataclass
class Principal:
    """The human on whose behalf an agent acts."""
    principal_id: str
    created_at: datetime
    spend_profile: SpendProfile
    rhythm_profile: RhythmProfile
    active_hour_start: int
    active_hour_end: int


@dataclass
class Agent:
    agent_id: str
    principal_id: str
    agent_type: AgentType
    registered_at: datetime


@dataclass
class Mandate:
    """The consent artifact. Caps how much, but not how."""
    mandate_id: str
    principal_id: str
    agent_id: str
    merchant_scope: list[str]
    reserve_limit_paise: int
    created_at: datetime
    last_confirmed_at: datetime
    expires_at: datetime
    status: MandateStatus = MandateStatus.ACTIVE


@dataclass
class Merchant:
    merchant_id: str
    category: MerchantCategory
    registered_at: datetime
    reputation_score: float
    volume_band: VolumeBand


@dataclass
class TransactionEvent:
    """One agent-initiated payment attempt. The core record."""
    event_id: str
    timestamp: datetime
    principal_id: str
    agent_id: str
    mandate_id: str
    merchant_id: str
    amount_paise: int
    merchant_category: MerchantCategory
    item_count: int
    channel: Channel
    instruction_source: InstructionSource


@dataclass
class Label:
    """Ground truth. Stored SEPARATELY from events — see Section 9.4.6."""
    event_id: str
    is_attack: bool
    attack_route: Optional[AttackRoute] = None
    attack_variant: Optional[str] = None