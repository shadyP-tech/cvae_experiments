"""Label-free opportunity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .shared import P_ACTION_ID, ProtocolError, _finite_tuple, _text, canonical_sha256

@dataclass(frozen=True, slots=True)
class ActionSurface:
    """A label-free candidate probability surface."""

    action_id: str
    family: str
    direction: str
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        action_id = _text(self.action_id, role="action id")
        if action_id == P_ACTION_ID:
            raise ProtocolError("P is an immutable anchor, not a challenger surface.")
        probabilities = _finite_tuple(self.probabilities, role="probability surface")
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ProtocolError("Probability surfaces must stay inside [0, 1].")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "family", _text(self.family, role="action family"))
        object.__setattr__(self, "direction", _text(self.direction, role="action direction"))
        object.__setattr__(self, "probabilities", probabilities)
@dataclass(frozen=True, slots=True)
class OpportunityMember:
    """Auditable membership of one action in a label-free equivalence class."""

    action_id: str
    family: str
    direction: str
    probability_hash: str
    crossing_hash: str
    structural_noop: bool
    exact_p_probability: bool
    representative_action_id: str | None

    def __post_init__(self) -> None:
        action_id = _text(self.action_id, role="opportunity action id")
        representative = (
            None
            if self.representative_action_id is None
            else _text(self.representative_action_id, role="representative action id")
        )
        if bool(self.structural_noop) != (representative is None):
            raise ProtocolError("Only structural no-ops may omit an opportunity representative.")
        if self.exact_p_probability and not self.structural_noop:
            raise ProtocolError("An exact-P probability surface must be a structural no-op.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "family", _text(self.family, role="opportunity family"))
        object.__setattr__(self, "direction", _text(self.direction, role="opportunity direction"))
        object.__setattr__(self, "probability_hash", _text(self.probability_hash, role="probability hash"))
        object.__setattr__(self, "crossing_hash", _text(self.crossing_hash, role="crossing hash"))
        object.__setattr__(self, "exact_p_probability", bool(self.exact_p_probability))
        object.__setattr__(self, "representative_action_id", representative)


@dataclass(frozen=True, slots=True)
class OpportunitySet:
    """Complete audit plus the unique active action representatives."""

    baseline_hash: str
    candidate_action_ids: tuple[str, ...]
    members: tuple[OpportunityMember, ...]
    active_representative_ids: tuple[str, ...]
    opportunity_hash: str

    def __post_init__(self) -> None:
        members = tuple(self.members)
        candidate_action_ids = tuple(
            sorted(_text(value, role="candidate action") for value in self.candidate_action_ids)
        )
        if (
            not members
            or len(set(candidate_action_ids)) != len(candidate_action_ids)
            or len({member.action_id for member in members}) != len(members)
        ):
            raise ProtocolError("Opportunity membership must be complete and action-unique.")
        if tuple(sorted(member.action_id for member in members)) != tuple(
            member.action_id for member in members
        ):
            raise ProtocolError("Opportunity membership must use canonical action order.")
        representatives = tuple(self.active_representative_ids)
        expected = tuple(
            sorted(
                {
                    member.representative_action_id
                    for member in members
                    if member.representative_action_id is not None
                }
            )
        )
        if representatives != expected:
            raise ProtocolError("Opportunity representatives drifted from membership.")
        if candidate_action_ids != tuple(member.action_id for member in members):
            raise ProtocolError("Opportunity audit omitted or added a frozen candidate action.")
        expected_hash = canonical_sha256({
            "schema": "pairwise_primitive_opportunity_set_v2", "baseline_hash": self.baseline_hash,
            "candidate_action_ids": candidate_action_ids,
            "members": tuple({"action_id": m.action_id, "family": m.family, "direction": m.direction,
                "probability_hash": m.probability_hash, "crossing_hash": m.crossing_hash,
                "structural_noop": m.structural_noop, "exact_p_probability": m.exact_p_probability,
                "representative_action_id": m.representative_action_id} for m in members),
            "active_representatives": representatives, "labels_used": False,
        })
        if self.opportunity_hash != expected_hash:
            raise ProtocolError("Opportunity hash drifted from its canonical payload.")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "candidate_action_ids", candidate_action_ids)
        object.__setattr__(self, "active_representative_ids", representatives)
        object.__setattr__(self, "baseline_hash", _text(self.baseline_hash, role="baseline hash"))
        object.__setattr__(self, "opportunity_hash", expected_hash)

    def member(self, action_id: object) -> OpportunityMember:
        key = str(action_id)
        for member in self.members:
            if member.action_id == key:
                return member
        raise ProtocolError(f"Unknown opportunity action: {key}")

    def equivalent_action_ids(self, representative_action_id: object) -> tuple[str, ...]:
        key = str(representative_action_id)
        if key not in self.active_representative_ids:
            raise ProtocolError(f"Unknown opportunity representative: {key}")
        return tuple(
            member.action_id
            for member in self.members
            if member.representative_action_id == key
        )


@dataclass(frozen=True, slots=True)
class OpportunityCaseReceipt:
    """Canonical label-free active/equivalence inventory for one case."""

    center_id: str
    case_id: str
    opportunity: OpportunitySet
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.opportunity, OpportunitySet):
            raise ProtocolError("Opportunity-case receipt requires a typed canonical opportunity set.")
        representatives = self.opportunity.active_representative_ids
        inventory = tuple((m.action_id, m.representative_action_id, m.structural_noop) for m in self.opportunity.members)
        active_from_inventory = tuple(
            sorted({representative for _, representative, noop in inventory if not noop and representative})
        )
        if (
            representatives != active_from_inventory
            or len({action for action, _, _ in inventory}) != len(inventory)
            or any(noop != (representative is None) for _, representative, noop in inventory)
        ):
            raise ProtocolError("Opportunity-case receipt inventory is inconsistent.")
        object.__setattr__(self, "center_id", _text(self.center_id, role="opportunity center"))
        object.__setattr__(self, "case_id", _text(self.case_id, role="opportunity case"))
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "opportunity_case_receipt_v2",
                    "center_id": self.center_id,
                    "case_id": self.case_id,
                    "opportunity_hash": self.opportunity.opportunity_hash,
                    "candidate_action_ids": self.opportunity.candidate_action_ids,
                    "active_representatives": representatives,
                    "equivalence_inventory": inventory,
                    "labels_used": False,
                }
            ),
        )

    @property
    def opportunity_hash(self) -> str:
        return self.opportunity.opportunity_hash

    @property
    def active_representative_ids(self) -> tuple[str, ...]:
        return self.opportunity.active_representative_ids

    @property
    def candidate_action_ids(self) -> tuple[str, ...]:
        return self.opportunity.candidate_action_ids

    @property
    def equivalence_inventory(self) -> tuple[tuple[str, str | None, bool], ...]:
        return tuple((m.action_id, m.representative_action_id, m.structural_noop) for m in self.opportunity.members)
