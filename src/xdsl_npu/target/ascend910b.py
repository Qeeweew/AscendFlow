"""Typed dav-2201 actors and completion milestones.

These types classify a fact supplied by a trusted native rule. They do not
certify an arbitrary instruction: in particular an individual native end is
not a cumulative completion prefix. An implicit Cube prefix additionally
needs the particular native recipe's completion precondition.
"""

from enum import Enum


class Actor(str, Enum):
    AIC = "aic"
    AIV0 = "aiv0"
    AIV1 = "aiv1"


class Engine(str, Enum):
    S = "S"
    MTE1 = "MTE1"
    MTE2 = "MTE2"
    MTE3 = "MTE3"
    V = "V"
    M = "M"
    FIX = "FIX"


class Milestone(str, Enum):
    # ISSUE belongs to the actor's controller and has no pipeline engine.
    ISSUE = "issue"
    ADMISSION = "admission"
    RETIREMENT_CAPTURE = "retirement_capture"
    IMPLICIT_RETIREMENT = "implicit_retirement"
    NATIVE_END = "native_end"
    SET_EFFECT = "set_effect"

    @property
    def cumulative(self):
        return self in _CUMULATIVE


_CUMULATIVE = frozenset(
    {
        Milestone.ISSUE,
        Milestone.ADMISSION,
        Milestone.RETIREMENT_CAPTURE,
        Milestone.IMPLICIT_RETIREMENT,
    }
)
_ENGINES = {
    Actor.AIC: frozenset({Engine.S, Engine.MTE1, Engine.MTE2, Engine.M, Engine.FIX}),
    Actor.AIV0: frozenset({Engine.S, Engine.MTE2, Engine.MTE3, Engine.V}),
    Actor.AIV1: frozenset({Engine.S, Engine.MTE2, Engine.MTE3, Engine.V}),
}


def supports_engine(actor: Actor, engine: Engine) -> bool:
    """Physical pipeline ownership, without any implicit retirement claim."""
    return isinstance(actor, Actor) and isinstance(engine, Engine) and engine in _ENGINES[actor]
