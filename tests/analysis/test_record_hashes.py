"""Structural equality survives hash reuse and immutable record replacement."""

from dataclasses import replace

from ascendflow.analysis.structured_schedule import Placement
from examples.vector.scale import scale
from tests.support.structured import selected


def test_hashed_notification_records_remain_structural_keys_after_replacement():
    notification = selected(scale.specialize(B=64)).notifications[0]
    placement = notification.sources[0]
    need = notification.requirements[0]
    for record in (notification, placement, need):
        table = {record: "original"}
        equal = replace(record)
        assert equal is not record and equal == record
        assert hash(equal) == hash(record)
        assert table[equal] == "original"
    # Replacement must not carry the original cached hash into a changed key.
    changed = replace(notification, period=notification.period + 1)
    table = {notification: "original", changed: "changed"}
    assert table[replace(notification)] == "original"
    assert table[replace(changed)] == "changed"
    changed_placement = replace(placement, placement=Placement.BEFORE)
    changed_need = replace(need, period=need.period + 1)
    for record, changed in ((placement, changed_placement), (need, changed_need)):
        assert {changed: "changed"}[replace(changed)] == "changed"
    # instance_of remains provenance, excluded from semantic equality/hash.
    assert {need: "same"}[replace(need, instance_of=need)] == "same"
