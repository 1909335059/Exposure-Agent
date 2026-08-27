from __future__ import annotations

import math

from exposure_agent.agent import Policy
from exposure_agent.camera import compute_relative_ev
from exposure_agent.models import ExposureAction, ExposureMetadata

from tests.conftest import quality_report


def test_policy_applies_absolute_targets_and_clamps_values() -> None:
    policy = Policy()
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=0)
    target = ExposureAction(target_iso=20000, target_shutter_speed_s=60.0)

    updated = policy.apply_action(metadata, target)

    assert updated.iso == 12800
    assert updated.shutter_speed_s == 30.0
    assert updated.ev == compute_relative_ev(12800, 30.0)


def test_ev_is_derived_and_not_an_independent_action() -> None:
    policy = Policy()
    metadata = ExposureMetadata(iso=100, shutter_speed_s=1 / 60, ev=99)

    updated = policy.apply_action(
        metadata,
        ExposureAction(target_iso=200, target_shutter_speed_s=1 / 30),
    )

    assert math.isclose(updated.ev, compute_relative_ev(200, 1 / 30))


def test_policy_detects_unchanged_absolute_target() -> None:
    policy = Policy()
    metadata = ExposureMetadata(iso=400, shutter_speed_s=0.01, ev=0)
    target = ExposureAction.for_metadata(metadata)

    assert policy.is_small_action(target, metadata) is True
    assert policy.is_unchanged(metadata, policy.apply_action(metadata, target)) is True


def test_policy_lists_unmet_quality_criteria() -> None:
    report = quality_report(
        overall=0.4,
        acceptable=False,
        brightness=0.1,
        shadow=0.7,
    )

    issues = Policy.unmet_quality_criteria(report)

    assert "shadow_ratio_too_high" in issues
    assert "midtone_ratio_too_low" in issues
    assert "overall_quality_too_low" in issues


def test_policy_satisfaction_uses_objective_report() -> None:
    assert Policy.is_satisfactory(quality_report(acceptable=True)) is True
    assert Policy.is_satisfactory(quality_report(acceptable=False)) is False
