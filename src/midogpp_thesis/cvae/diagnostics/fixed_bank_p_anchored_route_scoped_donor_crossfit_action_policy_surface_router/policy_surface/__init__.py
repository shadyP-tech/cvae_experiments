"""P-DCAPS complete-prefix policy response surface."""

from .contracts import (
    PolicyAction,
    PolicyObservation,
    PolicySurfaceProvenance,
    PrefixCell,
    PrefixSurface,
)
from .calibration_plan import (
    PolicyCalibrationFamilies,
    build_optimized_policy_calibration_families,
)
from .descriptors import POLICY_FEATURE_NAMES, PolicyDescriptor, descriptor_for_metric
from .envelope import (
    PolicyEnvelope,
    PolicyOOFResidual,
    apply_policy_envelope,
    build_policy_envelope,
)
from .ridge import (
    PolicyCalibration,
    PolicyRidgeModel,
    equal_center_route_prefix_weights,
    fit_policy_calibration,
)
from .runtime import (
    NestedPolicyCalibration,
    attach_prefix_responses,
    build_prefix_surface,
    calibrate_and_select_prefix,
    calibrate_and_select_prefix_with,
    calibrate_prefix_surface,
    calibrate_prefix_surface_with,
    fit_nested_policy_calibrations,
    observations_from_surfaces,
    policy_action_from_selection,
    strip_prefix_responses,
)
from .selection import CalibratedPrefixCell, PolicySelection, select_policy_prefix

__all__ = (
    "CalibratedPrefixCell",
    "NestedPolicyCalibration",
    "POLICY_FEATURE_NAMES",
    "PolicyAction",
    "PolicyCalibration",
    "PolicyCalibrationFamilies",
    "PolicyDescriptor",
    "PolicyEnvelope",
    "PolicyOOFResidual",
    "PolicyObservation",
    "PolicyRidgeModel",
    "PolicySelection",
    "PolicySurfaceProvenance",
    "PrefixCell",
    "PrefixSurface",
    "apply_policy_envelope",
    "attach_prefix_responses",
    "build_policy_envelope",
    "build_optimized_policy_calibration_families",
    "build_prefix_surface",
    "calibrate_and_select_prefix",
    "calibrate_and_select_prefix_with",
    "calibrate_prefix_surface",
    "calibrate_prefix_surface_with",
    "descriptor_for_metric",
    "equal_center_route_prefix_weights",
    "fit_nested_policy_calibrations",
    "fit_policy_calibration",
    "observations_from_surfaces",
    "policy_action_from_selection",
    "select_policy_prefix",
    "strip_prefix_responses",
)
