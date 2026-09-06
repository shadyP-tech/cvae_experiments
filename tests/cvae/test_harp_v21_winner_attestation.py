"""Fresh validation authenticates the gate and the routed decision rule."""
from copy import deepcopy
import math

import pytest

from test_harp_v21_reconstruction_integration import real_fitted_surface
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.harp_v21_execution.support_validation import _verified_model_manifest
from midogpp_thesis.cvae.runtime.harp_v21_execution.winner_evidence import validate_winner_evidence
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.correction_mass_router_v21.winner_gate import WinnerGatePrediction


def test_fresh_validator_rejects_changed_gate_even_with_refreshed_artifact_hash(real_fitted_surface):
    manifest = deepcopy(dict(real_fitted_surface[2].manifest))
    gate = manifest['policy']['model']['winner_gate']
    # Public projections may carry immutable tuples before JSON serialization.
    changed = list(gate['coefficients'])
    changed[0] += .5
    gate['coefficients'] = changed
    manifest.pop('artifact_hash')
    manifest['artifact_hash'] = canonical_hash(manifest)
    with pytest.raises(ProtocolError, match='winner gate model hash'):
        _verified_model_manifest(manifest, centers=CENTERS)


def gate_model():
    return dict(model_hash='a'*64, feature_names=['predicted_gain'],
        participating_case_keys=[['center', 'case']], means=[0.], scales=[1.],
        coefficients=[math.log(.1/.9), 0.])


def evidence(signed=.02):
    transcript = WinnerGatePrediction('c'*64, 0., .1, .9, 'a'*64,
        ('predicted_gain',), (signed,)).public_payload()
    return dict(route_threshold=.75, route_score=.9, winner_gate_score=.9,
        winner_risk_adjusted_score=signed, winner_gate_harm_probability=.1,
        winner_gate_model_hash='a'*64, winner_gate_prediction_hash=transcript['prediction_hash'],
        winner_gate_prediction_payload=transcript,
        winner_composite_hash='c'*64, composite_hash='c'*64,
        winner_arm_id='D01_ONLY', selected_arm_id='D01_ONLY',
        admitted=True, fallback_reason=None, prediction_changed=True)


@pytest.mark.parametrize('field,value', [
    ('winner_risk_adjusted_score', -.01),
    ('winner_gate_harm_probability', .8),
    ('winner_gate_model_hash', 'd'*64),
    ('winner_arm_id', 'D10_ONLY'),
    ('route_threshold', .0),
    ('prediction_changed', False),
])
def test_fresh_validator_rejects_invalid_selected_winner(field, value):
    payload = evidence()
    payload[field] = value
    with pytest.raises(ProtocolError):
        validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=True)


def test_negative_winner_is_valid_evidence_for_exact_b_abstention():
    payload = evidence(-.01)
    payload.update(route_score=0., selected_arm_id='B',
                   composite_hash='d'*64, fallback_reason='NONPOSITIVE_PREDICTED_GAIN')
    validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=False)


def test_complete_gate_replay_accepts_honest_prediction():
    validate_winner_evidence(evidence(), gate_model=gate_model(), admitted=True, threshold=.75, routed=True)


def test_admitted_qualifying_winner_cannot_be_silently_replaced_by_baseline():
    payload = evidence()
    payload.update(selected_arm_id='B', composite_hash='d'*64, route_score=0.,
                   fallback_reason='WINNER_GATE_BELOW_THRESHOLD')
    with pytest.raises(ProtocolError, match='either direction'):
        validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=False)


def test_refreshed_prediction_hash_cannot_hide_changed_gate_probability():
    payload = evidence()
    transcript = WinnerGatePrediction('c'*64, 0., .6, .4, 'a'*64,
        ('predicted_gain',), (.02,)).public_payload()
    payload.update(winner_gate_prediction_payload=transcript,
        winner_gate_prediction_hash=transcript['prediction_hash'],
        winner_gate_harm_probability=.6, winner_gate_score=.4, selected_arm_id='B',
        composite_hash='d'*64, route_score=0., fallback_reason='WINNER_GATE_BELOW_THRESHOLD')
    with pytest.raises(ProtocolError, match='independent replay'):
        validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=False)


def test_winner_gate_feature_order_is_bound_to_authenticated_model():
    payload = evidence()
    transcript = WinnerGatePrediction('c'*64, 0., .1, .9, 'a'*64,
        ('wrong_feature',), (.02,)).public_payload()
    payload.update(winner_gate_prediction_payload=transcript,
                   winner_gate_prediction_hash=transcript['prediction_hash'])
    with pytest.raises(ProtocolError, match='independent replay'):
        validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=True)


def test_empty_calibrator_cannot_authorize_a_route_at_threshold_zero():
    transcript = WinnerGatePrediction('c'*64, 0., 1., 0., 'a'*64,
        (), (), False).public_payload()
    gate = dict(model_hash='a'*64, feature_names=[], participating_case_keys=[],
                means=[], scales=[], coefficients=[0.])
    payload = evidence()
    payload.update(route_threshold=0., route_score=0., winner_gate_score=0.,
        winner_gate_harm_probability=1., winner_gate_prediction_hash=transcript['prediction_hash'],
        winner_gate_prediction_payload=transcript, selected_arm_id='B', composite_hash='d'*64,
        fallback_reason='WINNER_GATE_UNAVAILABLE')
    validate_winner_evidence(payload, gate_model=gate, admitted=True, threshold=0., routed=False)
    with pytest.raises(ProtocolError, match='either direction'):
        validate_winner_evidence(payload, gate_model=gate, admitted=True, threshold=0., routed=True)


def test_calibration_availability_cannot_be_forged_with_a_refreshed_hash():
    payload = evidence()
    transcript = dict(payload['winner_gate_prediction_payload'])
    transcript.pop('prediction_hash')
    transcript['calibration_available'] = False
    transcript['prediction_hash'] = canonical_hash(transcript)
    payload.update(winner_gate_prediction_payload=transcript,
                   winner_gate_prediction_hash=transcript['prediction_hash'])
    with pytest.raises(ProtocolError, match='independent replay'):
        validate_winner_evidence(payload, gate_model=gate_model(), admitted=True, threshold=.75, routed=False)


@pytest.mark.parametrize('tamper', ['final_enabled', 'admission'])
def test_final_policy_flags_cannot_escape_their_scientific_hashes(real_fitted_surface, tamper):
    manifest = deepcopy(dict(real_fitted_surface[2].manifest))
    if tamper == 'final_enabled':
        manifest['policy']['crossfit']['final_policy_enabled'] = False
    else:
        manifest['policy']['admission']['admitted'] = False
    manifest.pop('artifact_hash')
    manifest['artifact_hash'] = canonical_hash(manifest)
    with pytest.raises(ProtocolError):
        _verified_model_manifest(manifest, centers=CENTERS)
