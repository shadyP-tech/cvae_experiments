"""Bind final enablement and admission to their advertised scientific hashes."""
from ...protocol import ProtocolError
from ...routing.correction_mass_router_v21.hashing import canonical_hash


def verify_crossfit_and_admission(crossfit, admission, *, config):
    """Reconstruct public evidence only; never fit or open a truth capability."""
    from ...routing.correction_mass_router_v21.admission import ApproximateSourceOOFBounds, SourceOnlyAdmission
    from ...routing.correction_mass_router_v21.contracts import AdmissionStatus
    try:
        compact = dict(crossfit)
        digest = compact.pop('result_hash')
        for field, key in (('records', 'score_hash'), ('frontier_rows', 'frontier_row_hash'),
                           ('candidate_prediction_outcome_joins', 'join_hash'),
                           ('winner_gate_diagnostics', 'diagnostic_hash')):
            compact[field] = [row[key] for row in crossfit[field]]
        if canonical_hash(compact) != digest:
            raise ProtocolError('HARP v21 final crossfit selection identity drifted.')
        raw_bounds = admission['bounds']
        bounds = None
        if raw_bounds is not None:
            observed, values = raw_bounds['observed_moments'], raw_bounds['bounds']
            bounds = ApproximateSourceOOFBounds(
                *(observed[k] for k in ('g', 'h', 'b', 'l')),
                *(values[k] for k in ('gain_lower', 'harm_upper', 'brier_upper', 'log_loss_upper')),
                raw_bounds['max_stat_critical_value'], tuple(raw_bounds['standard_errors']),
                raw_bounds['bootstrap_replicates'], raw_bounds['bootstrap_alpha'], raw_bounds['seed'],
                raw_bounds['missing_class_support_replicates'])
            if canonical_hash(bounds.public_payload()) != canonical_hash(raw_bounds):
                raise ProtocolError('HARP v21 source admission bound identity drifted.')
        restored = SourceOnlyAdmission(AdmissionStatus(admission['status']), admission['admitted'],
            admission['routed_case_count'], admission['routed_center_count'],
            tuple((row['center_id'], row['routed_case_count']) for row in admission['routed_cases_by_center']),
            admission['total_case_count'], admission['total_center_count'], bounds,
            admission['bootstrap_performed'], tuple(admission['routed_risk_moments'].items()),
            admission['qualifying_routed_center_count'])
        if canonical_hash(restored.public_payload()) != canonical_hash(admission):
            raise ProtocolError('HARP v21 source admission identity drifted.')
        if restored.admitted and (bounds is None or not bounds.passes
            or restored.routed_case_count < config.minimum_routed_oof_cases
            or restored.qualifying_routed_center_count < config.minimum_routed_oof_centers
            or sum(count >= config.minimum_routed_oof_cases_per_center for _, count in restored.routed_cases_by_center)
               < config.minimum_routed_oof_centers):
            raise ProtocolError('HARP v21 admitted policy lacks passing source risk/coverage evidence.')
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProtocolError('HARP v21 frozen crossfit/admission evidence is malformed.') from exc
