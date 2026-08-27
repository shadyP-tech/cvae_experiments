"""Exact-eight executable inputs and immutable SCEPTRE v2 anchors."""

from __future__ import annotations

from .identity import EXPERIMENT_ID


CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_sceptre_router/v2"
)
CANONICAL_SCRATCH_ROOT = "/data/local/fixed_bank_sceptre_router_v2"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
SOURCE_INNER_ALIAS_ARTIFACT_ID = (
    "midogpp_stage90_sceptre_source_inner_candidate_utility_reuse_v2"
)
SOURCE_INNER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_stage90_sceptre_source_inner_adaptive_reuse_amendment_v2"
)
TEST_CACHE_ARTIFACT_ID = "midogpp_stage90_sceptre_test_cache_v2"
TEST_MANIFEST_ARTIFACT_ID = "midogpp_stage90_sceptre_test_manifest_v2"
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_sceptre_parent_v2"
)
EXECUTION_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_sceptre_execution_amendment_v2"
)
SOURCE_INNER_ORIGINAL_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_source_inner_candidate_utility_v1"
)

SOURCE_INNER_AMENDMENT_FILENAME = "source_inner_adaptive_reuse_amendment_v2.json"
EXECUTION_AMENDMENT_FILENAME = "sceptre_v2_execution_amendment.json"

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    EXECUTION_AMENDMENT_ARTIFACT_ID,
)
AUTHORIZED_INPUT_ROLES = (
    "frozen_routing_authorized_expert_bank",
    "frozen_generation_lock",
    "consumer_fenced_v2_source_inner_seven_member_alias",
    "v2_source_inner_adaptive_development_amendment",
    "v2_label_free_consumed_test_cache_alias",
    "v2_role_scoped_consumed_test_manifest_alias",
    "exact_parent_test_consumption_ledger_alias",
    "v2_single_use_execution_amendment",
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_SOURCE_INNER_AMENDMENT_SHA256 = (
    "c00124f8c1016e99dac6b576055322c37917775eadb4f288ec1e8b68c5307443"
)

EXPECTED_EXECUTION_AMENDMENT_SHA256 = (
    "99dd6c0a945f1bfe17dd3a5b9444458e217327b70057235faac420446c6ee7e4"
)

EXPECTED_SOURCE_POLICY_LOCK_HASH = "6c18c72a017403a7"
EXPECTED_SOURCE_UTILITY_LOCK_SHA256 = (
    "cd50ea2420b332babc0788f2457e8e948410b8e88308d414d0c95db569ec6bac"
)
EXPECTED_SOURCE_UTILITY_TABLE_SHA256 = (
    "91d750e04a0eb8a7b44a9abe629fbcc1565a1206b20f530b8a9171c39bba5de3"
)
EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256 = (
    "78f638c361282b322a145af492d53e1690002ab8a49de3f67883a4f579e7a81f"
)
EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256 = (
    "76d887b3181af8f31a825c2b877189e89c4fe74b029cb82bbbfed23a0437b0bc"
)
EXPECTED_SOURCE_PREDICTION_INDEX_SHA256 = (
    "bfa96c50b0fae4ffd58f3354d95f00c9eb95e550ed0a96d3c28e620f6a05f62f"
)
EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256 = (
    "55564f32ed34977b05b2f6448148f0653a6e4bdb34bf8fcd93272f2ca1729bed"
)
EXPECTED_SOURCE_EVALUATION_ROWS_SHA256 = (
    "400face5697a00a96593cbbc1031cbab03810895b07f174505d45396dedf61e7"
)
EXPECTED_SOURCE_UTILITY_ROWS = 648
EXPECTED_SOURCE_CASE_CONFUSION_ROWS = 3168
EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS = 81
EXPECTED_SOURCE_EVALUATION_ROW_COUNT = 2615

EXPECTED_TEST_ROWS = 9_928
EXPECTED_TEST_FEATURE_DIM = 3_840
EXPECTED_TEST_CASES = 218
EXPECTED_TEST_ROWS_BY_CENTER = {
    "0": 1532,
    "1": 866,
    "2": 3210,
    "3": 1278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}
EXPECTED_TEST_CASES_BY_CENTER = {
    "0": 23,
    "1": 20,
    "2": 24,
    "3": 39,
    "5": 23,
    "6": 23,
    "7": 21,
    "8": 22,
    "9": 23,
}

SOURCE_INNER_MEMBERS = (
    "manifests/utility_lock.json",
    "tables/candidate_utility.csv",
    "tables/case_confusions.csv",
    "arrays/candidate_predictions.npz",
    "manifests/prediction_index.json",
    "tables/classifier_fits.csv",
    "tables/evaluation_rows.csv",
)
SOURCE_INNER_MEMBER_SHA256 = {
    SOURCE_INNER_MEMBERS[0]: EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
    SOURCE_INNER_MEMBERS[1]: EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
    SOURCE_INNER_MEMBERS[2]: EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
    SOURCE_INNER_MEMBERS[3]: EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256,
    SOURCE_INNER_MEMBERS[4]: EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
    SOURCE_INNER_MEMBERS[5]: EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
    SOURCE_INNER_MEMBERS[6]: EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
}

FORBIDDEN_INPUT_FRAGMENTS = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v1",
    "uniform_b_v2_consumed_test_fixed_bank_sceptre_router/v1",
    "fixed_bank_sceptre_router_v1-scratch",
    "reports/run_state.json",
    ".authorization_lease",
    "/checkpoints/",
    "/scratch/",
)

SOURCE_INNER_AMENDMENT_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "sceptre_router_v2/source_inner_adaptive_reuse_amendment_v2.json"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
