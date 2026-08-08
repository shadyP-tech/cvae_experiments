"""Action-neutral runtime contract for fresh Stage-70 evaluations."""

from __future__ import annotations

from ...real_features.classifier_reference.classifiers import ClassifierSpec


EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
MINIMUM_LOGICAL_CPU_COUNT = 12
MINIMUM_PHYSICAL_RAM_BYTES = 100 * 1024**3
MINIMUM_ARTIFACT_DISK_FREE_BYTES = 8 * 1024**3
MINIMUM_GPU_FREE_MIB_PER_DEVICE = 18000
OPTIONAL_LOCAL_SCRATCH_ROOT = "/data/local"
SOURCE_BLOCK_PER_CLASS = 256
CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3

DOWNSTREAM_CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "source_block_per_class": SOURCE_BLOCK_PER_CLASS,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "amp_enabled": False,
        "parent_cuda_context_forbidden": True,
        "gpu_and_cpu_phases_disjoint": True,
        "source_cache_format": "float32_npy_memmap",
        "prediction_cache_format": "float32_npy_memmap",
        "persistent_cache_policy": "hash_validated_resume",
        "optional_local_scratch_root": OPTIONAL_LOCAL_SCRATCH_ROOT,
        "canonical_publication_requires_validated_atomic_copy": True,
        "minimum_logical_cpu_count": MINIMUM_LOGICAL_CPU_COUNT,
        "minimum_physical_ram_bytes": MINIMUM_PHYSICAL_RAM_BYTES,
        "minimum_artifact_disk_free_bytes": MINIMUM_ARTIFACT_DISK_FREE_BYTES,
        "minimum_gpu_free_mib_per_device": MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    }


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "CLASSIFIER_WORKERS",
    "DOWNSTREAM_CLASSIFIER",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_GENERATION_LOCK_HASH",
    "MINIMUM_ARTIFACT_DISK_FREE_BYTES",
    "MINIMUM_GPU_FREE_MIB_PER_DEVICE",
    "MINIMUM_LOGICAL_CPU_COUNT",
    "MINIMUM_PHYSICAL_RAM_BYTES",
    "OPTIONAL_LOCAL_SCRATCH_ROOT",
    "SOURCE_BLOCK_PER_CLASS",
    "WORKSTATION_PROFILE",
    "canonical_runtime_payload",
)
