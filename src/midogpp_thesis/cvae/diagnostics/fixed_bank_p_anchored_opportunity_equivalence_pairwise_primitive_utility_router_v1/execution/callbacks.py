"""Source- and code-sealed callback resolution for OE-PPUR spawn workers."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib
import inspect
import marshal
from pathlib import Path
import stat

from ..protocol import ProtocolError
from ..source_fence import assert_not_predecessor_reference
from .dtos import SealedCallbackDescriptorDTO, assert_pickle_safe_label_free_dto


_CURRENT_ADAPTER_PREFIX = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_"
    "router_v1"
)
_NEUTRAL_CORE_PREFIX = "midogpp_thesis.cvae.routing.pairwise_primitive_utility"


def seal_callback_descriptor(
    callback: object,
    *,
    callback_role: str,
    result_evidence_mode: str = "auto",
) -> SealedCallbackDescriptorDTO:
    """Seal one exact top-level source member without transporting the function."""

    function = _validate_top_level_function(callback)
    module_name = assert_not_predecessor_reference(
        function.__module__,
        role="worker callback module",
    )
    source_path = _callback_source_path(function)
    is_test_fixture = _is_repository_test_callback(
        module_name,
        source_path,
        function.__name__,
    )
    if not (
        module_name == _CURRENT_ADAPTER_PREFIX
        or module_name.startswith(f"{_CURRENT_ADAPTER_PREFIX}.")
        or module_name == _NEUTRAL_CORE_PREFIX
        or module_name.startswith(f"{_NEUTRAL_CORE_PREFIX}.")
        or is_test_fixture
    ):
        raise ProtocolError("OE-PPUR callback module is outside the sealed source scopes.")
    mode = str(result_evidence_mode)
    if mode == "auto":
        mode = (
            "strict_test_fixture"
            if is_test_fixture and function.__name__.startswith("_synthetic_")
            else "regular_file"
        )
    if mode == "strict_test_fixture" and not (
        is_test_fixture and function.__name__.startswith("_synthetic_")
    ):
        raise ProtocolError("OE-PPUR strict result fixtures are test-source only.")
    descriptor = SealedCallbackDescriptorDTO(
        callback_role=callback_role,
        module_name=module_name,
        member_name=function.__name__,
        source_path=source_path.as_posix(),
        source_sha256=_sha256_source(source_path),
        member_code_sha256=_member_code_sha256(function),
        result_evidence_mode=mode,
    )
    assert_pickle_safe_label_free_dto(descriptor)
    return descriptor


def resolve_sealed_callback(
    descriptor: SealedCallbackDescriptorDTO,
    *,
    expected_role: str,
) -> Callable[[object], object]:
    """Revalidate the descriptor inside a worker, then resolve its exact member."""

    if not isinstance(descriptor, SealedCallbackDescriptorDTO):
        raise ProtocolError("OE-PPUR worker callback descriptor is untyped.")
    assert_pickle_safe_label_free_dto(descriptor)
    if descriptor.callback_role != expected_role:
        raise ProtocolError("OE-PPUR callback descriptor role drifted.")
    try:
        module = importlib.import_module(descriptor.module_name)
    except (ImportError, ValueError) as exc:
        raise ProtocolError("OE-PPUR sealed callback module could not be imported.") from exc
    callback = getattr(module, descriptor.member_name, None)
    function = _validate_top_level_function(callback)
    source_path = _callback_source_path(function)
    if (
        source_path.as_posix() != descriptor.source_path
        or _sha256_source(source_path) != descriptor.source_sha256
        or _member_code_sha256(function) != descriptor.member_code_sha256
    ):
        raise ProtocolError("OE-PPUR sealed callback source or code drifted.")
    rebuilt = seal_callback_descriptor(
        function,
        callback_role=expected_role,
        result_evidence_mode=descriptor.result_evidence_mode,
    )
    if rebuilt != descriptor:
        raise ProtocolError("OE-PPUR callback descriptor did not revalidate exactly.")
    return function


def _validate_top_level_function(callback: object):
    if not inspect.isfunction(callback):
        raise ProtocolError("OE-PPUR worker callback is not a top-level function.")
    name = str(getattr(callback, "__name__", ""))
    qualified = str(getattr(callback, "__qualname__", ""))
    if (
        not name
        or qualified != name
        or name == "<lambda>"
        or callback.__closure__ is not None
        or callback.__defaults__ is not None
        or callback.__kwdefaults__ not in (None, {})
    ):
        raise ProtocolError("OE-PPUR worker callback is not a sealed top-level member.")
    signature = inspect.signature(callback)
    parameters = tuple(signature.parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or parameters[0].default is not inspect.Parameter.empty
    ):
        raise ProtocolError("OE-PPUR worker callback signature drifted.")
    return callback


def _callback_source_path(callback: object) -> Path:
    source = inspect.getsourcefile(callback)
    if source is None:
        raise ProtocolError("OE-PPUR callback has no inspectable Python source.")
    path = Path(source)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR callback source is absent.") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or path.suffix != ".py":
        raise ProtocolError("OE-PPUR callback source is not a regular Python member.")
    return path.resolve(strict=True)


def _sha256_source(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError("OE-PPUR callback source could not be hashed.") from exc
    return digest.hexdigest()


def _member_code_sha256(callback: object) -> str:
    try:
        payload = marshal.dumps(callback.__code__)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR callback code could not be sealed.") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_repository_test_callback(
    module_name: str,
    source_path: Path,
    member_name: str,
) -> bool:
    normalized = source_path.as_posix()
    return (
        (module_name.startswith("test_") or ".test_" in module_name)
        and "/tests/cvae/" in normalized
        and member_name.startswith("_")
    )


__all__ = (
    "resolve_sealed_callback",
    "seal_callback_descriptor",
)
