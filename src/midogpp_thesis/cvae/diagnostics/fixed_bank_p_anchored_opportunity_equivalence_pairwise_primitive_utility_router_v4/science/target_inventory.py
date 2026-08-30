"""V4-owned immutable MIDOG++ consumed-test case inventory."""

from __future__ import annotations

from ..hashing import canonical_hash
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
)


_CASE_IDS_BY_CENTER = (
    ("0", "300 302 303 306 310 311 312 313 314 315 318 319 322 325 326 333 334 335 338 341 342 343 344"),
    ("1", "204 205 206 212 214 216 219 220 223 225 227 231 233 235 236 237 238 239 241 244"),
    ("2", "246 248 249 254 255 257 259 262 269 271 273 274 275 276 279 281 282 283 285 286 287 291 293 298"),
    ("3", "409 410 411 413 414 416 418 420 421 422 424 426 427 428 429 434 435 439 440 441 442 449 453 456 457 458 459 460 462 464 466 471 476 479 481 482 483 486 488"),
    ("5", "101 102 103 109 110 111 113 114 117 122 123 124 128 129 130 132 133 134 135 140 143 146 150"),
    ("6", "054 055 057 058 059 062 063 065 067 071 073 075 076 078 083 084 087 090 091 093 094 099 100"),
    ("7", "004 005 006 009 012 014 016 018 020 021 022 027 029 032 036 038 042 044 047 049 050"),
    ("8", "505 507 508 513 514 517 518 524 526 527 529 530 537 538 539 540 541 542 545 547 550 553"),
    ("9", "354 355 356 359 360 365 370 373 375 378 380 382 383 385 387 390 392 398 399 401 402 403 404"),
)

CANONICAL_TARGET_CASE_INVENTORY = tuple(
    (center, case_id)
    for center, case_ids in _CASE_IDS_BY_CENTER
    for case_id in case_ids.split()
)


def target_case_inventory_sha256(inventory: object) -> str:
    rows = tuple(inventory)  # type: ignore[arg-type]
    return canonical_hash({
        "schema_version": "oe_ppur_v1_terminal_case_manifest_v1",
        "dataset_family": "MIDOG++",
        "split": "test",
        "eligible_case_inventory": rows,
        "case_count": len(rows),
    })


if (
    len(CANONICAL_TARGET_CASE_INVENTORY) != EXPECTED_CASE_COUNT
    or len(set(CANONICAL_TARGET_CASE_INVENTORY)) != EXPECTED_CASE_COUNT
    or tuple(dict.fromkeys(center for center, _ in CANONICAL_TARGET_CASE_INVENTORY)) != CENTERS
    or target_case_inventory_sha256(CANONICAL_TARGET_CASE_INVENTORY) != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
):
    raise RuntimeError("OE-PPUR v4 canonical target case inventory constant drifted.")


__all__ = ("CANONICAL_TARGET_CASE_INVENTORY", "target_case_inventory_sha256")
