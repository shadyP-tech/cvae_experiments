# Paired Dense-All4 Reliability GIF Frames

This folder contains a compact conceptual animation of the paired dense-all4
reliability method. It intentionally avoids the full implementation pipeline and
keeps only the core thesis-facing flow.

## Frame Order

1. `frames/frame_01_setup.png` - four source experts, one target, inactive composition.
2. `frames/frame_02_dense_all4.png` - all four sources feed the dense all4 composition.
3. `frames/frame_03_soft_reliability.png` - heldout-excluded reliability changes influence but drops no source.
4. `frames/frame_04_geometric_pool.png` - generated source heads feed a geometric probability pool.
5. `frames/frame_05_clean_pass.png` - final BACC result and minimal PASS label.

## Generated Artifacts

- `contact_sheet.png` shows all frames.
- `paired_dense_all4_reliability_preview.gif` is the draft GIF preview.
- `scripts/generate_frames.py` regenerates all PNG frames, the contact sheet,
  and the preview GIF.

## Regenerate

```bash
/Users/stephpark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 presentations/paired_dense_all4_gif_frames/scripts/generate_frames.py
```

The preview GIF uses 1250 ms per frame. Edit `duration` in
`scripts/generate_frames.py` if a faster or slower GIF is needed.
