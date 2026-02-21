# Data layout (no real data committed)

Place your images under `data/` on this machine; nothing is tracked in git. Recommended structure:

```
data/
  uwf/                 # original UWF images (200°)
    images/            # .jpg/.png files
    labels.csv         # image_path,class
  uwf_cropped/         # synthetic 50° crops generated from uwf/
    images/
    labels.csv         # mirrors uwf labels with updated paths
  mock/                # tiny synthetic samples (created by scripts/make_mock_data.py)
```

## Label CSV schema
- `image_path`: relative path from repo root (e.g., `data/uwf/images/img001.png`).
- `class`: string class label (e.g., `choroidal_melanoma`, `retinoblastoma`, `healthy`).

## Notes
- Keep raw data out of version control.
- If your data already sits in a different layout, adjust paths in configs/base.yaml accordingly.
- Cropping script preserves labels but rewrites `image_path` to the cropped outputs.
