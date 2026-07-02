# Multi Crop Analysis

This repository is being refocused around a training-first VAE pipeline.

## Environment

- Python: `3.10+`
- Required packages: see [requirements.txt](D:\multi_crop_analysis\requirements.txt)

## Current structure

```text
backend/
    app.py              # Minimal placeholder while training is in progress

src/
    config.py
    dataset.py
    vae.py
    train_vae.py

model/
    vae_encoder.pth     # Keep this; it is the current VAE checkpoint artifact
```

## Dataset Layout

The VAE training pipeline expects `data_split/` to be organized like this:

```text
data_split/
    <crop_name>/
        train/
            <class_name>/
                image files
        val/
            <class_name>/
                image files
        test/
            <class_name>/
                image files
```

## What is active now

- `src/train_vae.py` trains the VAE on the dataset in `data_split/`
- `src/dataset.py` loads `train`, `val`, and `test` folders for each crop/class
- `src/vae.py` defines the autoencoder used for representation learning
- `backend/app.py` is intentionally minimal until the final classifier is ready

## How to train the VAE

1. Prepare the dataset in `data_split/` with the expected split structure.
2. Install the required packages from `requirements.txt`.
3. Run:

```bash
python -m src.train_vae
```

4. The trained weights are saved to `model/vae_encoder.pth`

## Notes

- The old ResNet-based backend has been retired for now.
- `backend/` will be updated again after the classifier pipeline is trained.
- If `model/vae_encoder.pth` is only a Git LFS pointer, pull the actual file before running inference or continuing training.
