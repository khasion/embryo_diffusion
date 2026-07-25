# Copilot instructions for this repo

## Big picture
- This repo is an MSc thesis project for **next‑frame prediction** in embryo time‑lapse data using a **context‑conditioned diffusion model**.
- Data flow: raw bronze frames + phase CSVs → presence filtering → **silver** dataset → diffusion training with **context→target** pairs (strictly adjacent $\Delta t=1$).
- Core notebooks (primary entry points):
  - preprocessing/preprocess_dataset.ipynb (builds silver dataset and preserves original 1‑based indices)
  - preprocessing/train_presence_classifier.ipynb (optional; produces presence model)
  - classifier/train_phase_classifier.ipynb (optional; external guidance only)
  - diffusion/train_diffusion.ipynb (main training + sampling)

## Project conventions (important)
- **Do not renumber frames**: preprocessed silver frames keep original 1‑based indices (e.g., 00001.jpeg). Missing numbers indicate filtered frames.
- Diffusion training pairs are **strictly adjacent** in time: context $[t-k+1..t]$ predicts target $t+1$.
- **Splits are by embryo ID**, not by frame; cached splits live under diffusion/cache/.
- Focal planes are configured via `FOCAL_PLANES` in the diffusion notebook; datasets are organized as data/embryo_dataset_silver/<plane>/<embryo>/.

## Model/training specifics to preserve
- The diffusion model is a **UNet2DModel** conditioned on a stack of $k$ previous frames, plus optional **absolute frame index embedding** (time map).
- Sampling uses **DDIM**; training uses **DDPM** schedules.
- Optional external phase classifier guidance is **sampling‑only** (not in training loss).
- Outputs:
  - checkpoints: diffusion/embryo_diffusion_models/
  - samples: diffusion/diffusion_samples/
  - caches: diffusion/cache/

## Workflow hints
- Start from the notebook map in diffusion/train_diffusion.ipynb and follow its section order.
- The full dataset (704 embryos, 7 focal planes) is present under data/; see README for layout.

## Files to reference for patterns
- README.md (project layout, run order, dataset rules)
- diffusion/train_diffusion.ipynb (config, dataset pairing logic, training loop, sampling grids)
- preprocessing/preprocess_dataset.ipynb (silver dataset creation + indexing rules)
- classifier/train_phase_classifier.ipynb (optional guidance model)
