Time Prediction for Embryo Development using Diffusion Models
=============================================================

Overview
--------
This repository contains the code and project structure for the MSc thesis:
"Time Prediction for Embryo Development using Diffusion Models".

Goal: Given a short **context window** of $k$ embryo frames ending at time $t$, predict the
next frame at $t+1$ (strictly adjacent) and roll out multiple steps by feeding generated frames
back into the context. The diffusion model is conditioned on the $k$ previous frames and an
optional **absolute frame index embedding**.

Guidance options:
- **Classifier-free guidance** (built-in during training/sampling).
- **External classifier guidance** at sampling time using your pretrained phase classifier
   (optional and safe to disable if the classifier cannot be loaded).

Core components:
- Automated preprocessing
- Frame presence filtering (keep only frames that contain an embryo)
- Phase classifier for guidance/analysis (16 morpho-kinetic phases)
- Conditional diffusion model for next frame prediction (x_t-k, ..., x_t → x_t+1)


Repository Layout
-----------------
```text
thesis/
├── data/
│   ├── embryo_dataset_bronze/                 # raw frames (per-embryo folders, original names)
│   ├── embryo_dataset_annotations/            # original *_phases.csv (kept as-is)
│   └── embryo_dataset_silver/
│       └── <plane_id>/                        # e.g., F0, F-15, F30, ...
│           └── <embryo_id>/
│               ├── 00001.jpeg, 00002.jpeg, ...
│               └── <embryo_id>_phases.csv (copied unchanged)
│
├── preprocessing/
│   ├── embryo_presence_models/                # saved presence models (.pth)   [ignored in git]
│   ├── train_presence_classifier.ipynb
│   └── preprocess_dataset.ipynb               # builds data/embryo_dataset_silver/F0
│
├── classifier/
│   ├── embryo_phase_models/                   # saved phase models (.pth)      [ignored in git]
│   └── train_phase_classifier.ipynb
│
├── diffusion/
│   ├── embryo_diffusion_models/               # saved diffusion weights (.pth) [ignored in git]
│   ├── diffusion_samples/                     # generated samples / predictions [ignored in git]
│   ├── cache/                                  # cached frame index + splits
│   └── train_diffusion.ipynb
│
└── old notebooks/
    ├── embryo_diffusion_old.pdf
    ├── embryo_phase_classifier_old.pdf
    └── embryo_presence_classifier_old.pdf
```


Data & Licensing
----------------
The project uses the public time-lapse human embryo dataset described in:
- Gomez et al., "A time-lapse embryo dataset for morphokinetic parameter prediction",
  Data in Brief 42 (2022) 108258, CC BY 4.0.

Dataset DOI: 10.5281/zenodo.6390798

IMPORTANT: To keep this repository small and license-compliant, only TWO example embryos
are included under `data/`:
- IDs: AA83-7 and AAL839-6
- Included in all relevant locations (raw bronze frames, matching `_phases.csv`, and preprocessed silver/F0)
- All remaining data are ignored via `.gitignore`.

If you have access to the full dataset, place it as:
- Raw frames (JPEG, 500x500): `data/embryo_dataset_bronze/<embryo_id>/*.jpg`
- Annotations CSV:            `data/embryo_dataset_annotations/<embryo_id>_phases.csv`

Preprocessing outputs will be written to:
- `data/embryo_dataset_silver/<plane_id>/<embryo_id>/` with filenames preserving original 1-based indices
   (e.g., 00001.jpeg, 00037.jpeg). Skipped frames remain missing (gaps kept).
- Phase CSVs are copied unchanged.

Please respect the dataset license and cite the authors when using the data in publications.


Setup
-----
1) Python environment (recommended)
   - Python >= 3.10 (tested on 3.10.18)
   - Create a virtual environment and install requirements.txt

2) Folder preparation
   - Place raw frames & annotations.
   - (Optional) Add `.gitkeep` files to empty model/samples folders to keep structure in git.

3) GPU
   - Training assumes a CUDA-capable GPU. Adjust batch sizes and num_workers for your hardware.


Run Order (Recommended)
-----------------------
1) Train/update presence classifier *(optional if you already have silver data)*
   - `preprocessing/train_presence_classifier.ipynb`
   - Saves to: `preprocessing/embryo_presence_models/embryo_presence_resnet18.pth`

2) Build preprocessed dataset (Silver)
   - `preprocessing/preprocess_dataset.ipynb`
   - Uses the presence model to drop non-embryo frames; preserves original indices.
   - Outputs under: `data/embryo_dataset_silver/<plane_id>/<embryo_id>/`

3) Train/update phase classifier *(optional, for external guidance)*
   - `classifier/train_phase_classifier.ipynb`
   - Saves to: `classifier/embryo_phase_models/`

4) Train the conditional diffusion model
   - `diffusion/train_diffusion.ipynb`
   - Consumes preprocessed frames and builds **context→target** pairs (strictly adjacent, $\Delta t = 1$).
   - Optional external classifier guidance at sampling time.
   - Saves weights to: `diffusion/embryo_diffusion_models/`
   - Samples / predictions: `diffusion/diffusion_samples/`


Conventions
-----------
- Filenames: Preprocessed frames keep original 1-based indices (e.g., 00001.jpeg).
- Gaps: Missing indices indicate filtered/absent frames; do not renumber.
- Data planes: Configure `FOCAL_PLANES` in the diffusion notebook (e.g., `['F0']`).
- Cache: Frame index + splits are cached under `diffusion/cache/`.
- Presence model path (default): `preprocessing/embryo_presence_models/embryo_presence_resnet18.pth`
- Phase classifier path (default): auto-detected in `classifier/embryo_phase_models/`
- Diffusion checkpoint (default): `diffusion/embryo_diffusion_models/nextframe_ctxk7_128px.pt`


Evaluation
----------------------
- Frame prediction fidelity: SSIM
- Perceptual similarity: LPIPS
- Distributional quality: FID on predicted vs. real future frames (matched Δt)
- Phase consistency: Compare phase transitions on generated sequences vs. annotations


Ethical Note
------------
Follow the dataset license and any local IRB/ethics guidelines when using clinical data. Generated results should not be used for clinical decision-making
without proper validation and regulatory approval.


Citing
------
If you use this repository or the included sample data, please cite:

Gomez T., Feyeux M., Boulant J., Normand N., David L., Paul-Gilloteaux P., Fréour T., Mouchère H.
"A time-lapse embryo dataset for morphokinetic parameter prediction."
Data in Brief 42 (2022) 108258. DOI: 10.5281/zenodo.6390798

And cite your own thesis as appropriate.


Acknowledgements
----------------
Thanks to the dataset authors and to the supervising team for guidance on the thesis:
- Supervisor: Elias Maglogiannis
- Committee: Theodoros Giannakopoulos, Michail Filippakis

