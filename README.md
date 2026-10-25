# Impact Analysis of Weight Space

Do SIRENs trained on visually similar images converge to nearby points in weight space? This project investigates how **initialization and training regime** shape where implicit neural representations (INRs) end up, and whether apparent distance between networks reflects real functional differences or just permutation symmetry.

> College research project. Written up as an IEEE-format paper (preprint, not peer-reviewed). [Paper/poster link here]

## Overview

A SIREN (sinusoidal representation network) overfits a single image by mapping pixel coordinates to values. Each trained SIREN is a point in weight space. This repo asks:

- Are SIRENs trained on similar images closer together than those trained on dissimilar ones?
- How much does the **training regime** change that answer?
- After removing permutation symmetry, are networks closer than they first appear?

## Training Regimes Compared

| Regime | Description |
|---|---|
| Independent random init | Each network starts from its own random weights |
| Shared init | All networks start from the same initial weights |
| Warm-start continuation | Each network continues from a previously trained one |
| Reptile meta-learning | Initialization learned via Reptile, then fine-tuned per image |

## Metrics

- **Weight-space distance** between trained networks
- **Functional distance** between the images the networks reproduce
- **Git Re-Basin-style permutation alignment** (`align.py`) to remove neuron-permutation symmetry before measuring distance

## Repository Structure

```
.
├── siren.py             # SIREN model definition
├── data.py              # Image loading and coordinate grids
├── train.py             # Training loops for each regime
├── align.py             # Permutation alignment between networks
├── metrics.py           # Weight-space and functional distance metrics
├── run_experiment.py    # Runs the full experiment
├── plot_results.py      # Generates result plots
├── sanity_check.py      # Sanity checks for alignment and metrics
├── data/                # Input images
└── results/             # Saved outputs
```

## Getting Started

```bash
git clone https://github.com/snehaberri/Impact-Analysis-on-Weight-Spaces.git
cd Impact-Analysis-on-Weight-Spaces
pip install torch numpy matplotlib scipy
```

Run the sanity check first, then the experiment, then plot:

```bash
python sanity_check.py
python run_experiment.py
python plot_results.py
```

## Results

![Results](plot_results.png)

*Add 2-3 lines here summarizing the main finding, e.g. which regime produced the closest networks and how much alignment changed the distances.*

![Sanity check](sanity_check.png)

## Tech Stack

Python, PyTorch, NumPy, Matplotlib

## References

- Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions* (SIREN), 2020
- Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*, 2022
- Nichol et al., *On First-Order Meta-Learning Algorithms* (Reptile), 2018

## Author

**Sneha Sinha**, B.Sc. Computer Science Honours & Mathematics, CHRIST University
[GitHub](https://github.com/snehaberri) · snehaxsinha05@gmail.com

## License

Add a license (e.g. MIT) if you want others to reuse the code.
