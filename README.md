# Quantum Ising vs. Equilibrium Propagation for AML cell classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Two non-standard learning paradigms on the same medical-imaging task: a **quantum Ising-model classifier** and an **energy-based Equilibrium Propagation network**, both distinguishing healthy white blood cells from those altered by **Acute Myeloid Leukemia (AML)**. This is the exploratory companion to my first-author paper ([arXiv:2601.18710](https://arxiv.org/abs/2601.18710)) and its [main code repo](https://github.com/azrabano23/quantum-blood-cell-classification) — where that work benchmarks a variational quantum circuit and EP against CNN/dense baselines, this one takes the comparison in a different direction: a *quantum Ising* formulation instead of a VQC.

---

## The problem

AML is diagnosed in part from blood-smear morphology, and automating that screening is a real clinical-ML target — but it's also a useful testbed for a research question I care about: **how far can learning paradigms that aren't standard backprop-on-a-CNN get on a real medical task?** Quantum classifiers and energy-based models are both candidates for a post-backprop, hardware-efficient future (see the [equilibrium-propagation tutorial](https://github.com/azrabano23/equilibrium-propagation-tutorial) for why that matters), and AML cell images are a concrete, clinically grounded place to compare them head to head.

## The two approaches

**Quantum Ising classifier** (`src/quantum_networks/ising_classifier.py`). Images are reduced to 16×16 and encoded into a small variational circuit; the model is trained by minimizing a cost over circuit parameters — an Ising-style energy formulation rather than the ZZ-feature-map VQC of the main paper. It's a different quantum ansatz for the same decision, which is the point of having both repos.

**Equilibrium Propagation** (`equilibrium_propagation_classifier.py`). An energy-based network trained without backpropagation via two-phase relaxation (free phase + label-nudged phase, local contrastive weight update) — the same family as the paper's EP result, re-implemented here for the side-by-side.

## What this demonstrates (honestly)

The contribution is the **controlled comparison and the visualizations**, not a claim of quantum advantage — these are small models on a hard task, and the value is in making "how do these two paradigms behave on identical medical data?" legible. The repo generates quantum-circuit diagrams, an EP-vs-quantum comparison plot, and per-method analysis figures so the behavior is inspectable rather than asserted. Where the main paper reports the rigorous benchmark numbers, this repo is the sandbox that explores an alternative quantum formulation alongside EP.

## Technical breakdown

- **Quantum circuit modelling** — a parameterized Ising-style classifier with a trainable cost (variational optimization), plus circuit-visualization tooling.
- **Energy-based learning** — a from-scratch EP classifier (two-phase relaxation, local updates, no autograd).
- **A shared data pipeline** — microscopy-image loading, resizing/flattening, and a common train/test protocol so the two methods are compared on identical inputs.
- **Analysis & figures** — generated comparison plots and circuit diagrams (`*.png`) that make the paradigm differences visible.

**Skills demonstrated:** quantum variational modelling (PennyLane-style circuits), implementing a non-backprop energy-based learner, building a controlled two-method comparison on real medical images, and the scientific-communication work of turning that into inspectable figures.

## Quick start

```bash
pip install -r requirements.txt
python quantum_blood_cell_demo.py          # quantum Ising analysis
python equilibrium_propagation_classifier.py
python quantum_circuit_visualization.py    # circuit diagrams
```

## Related

- **Paper:** *Analyzing Images of Blood Cells with Quantum Machine Learning Methods* — [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)
- **Main benchmark repo:** [quantum-blood-cell-classification](https://github.com/azrabano23/quantum-blood-cell-classification)
- **EP from scratch:** [equilibrium-propagation-tutorial](https://github.com/azrabano23/equilibrium-propagation-tutorial)

## License

MIT — see [LICENSE](LICENSE). Author: **Azra Bano**.
