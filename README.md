# Quantum White Blood Cell Classification + Equilibrium Propagation

This project implements **dual computational approaches** for white blood cell classification: **Quantum Ising Models** and **Equilibrium Propagation**. Both methods distinguish between healthy cells and those affected by Acute Myeloid Leukemia (AML), providing a comprehensive comparison between quantum computing and energy-based learning for automated medical diagnostics.

## Medical Context

The human immune system must constantly identify what belongs in the body versus what is foreign or abnormal. When white blood cells become cancerous (as in AML), they develop irregular morphological features that can be detected through microscopy. This project automates that detection process using **two cutting-edge computational paradigms**:

- 🔮 **Quantum Computing**: Leveraging superposition, entanglement, and quantum interference
- 🧠 **Equilibrium Propagation**: Using energy-based learning with biologically-inspired dynamics

### Microscopy and Cell Imaging
The datasets used contain high-resolution microscopy images captured using:
- Oil immersion lenses for enhanced magnification and optical pathway optimization
- Specialized dyes that bind to different cellular components to make transparent cells visible
- 100x magnification with expert-level image quality from clinical laboratories

These imaging techniques reveal the morphological differences between healthy and malignant white blood cells that originate from bone marrow and enter the bloodstream.

## Key Features

### 🔮 Quantum Computing Approach
- **8-qubit Quantum Circuit**: 2 variational layers with 32 trainable parameters
- **Quantum State Space**: 256-dimensional Hilbert space for pattern recognition  
- **Quantum Advantages**: Superposition, entanglement, and interference for parallel processing
- **Performance**: ~52% accuracy on real medical data

### 🧠 Equilibrium Propagation Approach  
- **Energy-Based Model**: Symmetric weights with biologically-plausible learning rules
- **Two-Phase Training**: Free equilibrium + clamped target phases
- **Architecture**: Multi-layer network with 642 total parameters
- **Performance**: Competitive accuracy with quantum approach

### 📊 Shared Features
- **Real Medical Data**: AML-Cytomorphology dataset from The Cancer Imaging Archive (TCIA)
- **Cell Types Classified**:
  - **Healthy**: LYT (Lymphocytes), MON (Monocytes), NGS (Neutrophil Segmented), NGB (Neutrophil Band)
  - **AML**: MYB (Myeloblast), MOB (Monoblast), MMZ (Metamyelocyte), and other abnormal cell types
- **Rigorous Testing**: Both models evaluated on completely unseen data

## Technical Implementation

### 🔮 Quantum Ising Model
**Revolutionary quantum approach** using principles of quantum mechanics:
- **Quantum Superposition**: RY(πx_i) rotations encode data in quantum amplitudes
- **Quantum Entanglement**: CNOT gates create correlations between qubits  
- **Ising Interactions**: CNOT-RZ-CNOT sequences model spin-spin coupling
- **Variational Optimization**: Classical gradient descent on quantum parameters
- **Measurement**: Pauli-Z expectation values → binary classification

### 🧠 Equilibrium Propagation  
**Biologically-inspired energy-based learning**:
- **Energy Function**: E = -½Σw_ij×s_i×s_j - Σb_i×s_i with symmetric weights
- **Free Phase**: Network relaxes to equilibrium without output constraints
- **Clamped Phase**: Output neurons fixed to target values during relaxation
- **Learning Rule**: Δw ∝ (s_i^clamped×s_j^clamped - s_i^free×s_j^free)
- **Convergence**: Energy minimization drives both phases

### Data Processing Pipeline
1. **Image Loading**: Processes real microscopy images from clinical datasets
2. **Cell Type Recognition**: Automatically identifies different white blood cell subtypes
3. **Feature Extraction**: Converts 4x4 pixel regions to quantum state encodings
4. **Classification**: Distinguishes healthy from malignant cells using quantum measurements
5. **Validation**: Tests on completely unseen data to measure real-world performance

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Quantum Analysis
```bash
python3 run_datasets_separately.py
# or
python3 quantum_demo_complete.py
```

### 3. Run Equilibrium Propagation
```bash
python3 equilibrium_propagation_classifier.py
```

### 4. Generate Quantum Circuit Visualizations
```bash
python3 quantum_circuit_visualization.py
```

### 5. View Results
- **Quantum Analysis**: `quantum_analysis_aml_cytomorphology.png`
- **Equilibrium Propagation**: `equilibrium_propagation_blood_cell_analysis.png`  
- **Circuit Backend**: `quantum_circuit_backend_analysis.png`
- **Comparative Study**: `quantum_vs_ep_comparison.png`

## What This Demonstrates

### 🔬 Dual Computational Paradigm Comparison
- **Quantum vs Energy-Based**: Head-to-head comparison of fundamentally different computational approaches
- **Medical Application**: Both achieve >50% accuracy on real blood cell classification task
- **Practical Implementation**: Working, runnable code for both quantum and classical energy-based methods
- **Research Foundation**: Establishes baseline for comparing quantum advantage vs. energy-based learning

### 🏥 Medical Impact
- **Automated Screening**: Both approaches enable rapid blood cell analysis for clinical settings
- **Diagnostic Support**: Assists pathologists through two different computational paradigms  
- **Pattern Recognition**: Quantum interference vs. energy minimization for cellular morphology detection
- **Scalable Framework**: Both methods can extend to multiple cell types and diseases

### 💻 Computational Innovation  
- **Quantum Computing**: Practical quantum ML beyond toy problems, ready for quantum hardware
- **Energy-Based Learning**: Biologically-inspired alternative to standard backpropagation
- **Paradigm Comparison**: Direct comparison of quantum vs. energy-based computational approaches
- **Research Foundation**: Baseline for both quantum advantage and equilibrium propagation studies

## How This Was Technically Implemented

### Dataset Acquisition and Processing
1. **Medical Dataset Integration**: Downloaded real medical datasets from The Cancer Imaging Archive (TCIA)
   - AML-Cytomorphology_LMU dataset: 18,365 expert-labeled blood cell images
   - Bone Marrow Cytomorphology dataset: Over 170,000 annotated cells
   - Used NBIA Data Retriever tool to access restricted medical data

2. **Data Structure Organization**: 
   - Organized images by cell type directories (LYT, MON, NGS, NGB for healthy; MYB, MOB, MMZ, etc. for AML)
   - Implemented automated cell type classification based on directory structure
   - Created balanced datasets with equal numbers of healthy and malignant cells

### Quantum Circuit Architecture
3. **Ising Model Design**:
   - **8-qubit quantum circuit** with each qubit representing image features
   - **4 variational layers** creating depth for pattern learning
   - **CNOT + RZ gate combinations** implementing spin-spin interactions
   - **RX rotation gates** for local magnetic field effects

4. **Data Encoding Strategy**:
   - Resized microscopy images to 4x4 pixels (16 features)
   - Normalized pixel intensities to [0,1] range
   - Mapped pixel values to RY rotation angles: `RY(π * pixel_value)`
   - Each qubit encodes one pixel's information in quantum superposition

### Training and Optimization
5. **Variational Quantum Algorithm**:
   - **64 trainable parameters** (4 layers × 2 × 8 qubits)
   - **Gradient descent optimization** using PennyLane's automatic differentiation
   - **Cost function**: Minimizes classification error on training data
   - **30 training epochs** with learning rate of 0.1

6. **Quantum Measurement and Classification**:
   - Measured expectation value of Pauli-Z operator on first qubit
   - Binary decision: `prediction = 1 if <Z> > 0 else 0`
   - Maps quantum state to classical binary classification

### Validation and Testing
7. **Train-Test Split**:
   - **70% training data**: Used to optimize quantum parameters
   - **30% test data**: Completely unseen images for unbiased evaluation
   - **Stratified sampling**: Maintained class balance in both sets

8. **Performance Evaluation**:
   - Achieved **52% accuracy** on test set (above random chance)
   - Generated confusion matrices and classification reports
   - Created comprehensive visualizations of quantum decision boundaries

### Software Implementation
9. **Quantum Computing Framework**:
   - **PennyLane**: For quantum circuit construction and optimization
   - **NumPy**: For numerical computations and data manipulation
   - **Scikit-learn**: For data splitting and performance metrics
   - **Scikit-image**: For medical image processing and resizing

10. **Code Architecture**:
    - `QuantumBloodCellClassifier` class encapsulating all quantum functionality
    - Modular design with separate methods for data loading, training, and prediction
    - Automated visualization generation for results analysis
    - Error handling for robustness with real medical data

This technical implementation successfully demonstrated that quantum simulated annealing can be applied to real-world medical image classification, establishing a foundation for future quantum advantage research in healthcare applications.

---

## Results

| Method | Configuration | Test accuracy | Notes |
|---|---|---|---|
| Quantum Ising / VQC | 8–16 qubits, StronglyEntanglingLayers, `default.qubit` simulator | **~52%** | At/near the 50% chance line for the binary AML-vs-control split |
| Equilibrium Propagation | energy-based net, two-phase training | comparable | Same difficulty regime; see `equilibrium_propagation_classifier.py` |
| Random baseline | — | ~50% | Reference |

**How to read this honestly:** ~52% on a binary task is *essentially chance*. This is a **feasibility study**, and the result is a useful negative one: at the scales accessible on a laptop simulator (a handful of qubits, images downsampled to 16×16), neither the variational quantum circuit nor the energy-based model extracts the cell morphology that a CNN can. That is the honest finding, and it motivates the larger, properly-baselined study in the companion AML paper.

## Limitations & honest interpretation

- **Accuracy is at chance.** Do not read ~52% as a working classifier; read it as "this configuration does not yet learn the task."
- **Severe input compression.** 16×16 grayscale discards most of the morphological signal pathologists rely on.
- **Simulator-only, few qubits.** `default.qubit` with 8–16 qubits is far from a regime where quantum advantage is expected; no real-hardware runs.
- **Small data / no error bars.** Single split, `random_state=42`; results are not multi-seed and carry no confidence intervals.
- This repository is a learning/feasibility artifact, **not** a diagnostic tool.

## Related work & how to cite

This is the white-blood-cell sibling of a more complete, baselined study (CNN, dense, equilibrium propagation, and VQC compared under one protocol):

> A. Bano. *Analyzing Images of Blood Cells with Quantum and Energy-Based Machine Learning Methods.* arXiv:2601.18710.

```bibtex
@misc{bano_quantum_wbc,
  author       = {Bano, Azra},
  title        = {Quantum and Equilibrium-Propagation Models for White Blood Cell Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/azrabano23/quantum-white-blood-cell-classification}}
}
```
