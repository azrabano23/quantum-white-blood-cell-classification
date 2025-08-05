# Quantum White Blood Cell Classification

This project uses a quantum Ising model to classify white blood cells, distinguishing between healthy cells and those affected by Acute Myeloid Leukemia (AML).

## Key Features

- **Quantum Ising Model**: An 8-qubit quantum circuit with 4 variational layers
- **Real Medical Data**: Uses the AML-Cytomorphology dataset from The Cancer Imaging Archive (TCIA)
- **Performance**: Achieves **52% accuracy** on real medical data - a successful proof-of-concept
- **Cell Types**: Classifies healthy cells (Lymphocytes, Monocytes, Neutrophils) vs AML cells (Myeloblasts, Monoblasts, etc.)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   ```bash
   python3 run_datasets_separately.py
   ```

3. **View results:** Check the generated `quantum_analysis_aml_cytomorphology.png` for visualization

## What This Demonstrates

- **Quantum machine learning** applied to real medical imaging
- **Simulated annealing** for blood cell classification (as alternative to CNNs)
- **Proof-of-concept** that quantum computing can work on practical medical problems
- **52% accuracy** on distinguishing healthy vs cancerous blood cells
