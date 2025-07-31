# Quantum Blood Cell Classification - Complete Demo

This project demonstrates the application of quantum computing to medical image analysis of blood cells using a quantum Ising model. The implementation showcases how quantum neural networks can be applied to healthcare diagnostics, specifically for distinguishing between healthy blood cells and those from patients with Acute Myeloid Leukemia (AML).

## What This Project Does

### Quantum Computing Application in Medicine
This project bridges quantum physics and medical diagnostics by implementing a quantum Ising model for automated blood cell classification. The system uses quantum superposition and entanglement to process cellular features in parallel, potentially offering advantages over classical machine learning approaches.

### Technical Implementation
- **Quantum Ising Model**: Uses spin interactions to model complex relationships between cellular features
- **8-Qubit Architecture**: Creates a 2^8 = 256-dimensional quantum state space for pattern recognition
- **Variational Quantum Circuits**: Employs 32 trainable quantum parameters with gradient-based optimization
- **Medical Feature Encoding**: Maps nuclear morphology and cellular patterns to quantum states

### Key Quantum Concepts Demonstrated

1. **Quantum Superposition**: Enables parallel processing of all pixel features simultaneously through RY rotation gates
2. **Quantum Entanglement**: CNOT gates create correlations between qubits to model complex cellular relationships
3. **Ising Spin Model**: RZ rotations implement spin-spin coupling terms, naturally mapping cellular interactions to quantum physics
4. **Variational Optimization**: Adaptive learning of quantum parameters for pattern recognition
5. **Quantum Measurement**: Uses Pauli-Z expectation values for binary classification output

## New Implementation

- **Dataset**: Now includes real medical datasets from TCIA, integrating both AML-Cytomorphology_LMU and Bone Marrow Cytomorphology datasets
- **Architecture**: 8 qubits with 2 variational layers
- **Performance**: Successfully processed real data with visualizations generated for practical insights
- **Visualization**: Comprehensive analysis plots showing quantum circuit performance and medical relevance for real datasets

## Real-World Datasets

This project is designed to work with real medical datasets from The Cancer Imaging Archive (TCIA). Two key datasets are integrated:

### 1. AML-Cytomorphology_LMU Dataset
- **Description**: Single-cell morphological dataset of leukocytes from AML patients and non-malignant controls
- **Size**: 11GB containing 18,365 expert-labeled images
- **Source**: Munich University Hospital (2014-2017)
- **Resolution**: 100x optical magnification with oil immersion
- **Classes**: AML cells vs. healthy controls
- **Clinical Relevance**: Used for training CNNs in the original study published in Nature Machine Intelligence
- **DOI**: 10.7937/tcia.2019.36f5o9ld
- **License**: CC BY 3.0

### 2. Bone Marrow Cytomorphology Dataset
- **Description**: Expert-annotated bone marrow cytology from hematologic malignancies
- **Size**: 6.8GB containing over 170,000 cells
- **Source**: Munich Leukemia Laboratory, processed by Helmholtz Munich and Fraunhofer IIS
- **Resolution**: 40x magnification with oil immersion
- **Patients**: 945 patients with various hematological diseases
- **Cell Types**: Multiple categories including blasts, lymphocytes, neutrophils, etc.
- **DOI**: 10.7937/TCIA.AXH3-T579
- **License**: CC BY 4.0

## Dataset Access and Setup

### Download Instructions
1. Visit [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
2. Create a free account and agree to data use policies
3. Download and install the NBIA Data Retriever tool
4. Navigate to the dataset collections:
   - [AML-Cytomorphology_LMU](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
   - [Bone Marrow Cytomorphology](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
5. Use the "Download" button to generate .tcia files
6. Open with NBIA Data Retriever to download
7. Extract to appropriate directories in the project

### Automated Setup
```bash
# Create directory structure
python src/data_processing/download_data.py --setup

# View dataset information
python src/data_processing/download_data.py --info-only
```

## Running the Demo

### Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Quantum Demo on Real Datasets**:
   ```bash
   python run_datasets_separately.py
   ```

3. **View Results**: Check generated visualizations like `quantum_analysis_aml_cytomorphology.png` for insights on how the quantum model applies to real medical data

### Script Options

- **`run_datasets_separately.py`**: Analyzes each dataset individually and generates detailed performance reports
- **`quantum_demo_complete.py`**: Original demo script now updated to work with real datasets

### Cell Type Classification

The program automatically classifies different cell types:

**Healthy Cells:**
- LYT (Lymphocytes)
- MON (Monocytes)
- NGS (Neutrophil Segmented)
- NGB (Neutrophil Band)

**AML/Malignant Cells:**
- MYB (Myeloblast)
- MOB (Monoblast)
- MMZ (Metamyelocyte)
- KSC, BAS, EBO, EOS, LYA, MYO

### Using Real Datasets
The quantum classifier now works directly with real medical images from TCIA datasets, automatically processing and classifying different cell types based on medical knowledge.

## Scientific Impact and Applications

### Medical Diagnostics
- **Automated Screening**: Potential for rapid blood cell analysis in clinical settings
- **Pattern Recognition**: Quantum advantage in detecting subtle morphological differences
- **Diagnostic Support**: Assisting pathologists in identifying malignant cells

### Quantum Computing Research
- **Proof of Concept**: Demonstrates practical quantum machine learning in healthcare
- **Scalability**: Framework can extend to larger quantum devices as they become available
- **Interdisciplinary Bridge**: Connects quantum physics with medical AI

### Educational Value
- **Clear Implementation**: Well-documented code showing quantum concepts in action
- **Visualization**: Comprehensive plots explaining quantum circuit behavior
- **Reproducible Research**: Complete pipeline from data to results

## Future Development

- **Real Dataset Integration**: Full implementation with TCIA medical images
- **Multi-class Classification**: Extension beyond binary to multiple cell types
- **Quantum Hardware**: Testing on actual quantum devices (IBM Quantum, IonQ, etc.)
- **Clinical Validation**: Collaboration with medical professionals for real-world testing
- **Quantum Advantage Analysis**: Detailed comparison with classical machine learning baselines

## Citation Requirements

If using the TCIA datasets, please cite:

**AML Dataset:**
```
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). 
A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls [Data set]. 
The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.36f5o9ld
```

**Bone Marrow Dataset:**
```
Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021). 
An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies [Data set]. 
The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.AXH3-T579
```

## Repository Structure

```
quantum-blood-cell-classification/
├── quantum_demo_complete.py          # Main demonstration script (updated for real data)
├── run_datasets_separately.py       # Individual dataset analysis script
├── requirements.txt                  # Python dependencies
├── README.md                        # This documentation
├── PROJECT_SUMMARY.md              # Detailed technical summary
├── src/
│   ├── data_processing/
│   │   └── download_data.py         # TCIA dataset utilities
│   └── quantum_networks/
│       └── ising_classifier.py     # Quantum classifier implementation
├── data/                           # Dataset storage (with real medical data)
│   ├── aml_cytomorphology/         # AML cytomorphology dataset
│   ├── bone_marrow_cytomorphology/ # Bone marrow dataset
│   └── real_datasets/              # Additional real datasets
├── results/                        # Generated visualizations and outputs
└── *.png                          # Generated analysis visualizations
```

## License

This project is licensed under the MIT License. The integrated datasets from TCIA have their own licensing terms (CC BY 3.0 and CC BY 4.0) which must be followed when using the data.

## Acknowledgments

- The Cancer Imaging Archive (TCIA) for providing high-quality medical datasets
- Munich University Hospital and Munich Leukemia Laboratory for data collection
- Helmholtz Munich and Fraunhofer IIS for data processing infrastructure
- PennyLane and Qiskit communities for quantum computing frameworks
