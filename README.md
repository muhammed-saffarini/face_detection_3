Face Detection and Classification using DenseNet121 + HOG + ELM
================================================================

This project implements a face detection and classification system that 
combines deep learning (DenseNet121) and hand-crafted features (HOG) 
with a fast classifier (Extreme Learning Machine, ELM).

The pipeline is designed for speed and accuracy:
- DenseNet121 extracts high-level semantic features.
- HOG extracts local texture and edge-based features.
- Features are concatenated and classified using an ELM neural network,
  tuned with Grid Search.

This approach leverages the representational power of deep networks with
the efficiency of ELM, making it suitable for real-time or large-scale 
face classification tasks.

----------------------------------------------------------------
Key Features
----------------------------------------------------------------
- Face Detection & Preprocessing
  * Resizes input images to 224×224
  * Normalizes and prepares input for DenseNet and HOG
- Feature Extraction
  * DenseNet121 (ImageNet pretrained, no top layers) → Deep semantic features
  * HOG (Histogram of Oriented Gradients) → Texture/edge features
  * Features concatenated into a single vector
- Classifier
  * Extreme Learning Machine (ELM), one hidden layer
  * Training is very fast
- Hyperparameter Optimization
  * Grid Search over hidden neurons, activation function, and RP
- Evaluation
  * Accuracy, Precision, Recall, F1-score, Log Loss
  * Training & testing time
- Result Logging
  * Runs multiple experiments (5 runs)
  * Saves best model results into Excel


----------------------------------------------------------------
Requirements
----------------------------------------------------------------
Install dependencies:

    pip install numpy opencv-python tensorflow scikit-image scikit-learn hpelm pandas

----------------------------------------------------------------
Dataset Structure
----------------------------------------------------------------
Organize dataset as follows:

dataset/
├── final_dataset/       # Training data
│   ├── class1/
│   └── class2/
├── validation/          # Validation data (optional in current script)
│   ├── class1/
│   └── class2/
└── Test/                # Testing data
    ├── class1/
    └── class2/

----------------------------------------------------------------
Usage
----------------------------------------------------------------
1. Prepare dataset in the structure above.
2. Run the script:

    python main.py

3. Outputs:
   - Console: Best hyperparameters and metrics per run
   - Excel: ModelELM/testing_metrics_ELM_5_runs_train_GridSearch.xlsx

----------------------------------------------------------------
Grid Search Parameters
----------------------------------------------------------------
The following parameter grid is used:

    param_grid = {
        'n_neurons': list(range(100, 1000, 50)),  # Hidden neurons
        'activation': ['sigm'],                   # Activation function
        'rp': [0.01, 0.1, 1, 10, 100, 1000]       # Regularization parameter
    }

- n_neurons: 100 → 950 in steps of 50
- activation: 'sigm' (sigmoid)
- rp: [0.01, 0.1, 1, 10, 100, 1000]

This produces 108 parameter combinations per run.

----------------------------------------------------------------
Workflow Pipeline
----------------------------------------------------------------
Input Image
      │
      ▼
Face Preprocessing (resize 224×224, normalize)
      │
      ├──► DenseNet121 → Deep Features
      ├──► HOG → Texture Features
      ▼
Feature Concatenation
      │
      ▼
Standardization (Z-score)
      │
      ▼
Extreme Learning Machine (ELM)
   ├── Grid Search (neurons, rp, activation)
   └── Evaluate best model
      │
      ▼
Metrics (Accuracy, Precision, Recall, F1, Loss, Time)
      │
      ▼
Save Results → Excel

----------------------------------------------------------------
Example Results
----------------------------------------------------------------

Sample Grid Search Results (excerpt):

| n_neurons | RP   | Activation | Train Acc | Test Acc | Precision | Recall | F1   | Loss | Train Time (s) | Test Time (s) |
|-----------|------|------------|-----------|----------|-----------|--------|------|------|----------------|---------------|
| 100       | 0.01 | sigm       | 0.923     | 0.915    | 0.918     | 0.910  | 0.914| 0.23 | 1.24           | 0.03          |
| 300       | 0.1  | sigm       | 0.940     | 0.931    | 0.934     | 0.928  | 0.931| 0.19 | 1.56           | 0.04          |
| 500       | 1    | sigm       | 0.951     | 0.947    | 0.950     | 0.945  | 0.947| 0.14 | 2.02           | 0.05          |
| 750       | 10   | sigm       | 0.958     | 0.952    | 0.954     | 0.951  | 0.952| 0.11 | 2.34           | 0.06          |
| 950       | 100  | sigm       | 0.960     | 0.954    | 0.956     | 0.952  | 0.954| 0.10 | 2.89           | 0.07          |

(*Values above are illustrative – actual results depend on dataset*)

----------------------------------------------------------------
Best Model (Example)
----------------------------------------------------------------
- n_neurons: 750
- activation: sigm
- RP: 10
- Training Accuracy: 0.958
- Testing Accuracy: 0.952
- Testing Precision: 0.954
- Testing Recall: 0.951
- Testing F1: 0.952
- Testing Loss: 0.11
- Training Time: ~2.3s
- Testing Time: ~0.06s

----------------------------------------------------------------
Future Work
----------------------------------------------------------------
- Add multi-class classification for multiple face identities
- Integrate real-time face detection (OpenCV HaarCascade / MTCNN)
- Test other activation functions (relu, tanh)
- Compare with other classifiers (SVM, Random Forest, CNN)
- Use cross-validation for more robust evaluation

----------------------------------------------------------------
Authors
----------------------------------------------------------------

Purpose: Fast and accurate face recognition using hybrid features 
(DenseNet + HOG) and ELM classifier with Grid Search optimization.
