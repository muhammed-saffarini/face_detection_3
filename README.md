# Face Detection and Classification using DenseNet121 + HOG + ELM

This project implements a **face detection and classification system** that combines:
- **Deep Learning** (DenseNet121 for semantic features)
- **Hand-crafted features** (HOG for texture/edges)
- **Fast classifier** (Extreme Learning Machine, ELM)

The approach is designed for **speed and accuracy**, making it suitable for real-time or large-scale face classification.

---

## 🚀 Pipeline Overview
1. **Face Preprocessing**
   - Resize images → `224×224`
   - Normalize for DenseNet and HOG input
2. **Feature Extraction**
   - DenseNet121 (pretrained on ImageNet, no top layers) → deep semantic features
   - HOG (Histogram of Oriented Gradients) → edge/texture features
   - Concatenate both features into a single vector
3. **Classification**
   - Extreme Learning Machine (ELM), one hidden layer
   - Hyperparameters tuned using Grid Search
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, Log Loss
   - Training & testing time
5. **Result Logging**
   - Multiple runs (default = 5)
   - Best results saved to Excel

---

## 🔑 Key Features
- Hybrid feature extraction (Deep + Handcrafted)
- Very fast training with ELM
- Automated **Grid Search** optimization
- Detailed metrics and logs
- Scalable to large datasets

---

## 📂 Dataset Structure

Organize your dataset like this:

```
dataset/
├── final_dataset/       # Training data
│   ├── class1/
│   └── class2/
├── validation/          # (Optional) validation data
│   ├── class1/
│   └── class2/
└── Test/                # Testing data
    ├── class1/
    └── class2/
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install numpy opencv-python tensorflow scikit-image scikit-learn hpelm pandas
```

---

## ▶️ Usage

1. Prepare dataset in the above structure  
2. Run the script:

```bash
python HogGridSearchElmPhase3.py
```

3. Outputs:
   - Console: Best hyperparameters + metrics per run
   - Excel: `ModelELM/testing_metrics_ELM_5_runs_train_GridSearch.xlsx`

---

## 🔍 Grid Search Parameters

We tune the following parameters:

```python
param_grid = {
    'n_neurons': list(range(100, 1000, 50)),  # Hidden neurons
    'activation': ['sigm'],                   # Activation function
    'rp': [0.01, 0.1, 1, 10, 100, 1000]       # Regularization parameter
}
```

- `n_neurons`: 100 → 950 (step 50)  
- `activation`: `'sigm'` (sigmoid)  
- `rp`: `[0.01, 0.1, 1, 10, 100, 1000]`  

That gives **108 parameter combinations per run**.

---

## 📊 Workflow Pipeline

```
Input Image
      │
      ▼
Preprocessing (resize 224×224, normalize)
      │
 ┌────┴─────┐
 ▼          ▼
DenseNet121 HOG Features
Deep Features Texture Features
 └────┬─────┘
      ▼
Feature Concatenation
      ▼
Standardization (Z-score)
      ▼
Extreme Learning Machine (ELM)
   ├── Grid Search
   └── Best Model
      ▼
Evaluation (Acc, Precision, Recall, F1, Loss, Time)
      ▼
Results saved → Excel
```

---

## 📈 Example Results

| n_neurons | RP   | Activation | Train Acc | Test Acc | Precision | Recall | F1   | Loss | Train Time (s) | Test Time (s) |
|-----------|------|------------|-----------|----------|-----------|--------|------|------|----------------|---------------|
| 100       | 0.01 | sigm       | 0.923     | 0.915    | 0.918     | 0.910  | 0.914| 0.23 | 1.24           | 0.03          |
| 300       | 0.1  | sigm       | 0.940     | 0.931    | 0.934     | 0.928  | 0.931| 0.19 | 1.56           | 0.04          |
| 500       | 1    | sigm       | 0.951     | 0.947    | 0.950     | 0.945  | 0.947| 0.14 | 2.02           | 0.05          |
| 750       | 10   | sigm       | 0.958     | 0.952    | 0.954     | 0.951  | 0.952| 0.11 | 2.34           | 0.06          |
| 950       | 100  | sigm       | 0.960     | 0.954    | 0.956     | 0.952  | 0.954| 0.10 | 2.89           | 0.07          |

> ⚠️ Values are **illustrative only** – actual results depend on your dataset.

---

## 🏆 Example Best Model
- `n_neurons`: 750  
- `activation`: sigm  
- `rp`: 10  
- Training Accuracy: 0.958  
- Testing Accuracy: 0.952  
- Precision: 0.954  
- Recall: 0.951  
- F1: 0.952  
- Loss: 0.11  
- Training Time: ~2.3s  
- Testing Time: ~0.06s  

---

## 🔮 Future Work
- Multi-class classification (multiple face identities)  
- Real-time detection (OpenCV HaarCascade / MTCNN)  
- Test more activation functions (`relu`, `tanh`)  
- Compare with other classifiers (SVM, Random Forest, CNN)  
- Cross-validation for more robust evaluation  

---

Purpose: *Fast and accurate face recognition using hybrid features (DenseNet + HOG) with ELM classifier optimized by Grid Search.*
