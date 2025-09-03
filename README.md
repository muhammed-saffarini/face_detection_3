# HDNetELM: A Hybrid Face Detection Framework for Highly Occluded Faces

## Description
This repository contains the implementation of HDNetELM, a hybrid face detection model designed for highly occluded faces. The pipeline integrates:

- **DenseNet121** for deep feature extraction.

- **Histogram of Oriented Gradients (HOG)** for capturing edge/texture information.

- **Extreme Learning Machine (ELM)** as a lightweight and efficient classifier.

- **Canny** edge detection for object proposals.

The model was evaluated on the Niqab dataset (faces with heavy coverings such as niqabs/veils) and a subset of the COCO dataset (non-face images), showing robust performance in detecting occluded faces.


---

## 🔑 Key Features
- Hybrid feature extraction (Deep + Handcrafted)
- Very fast training with ELM
- Automated **Grid Search** optimization
- Detailed metrics and logs
- Scalable to large datasets

---

## Dataset Information
* **Niqab Dataset**: Custom dataset containing ~10,000 images with niqab- and veil-covered faces. DOI/Link: [https://doi.org/10.5281/zenodo.17011207]

* **COCO Dataset**: Publicly available Common Objects in Context dataset. URL: https://cocodataset.org

* **Preprocessing**:

-- Images resized to 224×224 pixels.

-- Contextual labeling used for occluded faces (bounding boxes include surrounding regions).

-- Balanced with non-face COCO subset to reduce false positives.

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

## ⚙️ Installation Requirements

Install required dependencies:

```bash
pip install numpy opencv-python openpyxl tensorflow scikit-image scikit-learn hpelm pandas
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

## 📊 Output

All scripts save results as Excel files in `ModelELM/`.  
Metrics include:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Loss  
- Training & Testing Time (seconds)  
- Confusion Matrix  

---

## 📌 Notes
- Update dataset paths if your dataset is stored elsewhere.
- Ensure `ModelELM/` directory exists, otherwise Excel saving will fail.

---

## ✨ Citation
If you use this work, please cite the corresponding paper or repository.

## 📜 License
This repository is released as open access:

- **Code**: Licensed under the [MIT License](LICENSE), allowing free use, modification, and redistribution with attribution.  
- **Dataset (Niqab Dataset)**: Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the dataset as long as appropriate credit is given.  

