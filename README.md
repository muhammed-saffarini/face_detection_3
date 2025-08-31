# HDNetELM: A Hybrid Face Detection Framework for Highly Occluded Faces

## Description
This repository contains the implementation of HDNetELM, a hybrid face detection model designed for highly occluded faces. The pipeline integrates:

- **DenseNet121** for deep feature extraction.

- **Histogram of Oriented Gradients (HOG)** for capturing edge/texture information.

- **Extreme Learning Machine (ELM)** as a lightweight and efficient classifier.

- **Canny** edge detection for object proposals.

The model was evaluated on the Niqab dataset (faces with heavy coverings such as niqabs/veils) and a subset of the COCO dataset (non-face images), showing robust performance in detecting occluded faces.

Two training approaches are included:
1. **Manual Grid Search** (`file2.py`)
2. **Standalone DenseNet + ELM** (`face_elm.py`)

---

## Dataset Information
* **Niqab Dataset**: Custom dataset containing ~10,000 images with niqab- and veil-covered faces. DOI/Link: [Replace with Zenodo/Figshare DOI once uploaded]

* **COCO Dataset**: Publicly available Common Objects in Context dataset. URL: https://cocodataset.org

* **Preprocessing**:

-- Images resized to 224×224 pixels.

-- Contextual labeling used for occluded faces (bounding boxes include surrounding regions).

-- Balanced with non-face COCO subset to reduce false positives.

## 📂 Project Structure
```
├── file2.py        # Training with GridSearch hyperparameter tuning
├── face_elm.py     # Standalone DenseNet + ELM training (5-run evaluation)
├── ModelELM/       # Output folder for Excel results
├── dataset/
│   ├── final_dataset/   # Training dataset
│   ├── validation/      # Validation dataset
│   └── Test/            # Testing dataset
```

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install numpy pandas tensorflow scikit-learn scikit-image hpelm opencv-python openpyxl
```

---

## ▶️ Running the Code

### **1. Grid Search (file2.py)**

```bash
python file2.py
```

- Runs **manual Grid Search** over ELM hyperparameters.
- Evaluates best hyperparameters on training and test sets.
- Saves results to:
  ```
  ModelELM/testing_metrics_ELM_5_runs_train_GridSearch.xlsx
  ```

---

### **2. Standalone DenseNet + ELM (face_elm.py)**

```bash
python face_elm.py
```

- Extracts DenseNet121 features.
- Trains an ELM classifier with random hidden weights.
- Runs the experiment **5 times** to ensure reproducibility.
- Saves training and testing metrics into:
  ```
  ModelELM/testing_metrics_ELM_5_runs_train.xlsx
  ModelELM/testing_metrics_ELM_5_runs_test.xlsx
  ```

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
