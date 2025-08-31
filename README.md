# HDNetELM: A Hybrid Face Detection Framework for Highly Occluded Faces

This repository implements a **face recognition pipeline** that combines:
- **DenseNet121** for deep feature extraction
- **HOG (Histogram of Oriented Gradients)** for texture features
- **Extreme Learning Machine (ELM)** as the classifier  

Two training approaches are included:
1. **Manual Grid Search** (`file2.py`)
2. **Standalone DenseNet + ELM** (`face_elm.py`)

---

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
