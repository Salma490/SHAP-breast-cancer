# Project 7: SHAP Explanations for Breast Cancer Classification

## 1. Problem Statement
Machine learning models are often "black boxes." In healthcare, a doctor cannot trust a model that predicts cancer without knowing *why*. 
This project trains an XGBoost classifier on the Breast Cancer dataset and uses SHAP (SHapley Additive exPlanations) to provide both global and local interpretability.

## 2. Data
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Rows:** 569
- **Features:** 30 numerical features (e.g., mean radius, worst perimeter)
- **Target:** Malignant (0) vs Benign (1)
- **Missing Values:** None

## 3. Methodology
1. **Exploratory Data Analysis:** Visualized class distribution and key feature separations.
2. **Data Preparation:** 80/20 train-test split.
3. **Modeling:** Trained an XGBoost Classifier.
4. **Explainability:** Used `shap.TreeExplainer` to calculate SHAP values.
   - *Global:* Bar plot and Beeswarm plot to show overall feature importance.
   - *Local:* Waterfall plot to explain a single patient's prediction.

## 4. Results & Evaluation
- **Model Accuracy:** 95.6%
- **Key Finding:** Built-in XGBoost feature importance heavily favors early-split features (like `worst radius`), while SHAP provides a more balanced and mathematically rigorous attribution. SHAP revealed that features like `area error` and `compactness error` are highly influential, which were completely ignored by the built-in importance.

## 5. Visuals
- **Global Feature Importance (SHAP)**
  <img width="804" height="940" alt="Global Feature Importance (Mean SHAP value)" src="https://github.com/user-attachments/assets/ac3dcda4-f5f6-4c30-828c-096a0d520788" />
- **Beeswarm Plot**
  ![Uploading SHAP Summary Plot (Feature Impact on Prediction).png…]()
- **Local Explanation (Waterfall)**
  ![Uploading Global Feature Importance (Mean SHAP value).png…]()


## 6. Limitations
- The dataset is relatively small and already clean (no missing values), which is rare in real-world clinical data.
- SHAP explains the *model*, not the *biology*. A high SHAP value for `worst perimeter` means the model relies on it, not necessarily that it is the sole medical cause of malignancy.

## 7. How to Run
```bash
pip install -r requirements.txt
jupyter notebook shap_breast_cancer.ipynb
