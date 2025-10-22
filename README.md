# Stroke Disease Prediction Using Decision Tree Classification

## Project Overview
This project predicts the likelihood of stroke occurrence in individuals using a **Decision Tree Classifier**.  
Features include age, BMI, glucose level, hypertension, heart disease, smoking habits, marital status, work type, and residence type.

The dataset contains 5,110 records with 12 columns. The target variable `stroke` is highly imbalanced (~4.87% stroke cases).

---

## Dataset
- **Source**: `healthcare-dataset-stroke-data.csv` / Kaggle
- **Features**:
  - Numerical: `age`, `avg_glucose_level`, `bmi`
  - Categorical: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
  - Binary integer: `hypertension`, `heart_disease`
- **Target**: `stroke` (0 = No Stroke, 1 = Stroke)
- **Size**: 5,110 entries × 12 columns

---

## Data Preprocessing
- Filled missing `bmi` values with median.
- Dropped the `id` column.
- Replaced "Other" gender entry with "Female".
- Encoded categorical variables:
  - **Label Encoding**: `gender`, `ever_married`, `Residence_type`
  - **One-Hot Encoding**: `work_type`, `smoking_status`
- Addressed class imbalance using **SMOTE** on the training set.

---

## Model Training
- Dataset split:
  - Training: 3,270 samples  
  - Validation: 818 samples  
  - Test: 1,022 samples
- Baseline **Decision Tree Classifier**:
  - Validation Accuracy: ~91% (misleading due to imbalance)
  - Stroke recall: 0.17
- After SMOTE and hyperparameter tuning (`max_depth=6`):
  - Validation Accuracy: 0.8362
  - Test Accuracy: 0.8689
  - Stroke recall (test): 0.52 → improved minority detection

---

## Evaluation Metrics
- **Accuracy**: Overall correctness  
- **Confusion Matrix**: True vs. predicted  
- **Precision / Recall / F1-score**: Stroke detection performance  

**Test Set Example:**

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| No Stroke  | 0.97      | 0.89   | 0.93     | 972     |
| Stroke     | 0.19      | 0.52   | 0.28     | 50      |

---

## Feature Importance
Top predictors of stroke:
1. **Age** (most influential)
2. **Smoking status**
3. **Work type**
4. **Average glucose level**

Other features like BMI, gender, and residence type had minor contributions.

---

## Visualizations
- **BMI Distribution**: Moderate right skew  
- **Feature vs Stroke**: Boxplots and countplots  
- **Decision Tree Plot**: Top 3 levels visualized  
- **Feature Importance Bar Chart**: Highlights key predictors

---

## Libraries Used
```python
pandas, seaborn, matplotlib, scikit-learn, imbalanced-learn (SMOTE)
