# Stock_Market_Analysis

This project predicts whether a stock’s **closing price** will be higher than its **opening price** using various machine learning models. The dataset contains synthetic or simulated stock data from June 2025.

---

##  Dataset

- **File**: `stock_market_june2025.csv`
- **Size**: 1762 rows × 14 columns
- **Key Features**:
  - `Date`, `Ticker`, `Open Price`, `Close Price`, `High Price`, `Low Price`
  - `Volume Traded`, `Market Cap`, `PE Ratio`, `Dividend Yield`, `EPS`
  - `52 Week High`, `52 Week Low`, `Sector`

---

##  Objective

The goal is to classify:

```python
Target = 1 if Close Price > Open Price else 0
```

This forms a **binary classification task**:  
- `1` → stock gained value  
- `0` → stock lost or remained flat

---

##  Data Preprocessing

- Converted `Date` to datetime format
- Created a `Target` variable based on price movement
- One-hot encoded the `Sector` column
- Dropped columns like `Date`, `Ticker`, and `Close Price` to prevent leakage
- Standardized features using `StandardScaler`

---

##  Exploratory Data Analysis (EDA)

- Correlation heatmap shows:
  - High correlation among price-based features (Open, High, Low)
  - Mild influence from PE Ratio, EPS, Market Cap
  - Weak but interesting effects from Sectors

---

##  Models Used

| Model                | Tuning Method     |
|---------------------|------------------|
| Logistic Regression | GridSearchCV     |
| Decision Tree       | None (baseline)  |
| Random Forest       | GridSearchCV     |
| SVM                 | (optional)       |

---

##  Evaluation Metrics

Each model is evaluated on:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ Classification Report (after tuning)

---

## 🔧 Hyperparameter Tuning

###  Logistic Regression
```python
param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
```

###  Random Forest
```python
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
```

- Both models were tuned using `GridSearchCV` (5-fold cross-validation) and the **F1-score** as the scoring metric.

---

##  Results

Final classification reports helped select the best model based on performance.Logistic Regression has higher accuracy in this instance.

You can compare models like so:
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
```

---

## How to Run

1. **Clone this repository**:
```bash
git clone https://github.com/aiyan-shamshad/stock_market_analysis.git
cd stock-market-prediction
```

2. **Install dependencies**:
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyter
```

3. **Run the notebook**:
```bash
jupyter notebook Stock_Market_Analysis.ipynb
```

---

## 👤 Author

GitHub: [@aiyan-shamshad](https://github.com/aiyan-shamshad)

---
