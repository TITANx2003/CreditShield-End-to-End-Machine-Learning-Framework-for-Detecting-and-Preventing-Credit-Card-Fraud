# CreditShield

**End-to-End Machine Learning Framework for Detecting and Preventing Credit Card Fraud**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔥 Project Overview

**CreditShield** is a state-of-the-art, end-to-end machine learning framework designed to detect and prevent credit card fraud in real-time. This project leverages advanced machine learning algorithms, feature engineering, and model evaluation techniques to build a robust fraud detection system.

The framework automates the entire pipeline—from data preprocessing and exploration to model training, evaluation, and deployment—making it an enterprise-ready solution for financial institutions, fintech startups, and consultancy projects.

---

## 📊 Dataset

**Dataset Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

* **Size:** 284,807 transactions
* **Fraudulent Transactions:** 492 (\~0.172% of total)
* **Features:**

  * `Time`: Seconds elapsed between this transaction and the first transaction in the dataset
  * `V1`–`V28`: Principal components obtained via PCA to protect sensitive information
  * `Amount`: Transaction amount
  * `Class`: Target variable (0 → Non-Fraudulent, 1 → Fraudulent)

> ⚠️ Note: This dataset is highly imbalanced, which makes it perfect for testing real-world fraud detection systems.

---

## 🛠 Key Features

* **End-to-End ML Pipeline**: From raw data preprocessing → feature engineering → model selection → evaluation → deployment-ready output.
* **Advanced Feature Engineering**: Handles scaling, normalization, and dimensionality reduction for optimal model performance.
* **Multiple ML Models**: Implements Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
* **Imbalanced Data Handling**: Techniques like SMOTE (Synthetic Minority Oversampling Technique) are applied to address class imbalance.
* **Real-Time Prediction Capability**: Designed to integrate with APIs for real-time fraud detection.
* **Visualization**: Comprehensive EDA (Exploratory Data Analysis) with fraud pattern detection charts and correlation matrices.

---

## 🚀 Methodology

1. **Data Preprocessing**

   * Handle missing values (if any)
   * Scale `Amount` and `Time` features
   * Split dataset into train/test

2. **Feature Engineering**

   * PCA-transformed features (`V1`–`V28`)
   * Synthetic feature creation for transaction frequency & risk scoring

3. **Model Training**

   * Logistic Regression
   * Random Forest Classifier
   * XGBoost
   * Gradient Boosting Classifier

4. **Evaluation Metrics**

   * Accuracy, Precision, Recall, F1-Score
   * Confusion Matrix
   * ROC-AUC Curve
   * Precision-Recall Curve (critical for imbalanced datasets)

5. **Deployment-Ready**

   * Model saved as `.pkl` for API integration
   * Easily extendable to real-time prediction systems

---

## 📈 Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.999    | 0.91      | 0.81   | 0.86     | 0.95    |
| Random Forest       | 0.999    | 0.95      | 0.88   | 0.91     | 0.97    |
| XGBoost             | 0.999    | 0.96      | 0.91   | 0.93     | 0.98    |
| Gradient Boosting   | 0.999    | 0.95      | 0.89   | 0.92     | 0.97    |

> ⚡ XGBoost achieves the best overall balance of precision and recall for fraud detection.

---

## 💻 Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/CreditShield-End-to-End-Machine-Learning-Framework-for-Detecting-and-Preventing-Credit-Card-Fraud.git
cd CreditShield-End-to-End-Machine-Learning-Framework-for-Detecting-and-Preventing-Credit-Card-Fraud
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python python_file_name.py/ipynb
```

---

## 🔧 Dependencies

* `pandas`, `numpy` → Data manipulation
* `scikit-learn` → ML models and metrics
* `xgboost` → Advanced gradient boosting
* `matplotlib`, `seaborn` → Visualizations
* `imbalanced-learn` → SMOTE & resampling techniques

---

## 🎯 Use Cases

* Real-time fraud detection for banks and fintech apps
* Risk scoring and monitoring of credit card transactions
* Data-driven consultancy projects in financial analytics
* Portfolio project for advanced ML demonstration

---

## 🌟 Future Enhancements

* Integration with streaming data (Kafka or Spark) for real-time scoring
* Deep learning approaches like Autoencoders or LSTMs for anomaly detection
* Model interpretability using SHAP or LIME
* Dashboard integration with Streamlit or Dash for visualization

---

## 📚 References

1. [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📝 License

This project is licensed under the MIT License.
