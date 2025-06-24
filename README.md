# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent credit card transactions. The aim is to help financial institutions identify and prevent fraudulent activities, minimizing loss and enhancing security for cardholders.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud is a significant threat in the financial sector. By leveraging machine learning algorithms, this project attempts to classify transactions as fraudulent or legitimate. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and visualization of results.

## Dataset

- The dataset used in this project is typically derived from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **[Download the CSV dataset directly from Kaggle here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3).**
- It contains anonymized transaction details for European cardholders in September 2013.
- The dataset is highly imbalanced, with a small fraction of transactions labeled as fraud.

## Features

- Data cleaning and preprocessing (handling missing values, feature scaling)
- Exploratory data analysis and visualization
- Handling class imbalance (e.g., using SMOTE, undersampling)
- Training various machine learning models (Logistic Regression, Random Forest, XGBoost, etc.)
- Model evaluation using precision, recall, F1-score, ROC-AUC
- Saving and loading trained models for future use

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chandrakant1212/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the data:**
   - Download the dataset from the link above and place the `creditcard.csv` file in the appropriate directory (e.g., `data/creditcard.csv`).

2. **Run the main script:**
   ```bash
   python main.py
   ```

3. **Jupyter Notebook:**
   - Open and run the provided notebook for interactive exploration:
     ```bash
     jupyter notebook Credit_Card_Fraud_Detection.ipynb
     ```

4. **Model Inference:**
   - Use the saved model to predict new transactions by providing input features.

## Model Architecture

- Various models can be trained and compared, including:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - XGBoost
  - Neural Networks (optional)

- Feature engineering and selection may be performed to improve performance.

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Curve**
- **Confusion Matrix**

Special focus is given to precision and recall, as minimizing false negatives (missed frauds) is crucial.

## Results

- The results, including classification reports and confusion matrices, are available in the output directory or as notebook cells.
- ROC curves and other relevant plots are generated for model comparison.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Disclaimer:** This project is for educational purposes only. Use real-world data responsibly and ensure compliance with all applicable laws and regulations.
