# Concrete-strength

# Concrete Strength Prediction

This project predicts the compressive strength of concrete based on its mix composition using machine learning.
It provides a complete workflow including data analysis, preprocessing, model training, evaluation, and deployment with a simple Streamlit web application.

---

## Overview

Concrete strength is a critical parameter that defines the durability and load-bearing capacity of structures. Traditionally, it is determined through time-consuming and costly lab experiments.
This project demonstrates how regression-based machine learning models can estimate concrete compressive strength efficiently and accurately.

---

## Features

* Uses the UCI Concrete Compressive Strength dataset.
* Includes full exploratory data analysis (EDA).
* Performs data preprocessing, feature scaling, and correlation analysis.
* Trains and evaluates multiple regression models:

  * Linear Regression
  * Ridge Regression
  * Random Forest Regressor
* Performs hyperparameter tuning with GridSearchCV.
* Displays feature importance and performance metrics.
* Provides a Streamlit-based web interface for real-time prediction.

---

## Dataset

**Source:** UCI Machine Learning Repository – Concrete Compressive Strength Dataset
[https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

| Feature                    | Description                |
| -------------------------- | -------------------------- |
| Cement (kg/m³)             | Cement content             |
| Blast Furnace Slag (kg/m³) | Slag addition              |
| Fly Ash (kg/m³)            | Fly ash content            |
| Water (kg/m³)              | Water content              |
| Superplasticizer (kg/m³)   | Plasticizer used           |
| Coarse Aggregate (kg/m³)   | Gravel content             |
| Fine Aggregate (kg/m³)     | Sand content               |
| Age (days)                 | Age of concrete            |
| Target                     | Compressive Strength (MPa) |

---

## Tools and Libraries

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

## Project Structure

```
ConcreteStrengthPrediction/
├── concrete_strength.ipynb        # Jupyter notebook for analysis and training
├── app.py                         # Streamlit application for predictions
├── Concrete_Data.xls              # Dataset
├── concrete_rf_pipeline.joblib    # Saved trained model
├── concrete_predictions_test.csv  # Predictions on test data
└── README.md                      # Project documentation
```

---

## Methodology

1. **Data Preprocessing**

   * Load and clean the dataset.
   * Normalize input features using StandardScaler.
   * Split into training and testing sets.

2. **Exploratory Data Analysis (EDA)**

   * Analyze feature distributions and correlations.
   * Visualize relationships using heatmaps and scatter plots.

3. **Model Development**

   * Train multiple models: Linear Regression, Ridge Regression, Random Forest.
   * Perform hyperparameter tuning using GridSearchCV.

4. **Evaluation**

   * Compute RMSE, MAE, and R² metrics.
   * Analyze residuals and feature importances.

5. **Model Saving and Deployment**

   * Save the best model using Joblib.
   * Deploy a prediction app using Streamlit.

---

## Model Evaluation

| Metric   | Description             | Example Result |
| -------- | ----------------------- | -------------- |
| RMSE     | Root Mean Squared Error | ~4.2           |
| MAE      | Mean Absolute Error     | ~3.5           |
| R² Score | Goodness of fit         | ~0.90          |

Actual values depend on the training run and parameters.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Dakksh-Git/Concrete-strength-prediction.git
cd Concrete-strength-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook (optional)

```bash
jupyter notebook concrete_strength.ipynb
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## Example Usage

Input mix details such as:

* Cement: 300
* Blast Furnace Slag: 50
* Fly Ash: 20
* Water: 180
* Superplasticizer: 10
* Coarse Aggregate: 1000
* Fine Aggregate: 800
* Age: 28

Output: Predicted Compressive Strength ≈ 38.7 MPa

---

## Future Improvements

* Include additional models like LightGBM or Neural Networks.
* Automate dataset preprocessing and feature selection.
* Deploy the Streamlit app to Streamlit Cloud or Render for public use.
* Integrate real-time data from IoT sensors in concrete curing.

---

## Author

**Dakksh Gupta**
GitHub: [https://github.com/Dakksh-Git](https://github.com/Dakksh-Git)

---

## License

You are free to use, modify, and distribute it with appropriate attribution.

---
