# Simulation-Driven Synthetic Data Generation & Model Selection

## Project Overview

This project demonstrates an **end-to-end workflow** that integrates **Modelling & Simulation** with **Machine Learning** for predictive analysis and principled model selection.

A **single-server queuing system (M/M/1)** is modeled using **Discrete Event Simulation (SimPy)** to generate a realistic synthetic dataset. Multiple regression models are trained on this data, and instead of selecting a model using a single metric, a **Multi-Criteria Decision Making (MCDM)** approach using **TOPSIS** is applied to rank the models objectively.

---

## Modelling & Synthetic Data Generation

### Queueing System
- **Model:** M/M/1 Queue  
- **Simulation Framework:** SimPy  

### Simulation Parameters
- **Arrival Rate (λ):** Randomized within a controlled range  
- **Service Rate (μ):** Randomized to maintain queue stability  
- **Simulation Horizon:** Fixed duration per run  

### Dataset
- **Samples:** 1000 independent simulation runs  
- **Features:**
  - `Arrival_Rate`
  - `Service_Rate`
- **Target Variable:**
  - `Avg_Wait_Time`

This approach captures **stochastic and non-linear behavior** that mirrors real-world service systems.

---

## Machine Learning Pipeline

### Preprocessing
- Train/Test Split: 80% / 20%  
- Feature Scaling: `StandardScaler`  

### Regression Models Evaluated

**Linear Models**
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Huber Regressor  

**Instance / Kernel-Based**
- K-Nearest Neighbors (KNN)  
- Support Vector Regression (SVR)  

**Tree & Ensemble Models**
- Decision Tree  
- Random Forest  
- Extra Trees  
- Gradient Boosting  
- AdaBoost  

All models are trained and evaluated under identical conditions.

---

## Evaluation Metrics

Each model is evaluated using multiple complementary metrics:
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- Explained Variance  
- Maximum Error  

This ensures a balanced assessment across **accuracy, robustness, and worst-case performance**.

---

## Model Selection using TOPSIS

To avoid biased model selection, **TOPSIS** is used to rank models based on multiple criteria.

### Key Aspects
- Treats model selection as a **decision-making problem**  
- Considers both **benefit** and **cost** metrics  
- Produces a normalized **TOPSIS score (0–1)**  

Higher scores indicate models closer to the ideal solution.

---

## Results & Analysis

### TOPSIS-Based Model Ranking

The table below summarizes the relative performance of the top models according to TOPSIS.

| Rank | Model | R² Score | RMSE | Max Error | TOPSIS Score |
|---:|------------------|---------|-------|-----------|---------------|
| 1 | Huber Regressor | 0.836 | 0.012 | 0.041 | 0.995 |
| 2 | Ridge Regression | 0.834 | 0.012 | 0.040 | 0.994 |
| 3 | Linear Regression | 0.833 | 0.012 | 0.040 | 0.994 |
| 4 | AdaBoost | 0.820 | 0.013 | 0.039 | 0.976 |
| 5 | KNN | 0.815 | 0.013 | 0.041 | 0.944 |

**Insight:**  
Robust linear models (Huber, Ridge) perform exceptionally well due to their stability against outliers in the simulated waiting-time distribution.

---

## Visualizations

### Model Ranking via TOPSIS

This horizontal bar chart visualizes the **TOPSIS scores** of all evaluated models, clearly indicating their relative ranking and separation.

![Model Ranking via TOPSIS](topsis_ranking.png)

---

### Prediction Performance of Best Model

This scatter plot compares **Actual Waiting Time (Simulation Output)** vs **Predicted Waiting Time** for the **best-ranked model**.

- The red dashed line represents the ideal fit (`y = x`)
- Close alignment to this line indicates strong predictive accuracy

![Prediction Performance](best_model_performance.png)

---

## How to Run

1. Open the Jupyter Notebook in **Google Colab** or **Jupyter Lab**
2. Install the required dependency:
   ```python
   !pip install simpy
   ```
3. Run all cells sequentially to:
   - generate synthetic data  
   - train regression models  
   - compute TOPSIS rankings  
   - produce visualizations  

---

## Notes

- Dataset is **fully synthetic**, generated via simulation  
- No external data sources are used  
- Results are reproducible due to controlled randomization  
- Suitable for academic evaluation, viva discussions, and portfolio use