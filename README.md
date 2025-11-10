

# Ensemble Learning for Bike Share Demand Forecasting

This project focuses on predicting the hourly demand for a city's bike-sharing program using various ensemble machine learning techniques. The primary goal is to compare the performance of Bagging, Boosting, and Stacking against baseline models to identify the most accurate forecasting method, measured by the Root Mean Squared Error (RMSE).

***Author: Nishan (DA25M001)***

## Table of Contents
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Conclusion](#conclusion)
- [Setup and Usage](#setup-and-usage)

## Project Objective
The main objective is to build a regression model that accurately forecasts the hourly count of rented bikes. This involves:
1.  Preprocessing and cleaning the time-series data.
2.  Establishing baseline performance using simple models (Decision Tree and Linear Regression).
3.  Implementing and evaluating advanced ensemble techniques to reduce bias and variance:
    - **Bagging** (to reduce variance)
    - **Boosting** (to reduce bias)
    - **Stacking** (to combine model strengths for optimal performance)
4.  Comparing all models to determine the most effective approach for this forecasting problem.

## Dataset
The project uses the **Bike Sharing Demand** dataset, provided in the `hour.csv` file. This dataset contains 17,379 hourly records of bike rentals spanning two years.

**Key Features Used:**
- **Temporal:** `yr`, `mnth`, `hr`, `weekday`
- **Conditional:** `season`, `holiday`, `workingday`, `weathersit`
- **Environmental:** `temp`, `atemp`, `hum`, `windspeed`
- **Target Variable:** `cnt` (total hourly bike rentals)

## Methodology

The project is structured in four main parts:

### Part A: Data Preprocessing and Baseline Models
- **Data Loading and Cleaning:** The `hour.csv` dataset is loaded. Redundant or data-leaking columns (`instant`, `dteday`, `casual`, `registered`) are dropped.
- **Feature Engineering:** Categorical features (`season`, `mnth`, `hr`, etc.) are identified and prepared for one-hot encoding.
- **Train-Test Split:** The data is split into an 80% training set and a 20% testing set.
- **Baseline Models:** Two fundamental models are trained to establish a performance benchmark:
  - Decision Tree Regressor
  - Linear Regression

### Part B: Ensemble Techniques for Bias and Variance Reduction
- **Bagging (Variance Reduction):** A `BaggingRegressor` with a Decision Tree base estimator is implemented. This technique trains multiple trees on different bootstrap samples of the data and averages their predictions to create a more stable, lower-variance model.
- **Boosting (Bias Reduction):** A `GradientBoostingRegressor` is implemented. This technique builds models sequentially, with each new model trained to correct the errors of the previous ones, effectively reducing the overall model bias.

### Part C: Stacking for Optimal Performance
- **Stacked Generalization:** A `StackingRegressor` is constructed to leverage the strengths of multiple diverse models.
  - **Base Learners (Level 0):** K-Nearest Neighbors, Bagging Regressor, and Gradient Boosting Regressor.
  - **Meta-Learner (Level 1):** A `Ridge` regressor is trained on the predictions of the base models to produce the final, optimized forecast.

### Part D: Final Analysis
- All model results are compiled, compared, and visualized. A final conclusion is drawn based on the performance metrics.

## Results Summary

The performance of each model was evaluated using the **Root Mean Squared Error (RMSE)** on the test set. A lower RMSE indicates a more accurate model.

| Model                           | RMSE       |
| ------------------------------- | :--------- |
| Linear Regression (Baseline)    | 100.45     |
| Decision Tree (Baseline)        | 71.72      |
| Gradient Boosting Regressor     | 64.48      |
| Bagging Regressor               | 52.45      |
| **Stacking Regressor**          | **51.12**  |

### Performance Visualization


## Conclusion

The **Stacking Regressor** emerged as the best-performing model with the lowest **RMSE of 51.12**.

- **Why Stacking Excelled:** Stacking's superior performance is attributed to its ability to intelligently combine the predictions of diverse base models. While Bagging effectively reduced the high variance of the single Decision Tree and Boosting reduced the bias of weak learners, Stacking created an optimal blend. The meta-learner learned how to weigh the outputs from the bias-reducing (Boosting), variance-reducing (Bagging), and local-pattern-capturing (KNN) models to correct for their individual errors.

- **Final Recommendation:** For this bike share demand forecasting problem, the Stacking Regressor is the recommended model, as it provides the most accurate and robust predictions by leveraging the collective intelligence of multiple learning algorithms.
