# Hotel Reservations Classification

## Project Overview
This project is part of a university assignment aimed at building a machine learning model to classify hotel reservations. The project encompasses several stages, including data preparation, data analysis, model comparison, final model training, and deployment to a production server.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Data Analysis](#data-analysis)
- [Model Comparison](#model-comparison)
- [Deployment](#deployment)
- [Conclusion](#conclusion)
- [Authors](#authors)

## Data Preparation
The data preparation phase involves cleaning and preprocessing the raw data to make it suitable for analysis and modeling. This includes handling missing values, encoding categorical variables, and normalizing numerical features.

## Data Analysis
In the data analysis phase, we explore the dataset to understand the underlying patterns and relationships between features. This includes:
- Descriptive statistics
- Visualizations (e.g., histograms, scatter plots, correlation heatmaps)
- Identifying and handling anomalies

## Model Comparison
We compare multiple machine learning models to identify the best-performing one for our classification task. The models evaluated include:
- Logistic Regression
- K Nearest Neighbors
- Multilayer Perceptron
- Random Forest
- XGBoost
- TPOT (AutoML Python library)

Each model is evaluated based on performance metrics such as accuracy, precision, recall, F1-score, and more.

## Deployment
The final trained model is deployed to a production server. For this purpose we utilized a relatively new thing in model serving called BentoML and created two API endpoints for making predictions depending on how user wants to create predictions (one by one or multiple at once).

## Conclusion
This project demonstrates the end-to-end process of building and deploying a machine learning model for hotel reservations classification. The steps taken ensure that the model is robust, accurate, and ready for real-world applications.

## Authors
This project was completed as part of a university assignment by:
- [Radovan Drašković](https://github.com/Drashko73)
- [Marija Jolović](https://github.com/marijajolovic)

We hope this project serves as a valuable learning experience and a useful reference for future work in machine learning and data science.