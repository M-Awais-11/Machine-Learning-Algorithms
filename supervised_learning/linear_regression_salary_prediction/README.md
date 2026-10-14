# Salary Prediction using Linear Regression

## Project Overview

This project focuses on predicting employee salaries using Linear Regression techniques.

The dataset contains information about employees, including:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience
- Salary

Two regression approaches are implemented:

1. Simple Linear Regression
2. Multiple Linear Regression

The models are evaluated using RMSE and R² Score, and their performance is compared to select the better-performing model.

---

## Objectives

The main objectives of this project are:

- Explore and understand the salary dataset.
- Clean and preprocess the data.
- Handle duplicate and missing values.
- Perform exploratory data analysis.
- Build a Simple Linear Regression model.
- Build a Multiple Linear Regression model.
- Handle categorical features using One-Hot Encoding.
- Evaluate models using RMSE and R² Score.
- Compare the performance of both models.
- Select and save the best-performing model.
- Use the trained model to predict the salary of a new employee.

---

## Dataset Features

| Feature | Description |
|---|---|
| Age | Age of the employee |
| Gender | Gender of the employee |
| Education Level | Highest education level |
| Job Title | Job position/title |
| Years of Experience | Years of professional experience |
| Salary | Employee salary and target variable |

---

## Machine Learning Workflow

```text
Load Dataset
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Simple Linear Regression
     ↓
Multiple Linear Regression
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Save Best Model
     ↓
New Salary Prediction