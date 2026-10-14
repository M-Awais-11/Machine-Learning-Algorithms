LINEAR REGRESSION - COMPLETE NOTES
===================================

1. WHAT IS LINEAR REGRESSION?
-----------------------------

Linear Regression is a supervised machine learning algorithm used to
predict a continuous numerical target variable.

Examples:
- Predicting Salary
- Predicting House Price
- Predicting Sales
- Predicting Temperature

In this project:

Input Features:
- Age
- Gender
- Education Level
- Job Title
- Years of Experience

Target:
- Salary


2. TYPES OF LINEAR REGRESSION
-----------------------------

There are two main types used in this project:

1. Simple Linear Regression
2. Multiple Linear Regression


3. SIMPLE LINEAR REGRESSION
---------------------------

Simple Linear Regression uses only ONE independent variable to predict
the dependent variable.

Example:

Years of Experience -> Salary

General equation:

y = b0 + b1x

Where:

y  = Predicted value / dependent variable
x  = Independent variable
b0 = Intercept
b1 = Slope / coefficient


4. INTERCEPT
------------

The intercept is the predicted value of y when x = 0.

Formula:

b0 = y_bar - b1 * x_bar

Where:

y_bar = Mean of y
x_bar = Mean of x


5. SLOPE / COEFFICIENT
----------------------

The slope represents how much the target variable changes when the
independent variable increases by one unit.

Formula:

             SUM((xi - x_bar)(yi - y_bar))
b1 = ----------------------------------------------
             SUM((xi - x_bar)^2)

Where:

xi    = Individual x value
yi    = Individual y value
x_bar = Mean of x
y_bar = Mean of y


6. PREDICTION FORMULA
---------------------

After calculating the intercept and slope:

y_hat = b0 + b1x

Example:

If:

b0 = 30000
b1 = 10000
x  = 5 years

Then:

Predicted Salary = 30000 + (10000 × 5)

Predicted Salary = 80000


7. MULTIPLE LINEAR REGRESSION
-----------------------------

Multiple Linear Regression uses two or more independent variables to
predict a continuous target variable.

Example:

Age
Gender
Education Level
Job Title
Years of Experience

        ↓

     Salary


General equation:

y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn

Where:

y  = Predicted target
b0 = Intercept
b1, b2, ..., bn = Model coefficients
x1, x2, ..., xn = Input features


8. DIFFERENCE BETWEEN SIMPLE AND MULTIPLE LINEAR REGRESSION
------------------------------------------------------------

Simple Linear Regression:

One feature -> Target

Example:

Years of Experience -> Salary


Multiple Linear Regression:

Multiple features -> Target

Example:

Age + Education + Experience + Job Title -> Salary


9. DEPENDENT AND INDEPENDENT VARIABLES
--------------------------------------

Independent Variables (Features):
Variables used to make predictions.

Example:

Age
Years of Experience
Education Level
Job Title

Dependent Variable (Target):
The variable that we want to predict.

Example:

Salary


10. RESIDUAL / ERROR
--------------------

Residual is the difference between the actual value and predicted value.

Formula:

Residual = Actual Value - Predicted Value

e_i = y_i - y_hat_i

Where:

y_i     = Actual value
y_hat_i = Predicted value


11. PREDICTION ERROR
--------------------

Prediction error represents how far the model prediction is from the
actual value.

Formula:

Error = y - y_hat


12. LEAST SQUARES METHOD
------------------------

Linear Regression commonly uses the Ordinary Least Squares (OLS)
method.

The objective is to find the best-fitting line by minimizing the
sum of squared residuals.

Objective:

Minimize:

SUM((yi - y_hat_i)^2)

This is called the:

Sum of Squared Errors (SSE)


13. SUM OF SQUARED ERRORS (SSE)
-------------------------------

SSE measures the total squared prediction error.

Formula:

SSE = SUM((yi - y_hat_i)^2)

Lower SSE means smaller total squared errors.


14. MEAN SQUARED ERROR (MSE)
----------------------------

MSE is the average of squared errors.

Formula:

             1
MSE = ------- SUM((yi - y_hat_i)^2)
             n

Where:

n = Number of observations


15. ROOT MEAN SQUARED ERROR (RMSE)
----------------------------------

RMSE is the square root of MSE.

Formula:

RMSE = sqrt(MSE)

Or:

             1
RMSE = sqrt(--- SUM((yi - y_hat_i)^2))
             n

RMSE is useful because it is expressed in the same unit as the target.

For Salary Prediction:

RMSE is expressed in salary units.

Lower RMSE = Better prediction error performance.


16. R-SQUARED (R²)
------------------

R² measures how much of the variation in the target variable is
explained by the regression model.

Formula:

             SSE
R² = 1 - ---------
             SST

Where:

SSE = Sum of Squared Errors

SST = Total Sum of Squares


17. TOTAL SUM OF SQUARES (SST)
------------------------------

SST measures the total variation in the actual target values.

Formula:

SST = SUM((yi - y_bar)^2)

Where:

yi    = Actual value
y_bar = Mean of actual values


18. INTERPRETING R²
-------------------

R² = 1

The model explains 100% of the variation in the target.

R² = 0

The model does not explain the variation better than predicting the
mean of the target.

R² can sometimes be negative on unseen/test data.

Higher R² generally means that the model explains more of the
variation in the target.


19. ADJUSTED R²
---------------

Adjusted R² is an extension of R² that accounts for the number of
predictors in the model.

Formula:

Adjusted R² =
1 - [(1 - R²)(n - 1) / (n - p - 1)]

Where:

n = Number of observations
p = Number of independent variables

Adjusted R² is especially useful when comparing models with different
numbers of features.


20. TRAINING DATA AND TEST DATA
------------------------------

The dataset is usually divided into:

Training Set:
Used to train the model.

Test Set:
Used to evaluate the model on unseen data.

Common split:

80% Training
20% Testing


21. TRAIN/TEST SPLIT
--------------------

Example in Scikit-learn:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

Where:

test_size=0.2 means 20% data is used for testing.

random_state=42 makes the split reproducible.


22. FEATURES (X) AND TARGET (y)
--------------------------------

Features:

X = Input variables

Target:

y = Output variable

Example:

X = Age, Gender, Education Level, Job Title,
    Years of Experience

y = Salary


23. CATEGORICAL FEATURES
------------------------

Categorical variables contain categories instead of numerical values.

Examples:

Gender:
- Male
- Female

Education Level:
- Bachelor's
- Master's
- PhD

Job Title:
- Software Engineer
- Data Analyst
- Manager


24. ONE-HOT ENCODING
--------------------

Machine learning models require numerical input.

One-Hot Encoding converts categorical values into binary columns.

Example:

Gender:

Male
Female

Can become:

Gender_Male
Gender_Female

Male:

1 0

Female:

0 1

In Scikit-learn:

OneHotEncoder(handle_unknown="ignore")


25. DUMMY VARIABLE TRAP
-----------------------

When using one-hot encoding, including all dummy variables can create
redundant information in some regression settings.

For k categories, traditionally k-1 dummy variables can be used.

Example:

Education:

Bachelor's
Master's
PhD

Can be represented using:

Education_Master's
Education_PhD

Bachelor's becomes the reference category.

Scikit-learn pipelines can manage categorical encoding depending on
the configuration used.


26. DATA PREPROCESSING
----------------------

Common preprocessing steps:

1. Load dataset
2. Inspect data
3. Check data types
4. Check missing values
5. Check duplicate values
6. Handle missing values
7. Encode categorical features
8. Separate X and y
9. Split training and testing data


27. MISSING VALUES
------------------

Missing values are values that are not available in the dataset.

Examples:

NaN
None
Empty values

For numerical features:

Median or Mean can be used for imputation.

For categorical features:

Mode / Most Frequent value can be used.

Important:

The target variable should generally not be artificially filled with
the median when it is unknown, because the actual target value is
required for supervised training.


28. DATA LEAKAGE
----------------

Data leakage occurs when information from outside the training data
is used during model training in a way that gives the model
unrealistic information.

Example:

Calculating preprocessing statistics using the complete dataset
before train/test splitting.

Best practice:

Fit preprocessing only on training data.

Scikit-learn Pipeline and ColumnTransformer help prevent this issue.


29. LINEAR REGRESSION MODEL IN SCIKIT-LEARN
-------------------------------------------

Import:

from sklearn.linear_model import LinearRegression

Create model:

model = LinearRegression()

Train:

model.fit(X_train, y_train)

Predict:

y_pred = model.predict(X_test)


30. MODEL COEFFICIENTS
----------------------

After training:

model.coef_

returns the coefficients of the features.

For simple regression:

model.coef_[0]

represents the slope.

For multiple regression:

Each coefficient represents the estimated change in the target
associated with a one-unit change in that feature, holding the other
model features constant, subject to the encoding used.


31. MODEL INTERCEPT
-------------------

The intercept can be obtained using:

model.intercept_

It represents the predicted target when the numerical model inputs
are zero, with categorical variables represented according to the
model's encoding.


32. MODEL EVALUATION
-------------------

Common regression metrics:

1. MAE
2. MSE
3. RMSE
4. R² Score
5. Adjusted R²


33. MEAN ABSOLUTE ERROR (MAE)
-----------------------------

MAE calculates the average absolute difference between actual and
predicted values.

Formula:

             1
MAE = ------- SUM(|yi - y_hat_i|)
             n

Lower MAE is better.


34. MSE VS RMSE
---------------

MSE:

MSE = Average of squared errors

RMSE:

RMSE = Square root of MSE

RMSE is easier to interpret because it uses the same unit as the
target variable.


35. MAE VS RMSE
--------------

MAE treats errors more linearly.

RMSE gives larger errors more influence because errors are squared.

Therefore, RMSE is more sensitive to large prediction errors.


36. ASSUMPTIONS OF LINEAR REGRESSION
------------------------------------

Important assumptions include:

1. Linearity
2. Independence of observations
3. Homoscedasticity
4. No severe multicollinearity
5. Approximately normal residuals for classical inference


37. LINEARITY
-------------

The relationship between predictors and the target should be
appropriately modeled as linear.

Example:

Years of Experience and Salary should have a relationship that can
reasonably be represented by a linear model.


38. HOMOSCEDASTICITY
--------------------

Homoscedasticity means that the variance of residuals is approximately
constant across predicted values.

If residual variance changes substantially, this is called
heteroscedasticity.


39. MULTICOLLINEARITY
---------------------

Multicollinearity occurs when independent variables are highly
correlated with each other.

Example:

Years of Experience and Age may have a strong relationship.

High multicollinearity can make coefficient estimates unstable and
harder to interpret.


40. VARIANCE INFLATION FACTOR (VIF)
-----------------------------------

VIF is commonly used to detect multicollinearity.

Formula:

VIF_j = 1 / (1 - R_j²)

Where:

R_j² is the R² obtained by regressing predictor j on the other
predictors.

Higher VIF indicates stronger multicollinearity.


41. OVERFITTING
---------------

Overfitting occurs when a model learns the training data too closely
and performs poorly on unseen data.

Possible signs:

High training performance
Low test performance

Possible solutions:

- Use fewer features
- Collect more data
- Regularization
- Cross-validation
- Remove irrelevant features


42. UNDERFITTING
----------------

Underfitting occurs when the model is too simple to capture important
patterns in the data.

Possible signs:

Poor training performance
Poor test performance


43. BIAS AND VARIANCE
---------------------

Bias:

Error caused by overly simple assumptions.

Variance:

Error caused by a model being too sensitive to training data.

A good model attempts to balance bias and variance.


44. CROSS-VALIDATION
--------------------

Cross-validation evaluates a model across multiple train/validation
splits.

K-Fold Cross-Validation:

The dataset is divided into K folds.

The model is trained and evaluated K times.

Each fold is used as the validation set once.


45. RANDOM STATE
----------------

random_state is used to make random operations reproducible.

Example:

train_test_split(X, y, test_size=0.2, random_state=42)

Using the same random_state produces the same split.


46. MODEL COMPARISON
--------------------

For this project:

Simple Linear Regression:

Years of Experience -> Salary

Multiple Linear Regression:

Age + Gender + Education Level + Job Title +
Years of Experience -> Salary

Models can be compared using:

RMSE
R² Score

For RMSE:

Lower is generally better.

For R²:

Higher is generally better.


47. MODEL SELECTION
-------------------

The final model should be selected based on performance on the
appropriate validation/test data and the project requirements.

For this assignment, RMSE and R² are used for comparison.

Example:

If Model A has a lower RMSE and similar or better R² than Model B,
Model A may provide better predictive performance on the test set.


48. REGULARIZATION
------------------

Regularization reduces model complexity by adding a penalty to the
objective function.

Two common linear regression regularization methods are:

1. Ridge Regression
2. Lasso Regression


49. RIDGE REGRESSION
-------------------

Ridge Regression uses L2 regularization.

Objective:

Minimize:

SSE + alpha * SUM(beta_j²)

Where:

alpha = Regularization strength
beta_j = Model coefficients

Ridge generally shrinks coefficients toward zero but usually does not
make them exactly zero.


50. LASSO REGRESSION
-------------------

Lasso Regression uses L1 regularization.

Objective:

Minimize:

SSE + alpha * SUM(|beta_j|)

Lasso can shrink some coefficients exactly to zero, which can perform
feature selection.


51. RIDGE VS LASSO
------------------

Ridge:

- L2 penalty
- Shrinks coefficients
- Usually keeps all features

Lasso:

- L1 penalty
- Can make coefficients exactly zero
- Can perform feature selection


52. LINEAR REGRESSION WORKFLOW
------------------------------

Complete workflow:

1. Load Dataset
2. Explore Dataset
3. Check Data Types
4. Check Missing Values
5. Check Duplicate Values
6. Clean Data
7. Perform EDA
8. Define Features (X)
9. Define Target (y)
10. Encode Categorical Features
11. Split Data
12. Train Model
13. Make Predictions
14. Calculate RMSE
15. Calculate R²
16. Compare Models
17. Select Best Model
18. Save Model
19. Make New Predictions


53. IMPORTANT SCIKIT-LEARN FUNCTIONS
------------------------------------

Linear Regression:

LinearRegression()

Train:

model.fit(X_train, y_train)

Prediction:

model.predict(X_test)

RMSE:

np.sqrt(mean_squared_error(y_test, y_pred))

R²:

r2_score(y_test, y_pred)

Train/Test Split:

train_test_split()


54. IMPORTANT FORMULAS SUMMARY
------------------------------

Simple Linear Regression:

y_hat = b0 + b1x


Multiple Linear Regression:

y_hat = b0 + b1x1 + b2x2 + ... + bnxn


Residual:

e = y - y_hat


SSE:

SSE = SUM((y - y_hat)^2)


MSE:

MSE = SSE / n


RMSE:

RMSE = sqrt(MSE)


R²:

R² = 1 - (SSE / SST)


SST:

SST = SUM((y - y_bar)^2)


Adjusted R²:

Adjusted R² =
1 - [(1 - R²)(n - 1) / (n - p - 1)]


55. KEY TERMS
-------------

Regression:
Predicting a continuous numerical value.

Feature:
An input variable used by the model.

Target:
The output variable being predicted.

Coefficient:
Represents the estimated effect of a feature in the fitted linear
model.

Intercept:
The model's baseline prediction when numerical inputs are zero.

Prediction:
The output generated by the trained model.

Residual:
Difference between actual and predicted values.

RMSE:
Measures the typical size of prediction errors with greater
sensitivity to large errors.

R²:
Measures the proportion of target variance explained by the model.

Overfitting:
Model performs well on training data but poorly on unseen data.

Underfitting:
Model is too simple and performs poorly on both training and unseen
data.

One-Hot Encoding:
Converts categorical values into numerical binary features.

Train Set:
Data used to train the model.

Test Set:
Unseen data used to evaluate the model.


56. SALARY PREDICTION PROJECT
----------------------------

For the Salary Prediction project:

Target:

Salary

Simple Model:

Years of Experience -> Salary

Multiple Model:

Age
Gender
Education Level
Job Title
Years of Experience

        ↓

     Salary

Evaluation Metrics:

RMSE
R² Score

Model Saving:

Joblib

Saved Model:

best_salary_model.pkl


57. FINAL CONCEPT
-----------------

The main idea of Linear Regression is to find a mathematical
relationship between input features and a continuous target variable.

The model learns coefficients from training data and uses these
coefficients to make predictions for new, unseen data.

In simple form:

INPUT FEATURES
      ↓
LINEAR REGRESSION MODEL
      ↓
PREDICTED CONTINUOUS VALUE

For this project:

Employee Information
      ↓
Linear Regression
      ↓
Predicted Salary