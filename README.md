# Machine Learning Projects

This repository contains a collection of machine learning models for various tasks, including classification and regression.

---

## Project Description Document

Below is a summary of the models implemented in this project, following the specified documentation structure.

### 1. Logistic Regression Model (Classification)

*   **a. General Information on Dataset:**
    *   **Name:** Fashion-MNIST
    *   **Number of Classes:** 10
    *   **Labels:** `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`
    *   **Total Samples:** 70,000
    *   **Training/Validation/Testing Split:** 60,000 samples were used for training and 10,000 for testing. During development, the training set was split 80/20 for training and validation.
    *   **Size of Each Sample:** 28x28 pixel grayscale images.

*   **b. Implementation Details:**
    *   **Feature Extraction:** The initial 784 pixel values were scaled using `StandardScaler`. Dimensionality was then reduced to **100 principal components** using PCA.
    *   **Cross-validation:** Yes, **5-fold cross-validation** was performed on the training data, yielding a mean accuracy of approximately **84.8%**.
    *   **Hyperparameters:**
        *   **Model:** A `Pipeline` combining `StandardScaler`, `PCA`, and `LogisticRegression`.
        *   **PCA `n_components`:** 100
        *   **Optimizer (`solver`):** `lbfgs`
        *   **Regularization:** L2
        *   **`max_iter`:** 1000

*   **c. Results Details (on Testing Data):**
    *   **Accuracy:** **85.1%**
    *   **Confusion Matrix:** A detailed confusion matrix was generated to show per-class performance.
    *   **ROC Curve:** ROC curves were plotted for each of the 10 classes, with AUC scores indicating good discriminative ability.
    *   **Loss Curve:** A loss curve was not generated for this model.

---

### 2. K-Means Clustering Model (Clustering)

*   **a. General Information on Dataset:**
    *   Same as the Logistic Regression model (Fashion-MNIST).

*   **b. Implementation Details:**
    *   **Feature Extraction:** Same as the Logistic Regression model (`StandardScaler` followed by `PCA` with 100 components).
    *   **Cross-validation:** Not used for this unsupervised model.
    *   **Hyperparameters:**
        *   **Model:** A `Pipeline` combining `StandardScaler`, `PCA`, and `KMeans`.
        *   **`n_clusters`:** 10
        *   **`random_state`:** 42

*   **c. Results Details:**
    *   As an unsupervised model, standard classification metrics do not apply. Performance was evaluated using clustering metrics on the test set:
    *   **Adjusted Rand Index (ARI):** 0.333
    *   **Normalized Mutual Information (NMI):** 0.486
    *   **Loss Curve, Accuracy, Confusion Matrix, ROC Curve:** Not applicable for this task.

---

### 3. Linear Regression Model (Regression)

*   **a. General Information on Dataset:**
    *   **Name:** Insurance Charges Prediction Dataset
    *   **Total Samples:** 1,199 (after removing outliers).
    *   **Training/Testing Split:** 80% for training, 20% for testing.

*   **b. Implementation Details:**
    *   **Feature Extraction:** 5 features were used: `age`, `sex`, `bmi`, `children`, and `smoker`.
    *   **Cross-validation:** Not used.
    *   **Hyperparameters:** A standard `LinearRegression` model with default parameters was used.

*   **c. Results Details (on Testing Data):**
    *   **R-squared (R²) Score:** **0.524** (The model explains ~52.4% of the variance in the charges).
    *   **Root Mean Squared Error (RMSE):** 5,364.79
    *   **Plots:** The analysis includes an "actual vs. prediction" scatter plot and an error distribution histogram.
    *   **Loss Curve, Accuracy, Confusion Matrix, ROC Curve:** Not applicable for this regression task.

---

### 4. K-Nearest Neighbors (KNN) Regressor Model (Regression)

*   **a. General Information on Dataset:**
    *   **Name:** Insurance Charges Prediction Dataset
    *   **Total Samples:** 1,338
    *   **Training/Testing Split:** 80% for training, 20% for testing.

*   **b. Implementation Details:**
    *   **Feature Extraction:** 8 features were used after one-hot encoding categorical variables (`sex`, `smoker`, `region`). All features were then scaled using `StandardScaler`.
    *   **Cross-validation:** Not used.
    *   **Hyperparameters:** `KNeighborsRegressor` was used with `n_neighbors=5`.

*   **c. Results Details (on Testing Data):**
    *   **R-squared (R²) Score:** **0.804** (The model explains ~80.4% of the variance in the charges).
    *   **Root Mean Squared Error (RMSE):** 5,519.05
    *   **Loss Curve:** A learning curve showing MSE loss versus training size was generated.
    *   **Plots:** The analysis includes a residual plot and an "actual vs. predicted" plot.
    *   **Accuracy, Confusion Matrix, ROC Curve:** Not applicable for this regression task.
