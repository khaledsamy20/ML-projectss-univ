import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(path):
    """Loads the dataset from the given path."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Preprocesses the data."""
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    return df

def train_model(X_train, y_train):
    """Trains the KNN Regressor model."""
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    return knn, sc

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluates the model and prints the metrics."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R² Score: {r2}')
    
    return y_pred

def plot_results(model, X_train, y_train, y_test, y_pred, output_path):
    """Generates and saves plots."""
    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_loss = -train_scores.mean(axis=1)
    test_loss = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_loss, label='Training Loss')
    plt.plot(train_sizes, test_loss, label='Validation Loss')
    plt.xlabel("Training Size")
    plt.ylabel("MSE Loss")
    plt.title("KNN Regression Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/learning_curve.png")
    plt.close()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals (Error)")
    plt.title("Residual Plot")
    plt.savefig(f"{output_path}/residual_plot.png")
    plt.close()

def save_model(model, scaler, model_path, scaler_path):
    """Saves the trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def main(data_path, model_output_path, scaler_output_path, plots_output_path):
    """Main function to run the KNN regressor model training."""
    df = load_data(data_path)
    df = preprocess_data(df)

    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, scaler = train_model(X_train, y_train)
    y_pred = evaluate_model(model, scaler, X_test, y_test)
    
    plot_results(model, X_train, y_train, y_test, y_pred, plots_output_path)
    save_model(model, scaler, model_output_path, scaler_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a KNN Regressor model.')
    parser.add_argument('--data_path', type=str, default='../../artifacts/row_data/structure_data/insurance.csv', help='Path to the input data file.')
    parser.add_argument('--model_output_path', type=str, default='../../artifacts/model/knn_regressor_model.pkl', help='Path to save the trained model.')
    parser.add_argument('--scaler_output_path', type=str, default='../../artifacts/model/knn_scaler.pkl', help='Path to save the scaler.')
    parser.add_argument('--plots_output_path', type=str, default='../../artifacts/plots/knn_regressor', help='Path to save the evaluation plots.')
    args = parser.parse_args()

    main(args.data_path, args.model_output_path, args.scaler_output_path, args.plots_output_path)
