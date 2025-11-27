import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

def load_data(path):
    """Loads the dataset from the given path."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Preprocesses the data."""
    df = df.drop(['region'], axis=1)
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})

    # Remove outliers
    q1 = df['charges'].quantile(q=0.25)
    q3 = df['charges'].quantile(q=0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr
    max_val = q3 + 1.5 * iqr
    df = df[(df['charges'] >= min_val) & (df['charges'] <= max_val)]
    
    return df

def train_model(X_train, y_train):
    """Trains the Linear Regression model."""
    linr = LinearRegression()
    linr.fit(X_train, y_train)
    return linr

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints the metrics."""
    y_predict = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    rmse = np.sqrt(mse)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}')
    print(f'Root Mean Squared Error: {rmse}')
    
    return y_predict

def plot_results(y_test, y_pred, output_path):
    """Generates and saves plots."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.grid()
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs Prediction')
    plt.savefig(f"{output_path}/actual_vs_prediction.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    the_error = y_test - y_pred
    sns.histplot(the_error, kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(f"{output_path}/error_distribution.png")
    plt.close()

def save_model(model, path):
    """Saves the trained model."""
    joblib.dump(model, path)

def main(data_path, model_output_path, plots_output_path):
    """Main function to run the linear regression model training."""
    df = load_data(data_path)
    df = preprocess_data(df)

    y = df['charges']
    X = df.drop('charges', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    
    plot_results(y_test, y_pred, plots_output_path)
    save_model(model, model_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Linear Regression model.')
    parser.add_argument('--data_path', type=str, default='../../artifacts/row_data/structure_data/insurance.csv', help='Path to the input data file.')
    parser.add_argument('--model_output_path', type=str, default='../../artifacts/model/linear_regression_model.pkl', help='Path to save the trained model.')
    parser.add_argument('--plots_output_path', type=str, default='../../artifacts/plots/linear_regression', help='Path to save the evaluation plots.')
    args = parser.parse_args()

    main(args.data_path, args.model_output_path, args.plots_output_path)
