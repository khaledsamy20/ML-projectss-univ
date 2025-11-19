from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FASHION_TRAIN_DATA_PATH = PROJECT_ROOT / "artifacts/row_data/structure_data/fashion-mnist_train.csv"
FASHION_TEST_DATA_PATH  = PROJECT_ROOT / "artifacts/row_data/structure_data/fashion-mnist_test.csv"
FASHION_LOGISTIC_MODEL_PATH = PROJECT_ROOT / "artifacts/model/fashion-mnist_model.pkl"
FASHION_KMEANS_MODEL_PATH = PROJECT_ROOT / "artifacts/model/fashion-mnist_kmeans_model.pkl"
