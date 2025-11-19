import pandas as pd
import numpy as np
from PIL import Image
import joblib
from constants import FASHION_LOGISTIC_MODEL_PATH

# Load the trained logistic regression pipeline
model = joblib.load(FASHION_LOGISTIC_MODEL_PATH)

# Map numeric labels to fashion item names
label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def load_image_as_df(path):
    """Load a single image and convert to DataFrame with pixel columns."""
    img = Image.open(path).convert("L").resize((28, 28))  # grayscale 28x28
    img_array = np.array(img).reshape(1, -1) / 255.0       # normalize 0-1
    columns = [f'pixel{i}' for i in range(1, 785)]        # pixel1 → pixel784
    df = pd.DataFrame(img_array, columns=columns)
    return df

# Path to your image
img_path = "test01.jpg"


# Convert image to DataFrame
X_new = load_image_as_df(img_path)
print("X_new shape:", X_new.shape)
print("X_new sample:", X_new.head())

# Predict using pipeline
predicted_label = model.predict(X_new)
predicted_class = label_map.get(predicted_label[0], "Unknown")

print("Predicted label:", predicted_label[0])
print("Predicted class:", predicted_class)
