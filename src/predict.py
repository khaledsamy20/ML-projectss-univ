import pandas as pd
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
from constants import FASHION_LOGISTIC_MODEL_PATH

# Load the trained logistic regression pipeline
model = joblib.load(FASHION_LOGISTIC_MODEL_PATH)

# Map numeric labels to fashion item names for the classes the model was trained on
label_map = {
    1: "Trouser",
    3: "Dress",
    5: "Sandal",
    8: "Bag",
    9: "Ankle boot"
}

def get_hog_features(path):
    """Load an image, compute HOG features, and return as a DataFrame."""

    img = Image.open(path).convert("L").resize((28, 28))
    img_array = np.array(img)

    hog_features = hog(
        img_array,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    # Reshape for a single sample and create DataFrame
    hog_features = hog_features.reshape(1, -1)
    columns = [f'pixel{i}' for i in range(hog_features.shape[1])]
    df = pd.DataFrame(hog_features, columns=columns)
    return df



img_path = "Trouser.jpeg" 

# Get HOG features for the image
X_new = get_hog_features(img_path)


predicted_label = model.predict(X_new)
predicted_class = label_map.get(predicted_label[0], "Unknown")

print("Predicted label:", predicted_label[0])
print("Predicted class:", predicted_class)
