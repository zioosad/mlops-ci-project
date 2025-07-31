import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import sys

# Define a minimum accuracy threshold
MIN_ACCURACY = 0.9

print("Starting validation...")

# 1. Load the model and test data
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    X_test = pd.read_csv('models/X_test.csv')
    y_test = pd.read_csv('models/y_test.csv')
    print("Model and test data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    sys.exit(1)


# 2. Make predictions
print("Making predictions on the test set...")
predictions = model.predict(X_test)

# 3. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# 4. Check if accuracy meets the threshold
if accuracy < MIN_ACCURACY:
    print(f"Validation FAILED: Accuracy ({accuracy:.4f}) is below the threshold ({MIN_ACCURACY}).")
    sys.exit(1)
else:
    print(f"Validation PASSED: Accuracy ({accuracy:.4f}) meets the threshold.")
    sys.exit(0)