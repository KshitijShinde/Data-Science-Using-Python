import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

    model_filename = "iris_model.pkl"
    print(f"Saving model to {model_filename}...")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully! Preparing to verify load...")
    
    # Verification step
    print(f"Verifying {model_filename} loads correctly...")
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
    
    # Test loaded model
    test_pred = loaded_model.predict(X_test[:1])
    print(f"Test prediction on first sample: predicted={test_pred[0]}, actual={y_test[0]}")
    assert test_pred[0] == y_test[0], "Prediction mismatch on loaded model!"
    print("Verification completed successfully.")

if __name__ == "__main__":
    train_and_save_model()
