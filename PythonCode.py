import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_dataset():
    """Loads the Iris dataset from the Kaggle API."""
    import kagglehub
    path = kagglehub.dataset_download("saurabh00007/iriscsv")
    df = pd.read_csv(f"{path}/IRIS.csv")
    return df

def preprocess_data(df):
    """Preprocesses the dataset: encodes labels, normalizes features."""
    species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    df["Species"] = df["Species"].map(species_mapping)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def tune_knn(X, y):
    """Finds the optimal K value using Stratified K-Fold cross-validation."""
    param_grid = {"n_neighbors": range(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=StratifiedKFold(n_splits=5))
    grid_search.fit(X, y)
    return grid_search.best_params_["n_neighbors"]

def plot_learning_curve(X, y):
    """Plots the learning curve to visualize training vs test accuracy with cross-validation."""
    train_sizes, train_scores, test_scores = learning_curve(
        KNeighborsClassifier(n_neighbors=5), X, y, cv=StratifiedKFold(n_splits=5), train_sizes=np.linspace(0.1, 0.8, 8), scoring='accuracy', n_jobs=-1
    )
    train_means = train_scores.mean(axis=1)
    test_means = test_scores.mean(axis=1)
    train_stds = train_scores.std(axis=1)
    test_stds = test_scores.std(axis=1)
    
    plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_means - test_stds, test_means + test_stds, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_means, label="Train Accuracy", marker='o', color="blue")
    plt.plot(train_sizes, test_means, label="Test Accuracy", marker='s', color="orange")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for KNN")
    plt.legend()
    plt.show()
    
    print(f"Mean cross-validation accuracy: {test_means[-1]:.4f} ± {test_stds[-1]:.4f}")

def train_and_evaluate(X_train, X_test, y_train, y_test, k):
    """Trains the KNN model and evaluates its performance."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    best_k = tune_knn(X_train, y_train)
    print(f"Best K: {best_k}")
    train_and_evaluate(X_train, X_test, y_train, y_test, best_k)
    plot_learning_curve(X_train, y_train)

if __name__ == "__main__":
    main()
