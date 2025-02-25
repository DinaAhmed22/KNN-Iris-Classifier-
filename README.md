# KNN-Iris-Classifier-
A machine learning project that implements and optimizes a K-Nearest Neighbors (KNN) classifier on the Iris dataset. The model undergoes hyperparameter tuning, cross-validation, and learning curve analysis to achieve optimal performance.
📌 Features
✅ Loads the Iris dataset from Kaggle
✅ Preprocesses the data (label encoding & feature scaling)
✅ Performs hyperparameter tuning using Grid Search & Stratified K-Fold Cross-Validation
✅ Trains and evaluates the KNN model with the best K-value
✅ Plots a learning curve to analyze train vs. test accuracy

📊 Technologies Used
Python 🐍
Scikit-learn (Machine Learning)
Matplotlib (Data Visualization)
Pandas & NumPy (Data Processing)
📈 Results & Accuracy
Best K-value is selected using GridSearchCV
Mean cross-validation accuracy: ~1.0000 ± 0.0000
Model achieves high test accuracy, ensuring generalization
🚀 What’s Next? A Better Model!
While KNN performs well, it can be slow for large datasets and is sensitive to noisy data. To improve performance, the next step is to use:

🔹 Support Vector Machine (SVM) – Finds the best decision boundary between classes, performing better on complex, high-dimensional data.
🔹 Random Forest – A robust ensemble learning method that reduces overfitting and improves accuracy.
🔹 Neural Networks (MLPClassifier) – A deep learning approach that can handle complex decision boundaries effectively.


