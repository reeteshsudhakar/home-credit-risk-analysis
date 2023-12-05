import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def fit_svm_classifier(X, y):
    pipeline = make_pipeline(MinMaxScaler(), SGDClassifier(loss="hinge", random_state=0, class_weight="balanced", max_iter=10000))
    pipeline.fit(X, y)
    return pipeline

def fit_lr_classifier(X, y):
    pipeline = make_pipeline(MinMaxScaler(), LogisticRegression(C=2, random_state=0, class_weight="balanced", max_iter=10000))
    pipeline.fit(X, y)
    return pipeline

def fit_random_forest_classifier(X, y):
    pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier(max_depth=10, random_state=0, class_weight="balanced"))
    pipeline.fit(X, y)
    return pipeline

classifier_functions = {
    "svm": fit_svm_classifier,
    "lr": fit_lr_classifier,
    "rf": fit_random_forest_classifier
}

classifier_names = {
    "svm": "Support Vector Machine",
    "lr": "Logistic Regression",
    "rf": "Random Forest"
}

def run_and_compare(train_X, train_y, test_x, test_y, model: str):
    print(f"baseline balanced-accuracy-score: {balanced_accuracy_score(test_y, np.zeros(test_y.shape))}")

    print(f"Running {classifier_names[model]} Classifier ....")
    
    fit_model = classifier_functions[model](train_X, train_y)
    fit_model_balanced_accuracy = balanced_accuracy_score(test_y, fit_model.predict(test_x))
    
    print(f"{model} balanced accuracy: {fit_model_balanced_accuracy}")
    plot_confusion_matrix(test_y, fit_model.predict(test_x), title=f"{model} Confusion Matrix")

def tune_hyperparameters(X, y, parameters, model):
    searcher = RandomizedSearchCV(model, parameters, scoring = "balanced_accuracy")
    searcher.fit(X, y)
    return searcher.best_params_, searcher.best_estimator_

def plot_confusion_matrix(ground_truth, predictions, title = "Confusion Matrix"):
    confusion_array = confusion_matrix(ground_truth, predictions, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_array)
    disp.plot()
    plt.show()




