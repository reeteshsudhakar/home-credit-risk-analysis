import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def fit_svm_classifier(X, y):
    pipeline = make_pipeline(RobustScaler(), SGDClassifier(loss = "hinge", random_state=0, tol=1e-5, class_weight="balanced", max_iter=10000, alpha = 0.05, early_stopping = True))
    pipeline.fit(X, y)
    return pipeline

def fit_logistic_regression(X, y):
    # It is called Logistic Regression but it is really a classifier (source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    
    regression = LogisticRegression(l1_ratio = 0.5, class_weight= "balanced", max_iter=10000, penalty = "elasticnet", solver="saga")
    pipeline = make_pipeline(StandardScaler(), regression)
    pipeline.fit(X, y)
    return regression

def run_and_compare(train_X, train_y, test_x, test_y):
    # print(f"baseline balanced-accuracy-score: {balanced_accuracy_score(test_y, np.zeros(test_y.shape))}")

    print("Running Support Vector Classifier ....")
    svm = fit_svm_classifier(train_X, train_y)
    svm_balanced_accuracy = balanced_accuracy_score(test_y, svm.predict(test_x))
    print(f"SVM balanced accuracy: {svm_balanced_accuracy}")
    plot_confusion_matrix(test_y, svm.predict(test_x), title="Support Vector Machine Confusion Matrix")
    
    print("Running Logistic Classifier ....")
    logit = fit_logistic_regression(train_X, train_y)
    logit_balanced_accuracy = balanced_accuracy_score(test_y, logit.predict(test_x))
    print(f"Logistic Classifier balanced accuracy: {logit_balanced_accuracy}")
    plot_confusion_matrix(test_y, logit.predict(test_x), title="Logistic Regression Confusion Matrix")

def tune_hyperparameters(X, y, parameters, model):
    searcher = RandomizedSearchCV(model, parameters, scoring = "balanced_accuracy")
    searcher.fit(X, y)
    return searcher.best_params_, searcher.best_estimator_

def plot_confusion_matrix(ground_truth, predictions, title = "Confusion Matrix"):
    confusion_array = confusion_matrix(ground_truth, predictions, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_array)
    disp.plot()
    plt.show()




