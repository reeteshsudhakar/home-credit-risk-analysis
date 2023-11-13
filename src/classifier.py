import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from xgboost import XGBClassifier


def fit_svm_classifier(X, y):
    pipeline = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, class_weight="balanced"))
    pipeline.fit(X, y)
    return pipeline

def fit_logistic_regression(X, y):
    # It is called Logistic Regression but it is really a classifier (source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    regression = LogisticRegression(class_weight= "balanced", solver="newton-cholesky")
    regression.fit(X, y)
    return regression

def fit_random_forest(X, y):
    forest = XGBClassifier(n_estimators=100, random_state=10, max_depth = 100, objective='binary:logistic', scale_pos_weight=40)
    forest.fit(X, y)
    return forest

def run_and_compare(train_X, train_y, test_x, test_y):
    print(f"baseline: {f1_score(test_y, np.zeros(test_y.shape))}")

    print("Running Support Vector Classifier ....")
    svm = fit_svm_classifier(train_X, train_y)
    svm_f1 = f1_score(test_y, svm.predict(test_x), average = "weighted")
    print(f"SVM f1-score: {svm_f1}")
    plot_confusion_matrix(test_y, svm.predict(test_x))
    
    print("Running Logistic Classifier ....")
    logit = fit_logistic_regression(train_X, train_y)
    logit_f1 = f1_score(test_y, logit.predict(test_x))
    print(f"Logistic Classifier f1-score: {logit_f1}")
    plot_confusion_matrix(test_y, logit.predict(test_x))
    
    print("Running Random Forest Classifier ....")
    forest = fit_random_forest(train_X, train_y)
    forest_predictions = forest.predict(test_x)
    forest_f1 = f1_score(test_y, forest_predictions)
    print(f"DT f1-score: {forest_f1}")
    plot_confusion_matrix(test_y, forest_predictions)

def tune_hyperparameters(X, y, parameters, model):
    searcher = RandomizedSearchCV(model, parameters, scoring = "balanced_accuracy")
    searcher.fit(X, y)
    return searcher.best_params_, searcher.best_estimator_

def plot_confusion_matrix(ground_truth, predictions):
    confusion_array = confusion_matrix(ground_truth, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_array)
    disp.plot()
    plt.show()



