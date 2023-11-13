import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier


def fit_svm_classifier(X, y):
    pipeline = make_pipeline(RobustScaler(), LinearSVC(random_state=0, tol=1e-5, class_weight="balanced"))
    pipeline.fit(X, y)
    return pipeline

def fit_logistic_regression(X, y):
    # It is called Logistic Regression but it is really a classifier (source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    
    regression = LogisticRegression(class_weight= "balanced", solver="newton-cholesky", max_iter=5000, n_jobs=-1)
    pipeline = make_pipeline(StandardScaler(), regression)
    pipeline.fit(X, y)
    return regression

def fit_random_forest(X, y):
    forest = XGBClassifier(n_estimators=100, random_state=10, max_depth = 100, objective='binary:logistic', scale_pos_weight=40)
    forest.fit(X, y)
    return forest

def run_and_compare(train_X, train_y, test_x, test_y):
    print(f"baseline balanced-accuracy-score: {balanced_accuracy_score(test_y, np.zeros(test_y.shape))}")

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
    
    # print("Running Random Forest Classifier ....")
    # forest = fit_random_forest(train_X, train_y)
    # forest_predictions = forest.predict(test_x)
    # forest_f1 = f1_score(test_y, forest_predictions)
    # print(f"DT f1-score: {forest_f1}")
    # plot_confusion_matrix(test_y, forest_predictions)

def tune_hyperparameters(X, y, parameters, model):
    searcher = RandomizedSearchCV(model, parameters, scoring = "balanced_accuracy")
    searcher.fit(X, y)
    return searcher.best_params_, searcher.best_estimator_

def plot_confusion_matrix(ground_truth, predictions, title = "Confusion Matrix"):
    confusion_array = confusion_matrix(ground_truth, predictions, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_array)
    disp.plot()
    plt.show()



