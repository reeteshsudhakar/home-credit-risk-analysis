import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, roc_auc_score


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

def fit_voting_classifier(X, y):
    svm_pipeline = make_pipeline(MinMaxScaler(), SGDClassifier(loss="log_loss", random_state=0, class_weight="balanced", max_iter=10000))
    lr_pipeline = make_pipeline(MinMaxScaler(), LogisticRegression(C=2, random_state=0, class_weight="balanced", max_iter=10000))
    rf_pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier(max_depth=10, random_state=0, class_weight="balanced"))
    pipeline = VotingClassifier(estimators=[('svm', svm_pipeline), ('lr', lr_pipeline), ('rf', rf_pipeline)], voting='soft')
    pipeline.fit(X, y)
    return pipeline


classifier_functions = {
    "svm": fit_svm_classifier,
    "lr": fit_lr_classifier,
    "rf": fit_random_forest_classifier,
    "voting": fit_voting_classifier
}

classifier_names = {
    "svm": "Support Vector Machine",
    "lr": "Logistic Regression",
    "rf": "Random Forest", 
    "voting": "Voting"
}

def run_and_compare(train_X, train_y, test_x, test_y, model: str):
    print(f"baseline balanced-accuracy-score: {balanced_accuracy_score(test_y, np.zeros(test_y.shape))}")

    print(f"Running {classifier_names[model]} Classifier ....")
    
    fit_model = classifier_functions[model](train_X, train_y)
    fit_model_balanced_accuracy = balanced_accuracy_score(test_y, fit_model.predict(test_x))
    fit_model_f1_score = f1_score(test_y, fit_model.predict(test_x), average="weighted")
    fit_model_precision_score = precision_score(test_y, fit_model.predict(test_x), average="weighted")
    
    print(f"{model} balanced accuracy: {fit_model_balanced_accuracy}")
    print(f"{model} f1 score: {fit_model_f1_score}")
    print(f"{model} precision score: {fit_model_precision_score}")
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




