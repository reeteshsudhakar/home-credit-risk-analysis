from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd



def get_feature_imp(X_train, y_train):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    importances = tree.feature_importances_
    feat_importances = pd.DataFrame(importances, index=X_train.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feat_importances.plot(kind='barh', figsize=(80,60))
