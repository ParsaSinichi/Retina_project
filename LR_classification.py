import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from RF_deep_features import extract_features
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc,f1_score,matthews_corrcoef
import matplotlib.pyplot as plt
def print_metrics(y_true, y_pred, y_prob):

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
def calculate_class_weights(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight

    return class_weights
param_grid = {
    'C': [0.001, 0.009, 0.01, 0.1, 0.5, 0.9, 1, 1.2, 1.5, 2],
    'penalty': ['l1', 'l2']
}

train_features, train_labels = extract_features('train')
valid_features, valid_labels = extract_features('val')
test_features, test_labels = extract_features('test')
X=np.concatenate((train_features,valid_features),axis=0)
y=np.concatenate((train_labels,valid_labels))
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(test_features)
class_weights=calculate_class_weights(y)

log_reg = LogisticRegression(solver='liblinear', max_iter=100000,class_weight=class_weights,random_state=242)
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X, y)
# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

optim_LR = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver='liblinear', max_iter=10000, class_weight=class_weights, random_state=82)


optim_LR.fit(X, y)
y_pred = optim_LR.predict(X_test)
y_prob = optim_LR.predict_proba(X_test)[:, 1]
print_metrics(test_labels,y_pred,y_prob)