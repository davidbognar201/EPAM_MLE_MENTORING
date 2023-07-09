import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

processed_df = pd.read_csv("data/processedData.csv")
print("------------------- Running the training script -------------------")

x_train, x_test, y_train, y_test = train_test_split(processed_df["review_stemmed_lemmatized"],processed_df["target"],
                                                    random_state=23,
                                                    test_size=0.2,
                                                    stratify=processed_df["target"])

tfidf = TfidfVectorizer()
x_train_vectorized = tfidf.fit_transform(x_train)
x_train_vectorized.shape

x_test_vectorized = tfidf.transform(x_test)
x_test_vectorized.shape

# Possible parameters for the Stochastic Gradient Descent classifier
parameters_SGD = {
    #'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'alpha': [1e-4, 1e-3],
    'penalty': ['l2', 'l1'],
    'n_jobs': [-1],
    #'loss': ["log", "huber", "modified_huber", "hinge"]
    'loss': ["modified_huber"]
}

grid_search_SGD = GridSearchCV(SGDClassifier(), parameters_SGD, scoring="roc_auc", cv=5)
grid_search_SGD.fit(x_train_vectorized, y_train)

SGD_predicted_y = grid_search_SGD.predict(x_test_vectorized)
SGD_predicted_y

print(classification_report(y_test,SGD_predicted_y))

with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": accuracy_score(y_test,SGD_predicted_y),
                    "recall": recall_score(y_test,SGD_predicted_y), 
                    "precision":precision_score(y_test,SGD_predicted_y), 
                    "auc":roc_auc_score(y_test,SGD_predicted_y)}, outfile)
        
print("------------------- Training script finished -------------------")
