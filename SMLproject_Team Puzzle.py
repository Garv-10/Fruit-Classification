import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load the training and testing datasets
data = pd.read_csv("train.csv")
test_data = pd.read_csv('test.csv')

# Separate the target variable
X = data.drop(["ID", "category"], axis=1)
y = data["category"]
X_test = test_data.drop(["ID"], axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA
pca = PCA(n_components=375)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)
print("PCA done")

# Perform LDA
lda = LDA(n_components=19)
X_train = lda.fit_transform(X_train, y_train)
X_val = lda.transform(X_val)
X_test = lda.transform(X_test)
print("LDA done")

# Perfrom clustering
kmeans = KMeans(n_clusters=4, random_state=42,
                max_iter=40000, algorithm="elkan")
X_train_clusters = kmeans.fit_predict(X_train)
X_train = np.column_stack((X_train, X_train_clusters))
X_val_clusters = kmeans.predict(X_val)
X_val = np.column_stack((X_val, X_val_clusters))
X_test_clusters = kmeans.predict(X_test)
X_test = np.column_stack((X_test, X_test_clusters))
print("KMeans done")

# remove outliers using LOF
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
yhat = lof.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
print("LOF done")

# Train the model using adaboost
model1 = BaggingClassifier(LogisticRegression(C=1, max_iter=40000, penalty='l2', solver='newton-cg'), n_estimators=100, random_state=42)
model2 = AdaBoostClassifier(LogisticRegression(C=1, max_iter=40000, penalty='l2', solver='newton-cg'), n_estimators=100, random_state=42)
model3 = VotingClassifier(estimators=[('lr', LogisticRegression(C=0.1, max_iter=40000, penalty='l2', solver='newton-cg')),('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))],  voting='hard')
model4 = LogisticRegression(C=1, max_iter=40000, penalty='l2', solver='newton-cg')
model5 = KNeighborsClassifier(n_neighbors=10)
model6 = DecisionTreeClassifier(max_depth=10, random_state=42)
model7 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

models = [model3, model1, model2, model4, model5, model6, model7]
best_model = None
best_accuracy = 0.0
for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model = model


best_model.fit(X_train, y_train)
predictions = best_model.predict(X_val)
# Calculate the accuracy score
score = accuracy_score(y_val, predictions)
print("Validation Accuracy: ", score)

# Make predictions on the test data using trained model
predictions = model.predict(X_test)
predictions = label_encoder.inverse_transform(predictions)

# Save the predictions to a csv file
submission = pd.DataFrame(predictions, columns=['Category'])
submission['ID'] = test_data['ID']
submission = submission[['ID', 'Category']]
submission.to_csv('submission.csv', index=False)