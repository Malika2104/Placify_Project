import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Load the Parkinson's dataset
data = pd.read_csv(r"C:\Placify_Assignment2\parkinsons_large.csv")

# Inspect dataset
print(data.head())

# Split data into features (X) and target (y)
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)
lr_model = LogisticRegression()

# Fit models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Calculate probabilities for ROC-AUC
rf_prob = rf_model.predict_proba(X_test)[:, 1]
svm_prob = svm_model.predict_proba(X_test)[:, 1]
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Evaluate the models
print("Random Forest:")
evaluate_model(y_test, rf_pred, rf_prob)

print("\nSupport Vector Machine:")
evaluate_model(y_test, svm_pred, svm_prob)

print("\nLogistic Regression:")
evaluate_model(y_test, lr_pred, lr_prob)

# Build a simple deep neural network
dl_model = Sequential()
dl_model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
dl_model.add(Dropout(0.3))
dl_model.add(Dense(64, activation='relu'))
dl_model.add(Dropout(0.3))
dl_model.add(Dense(1, activation='sigmoid'))

# Compile the model
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
dl_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate on test data
dl_pred_prob = dl_model.predict(X_test)
dl_pred = (dl_pred_prob > 0.5).astype(int)

# Evaluate the deep learning model
print("\nDeep Learning Model:")
evaluate_model(y_test, dl_pred, dl_pred_prob)
