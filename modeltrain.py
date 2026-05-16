
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Machine Learning Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 2. GRAPH STYLE


plt.style.use('ggplot')

# 3. LOAD DATASET


df = pd.read_csv("airquality.csv")

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")
print(df.head())

# ==============================
# 4. DATA CLEANING
# ==============================

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n==============================")
print("MISSING VALUES")
print("==============================")
print(df.isnull().sum())

# ==============================
# 5. CREATE TARGET VARIABLE
# ==============================

# 0 = Safe Air
# 1 = Risky Air

def label_risk(row):

    if (
        row['AQI'] > 150 or
        row['PM2.5'] > 120 or
        row['NO2'] > 50
    ):
        return 1

    return 0


df['Risk'] = df.apply(label_risk, axis=1)

print("\n==============================")
print("TARGET DISTRIBUTION")
print("==============================")
print(df['Risk'].value_counts())

# ==============================
# 6. FEATURES & TARGET
# ==============================

feature_names = ['AQI', 'PM2.5', 'NO2']

X = df[feature_names]
y = df['Risk']

# ==============================
# 7. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 8. FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ==============================
# 9. PCA FOR VISUALIZATION
# ==============================

pca = PCA(n_components=2)

X_vis = pca.fit_transform(X_test_scaled)

# ==============================
# 10. MACHINE LEARNING MODELS
# ==============================

models = {

    "SVM": SVC(
        kernel='rbf',
        C=1,
        gamma='scale'
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

# ==============================
# 11. TRAINING & EVALUATION
# ==============================

metrics = {}

best_model = None
best_accuracy = 0

for name, model in models.items():

    print("\n====================================")
    print(f"{name} TRAINING")
    print("====================================")

    # Train Model
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store Results
    metrics[name] = {
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1 Score": f1,
        "Prediction": y_pred
    }

    # Print Metrics
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Best Model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ==============================
# 12. SAVE BEST MODEL
# ==============================

joblib.dump(best_model, "air_quality_model.pkl")

print("\n====================================")
print("BEST MODEL SAVED SUCCESSFULLY")
print("====================================")

# ==============================
# 13. PERFORMANCE COMPARISON
# ==============================

labels = list(metrics.keys())

accuracy_values = [metrics[m]["Accuracy"] for m in labels]
precision_values = [metrics[m]["Precision"] for m in labels]
recall_values = [metrics[m]["Recall"] for m in labels]
f1_values = [metrics[m]["F1 Score"] for m in labels]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

ax.bar(x - 0.3, accuracy_values, width, label='Accuracy')
ax.bar(x - 0.1, precision_values, width, label='Precision')
ax.bar(x + 0.1, recall_values, width, label='Recall')
ax.bar(x + 0.3, f1_values, width, label='F1 Score')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)

ax.set_ylim(0, 1.1)

ax.set_title(
    "MODEL PERFORMANCE COMPARISON",
    fontsize=13,
    fontweight='bold'
)

ax.set_xlabel("Models", fontsize=10)
ax.set_ylabel("Scores", fontsize=10)

ax.legend(fontsize=8)

ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.show()

# ==============================
# 14. ALL CONFUSION MATRICES
# ==============================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10, 6),
    dpi=120
)

axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):

    y_pred = metrics[name]["Prediction"]

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Risky"]
    )

    disp.plot(
        ax=axes[idx],
        colorbar=False,
        values_format='d'
    )

    axes[idx].set_title(
        name,
        fontsize=9,
        fontweight='bold'
    )

    axes[idx].tick_params(labelsize=8)

# Remove last empty subplot
fig.delaxes(axes[-1])

plt.suptitle(
    "CONFUSION MATRICES OF ALL MODELS",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# ==============================
# 15. PCA VISUALIZATION
# ==============================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10, 6),
    dpi=120
)

axes = axes.flatten()

for idx, name in enumerate(models.keys()):

    y_pred = metrics[name]["Prediction"]

    scatter = axes[idx].scatter(
        X_vis[:, 0],
        X_vis[:, 1],
        c=y_pred,
        cmap='coolwarm',
        s=25,
        alpha=0.8,
        edgecolors='black'
    )

    axes[idx].set_title(
        name,
        fontsize=9,
        fontweight='bold'
    )

    axes[idx].set_xlabel(
        "PCA 1",
        fontsize=8
    )

    axes[idx].set_ylabel(
        "PCA 2",
        fontsize=8
    )

    axes[idx].tick_params(labelsize=7)

# Remove last empty subplot
fig.delaxes(axes[-1])

plt.suptitle(
    "PCA VISUALIZATION OF ALL MODELS",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# ==============================
# 16. USER PREDICTION SYSTEM
# ==============================

print("\n====================================")
print("AIR QUALITY PREDICTION SYSTEM")
print("====================================")

# User Input
aqi = float(input("Enter AQI Value   : "))
pm25 = float(input("Enter PM2.5 Value : "))
no2 = float(input("Enter NO2 Value   : "))

# Create DataFrame
user_data = pd.DataFrame(
    [[aqi, pm25, no2]],
    columns=feature_names
)

# Scale User Input
user_data_scaled = scaler.transform(user_data)

# Prediction
prediction = best_model.predict(user_data_scaled)

print("\n====================================")

if prediction[0] == 1:
    print("⚠️ Air Quality is RISKY")
else:
    print("✅ Air Quality is SAFE")

print("====================================")