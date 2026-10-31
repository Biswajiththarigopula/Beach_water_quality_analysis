import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("------Beach Turbidity Level Prediction------")
#Official data.gov / Chicago dataset. This does NOT predict swimming safety.
#Keep this CSV in the same folder and run the script from that folder.
df = pd.read_csv("Beach_Water_Quality_Automated_Sensors.csv")

#Exploratory Data Analysis (same order as the EV project)
print("------First 5 Rows------")
print(df.head())
print("------Last 5 Rows------")
print(df.tail())
print("------Shape------")
print(df.shape)
print("------Info------")
df.info()
print("------Summary------")
print(df.describe())
print("------Missing Values------")
print(df.isnull().sum())

#Data Preprocessing
#The file contains -100000 readings; treat these invalid sensor values as missing.
numeric_cols = ["Water Temperature", "Turbidity", "Transducer Depth",
                "Wave Height", "Wave Period", "Battery Life"]
df[numeric_cols] = df[numeric_cols].replace(-100000, np.nan)
df["Measurement Timestamp"] = pd.to_datetime(
    df["Measurement Timestamp"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
#Depth is mostly missing and is not a model feature: do not drop every row for it.
required = ["Measurement Timestamp", "Water Temperature", "Turbidity",
            "Wave Height", "Wave Period"]
df = df.dropna(subset=required)
df = df.drop_duplicates()
#Turbidity and wave magnitudes cannot be negative.
df = df[(df["Turbidity"] >= 0) & (df["Wave Height"] >= 0) & (df["Wave Period"] >= 0)]
df = df.sort_values(["Measurement Timestamp", "Beach Name"]).reset_index(drop=True)
df["Month"] = df["Measurement Timestamp"].dt.month
df["Hour"] = df["Measurement Timestamp"].dt.hour
print("Shape after cleaning:", df.shape)
print("Date range:", df["Measurement Timestamp"].min(), df["Measurement Timestamp"].max())

#Correlation and covariance (descriptive, not proof of causation)
print("------Correlation Matrix------")
print(df.select_dtypes(include="number").corr())
print("------Covariance Matrix------")
print(df.select_dtypes(include="number").cov())

#IQR outlier check (inspect only, do not remove legitimate high turbidity)
Q1 = df["Turbidity"].quantile(0.25)
Q3 = df["Turbidity"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Turbidity"] < lower_bound) | (df["Turbidity"] > upper_bound)]
print("------Turbidity IQR------")
print("Q1:", Q1, "Q3:", Q3, "IQR:", IQR)
print("Bounds:", lower_bound, upper_bound, "Outliers:", len(outliers))
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Turbidity"])
plt.title("Turbidity Outlier Inspection")
plt.tight_layout()
plt.show()

#Same five-feature structure; turbidity itself MUST NOT be an input.
X = df[["Water Temperature", "Wave Height", "Wave Period", "Month", "Hour"]]

#Earlier timestamps train; later timestamps test (not a random hourly split).
#All beaches at the boundary timestamp belong to the test set.
cutoff = df["Measurement Timestamp"].iloc[int(len(df) * 0.8)]
train_rows = df["Measurement Timestamp"] < cutoff
test_rows = df["Measurement Timestamp"] >= cutoff
X_train = X[train_rows]
X_test = X[test_rows]

#No native safety/category label exists. Define a RELATIVE turbidity level.
#Threshold comes from TRAINING data only, not a health/safety standard.
threshold = df.loc[train_rows, "Turbidity"].median()
df["Target"] = (df["Turbidity"] > threshold).astype(int)
df["Turbidity Level"] = df["Target"].replace({0: "Lower", 1: "Higher"})
y_train = df.loc[train_rows, "Target"]
y_test = df.loc[test_rows, "Target"]
print("Training-median threshold:", threshold)
print("Lower = turbidity <= threshold; Higher = turbidity > threshold")
print("NOT a safe/unsafe threshold")
print("Test period begins:", cutoff)
print("Training rows:", len(y_train), "Test rows:", len(y_test))
print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

#Visualization
plt.figure(figsize=(7, 5))
sns.countplot(x="Turbidity Level", data=df)
plt.title("Relative Turbidity Level Counts")
plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 5))
df["Beach Name"].value_counts().plot(kind="barh")
plt.title("Readings by Beach")
plt.xlabel("Number of Readings")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Water Temperature", hue="Turbidity Level", bins=30)
plt.title("Water Temperature by Relative Turbidity Level")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Water Temperature", "Turbidity", "Wave Height", "Wave Period", "Month", "Hour"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#Feature scaling fitted on training data only
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Model 1: Linear Regression (numeric baseline, not a classifier)
print("------Linear Regression------")
linear = LinearRegression()
linear.fit(X_train, y_train)
y_score_linear = linear.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_score_linear))
print("MSE:", mean_squared_error(y_test, y_score_linear))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_score_linear)))
print("R-squared:", r2_score(y_test, y_score_linear))
#For comparison only: scores below 0.5 are Lower; others are Higher.
#These scores are NOT probabilities and can be outside 0 to 1.
y_pred_linear = (y_score_linear >= 0.5).astype(int)
print("Thresholded accuracy:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear, zero_division=0))

#Model 2: Logistic Regression
print("------Logistic Regression------")
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred_logistic = logistic.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic, zero_division=0))

#Model 3: Decision Tree
print("------Decision Tree------")
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=0))

#Model 4: K-Nearest Neighbors
print("------K-Nearest Neighbors (KNN)------")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, zero_division=0))

#Model 5: K-Means Clustering (unsupervised, not a classifier)
print("------K-Means Clustering------")
km = KMeans(n_clusters=2, n_init=10, random_state=42)
km.fit(X_train)  #No turbidity-level labels used to fit the clusters.
train_clusters = km.labels_
test_clusters = km.predict(X_test)
print("Training inertia:", km.inertia_)
print("Test silhouette:", silhouette_score(X_test, test_clusters, sample_size=2000, random_state=42))
print("Test adjusted Rand index:", adjusted_rand_score(y_test, test_clusters))
print("Training cluster composition:\n", pd.crosstab(train_clusters, y_train.to_numpy()))
#Use TRAINING labels only to name each cluster by its majority turbidity level.
#Never assume cluster 0 means Lower or cluster 1 means Higher.
label0 = y_train[train_clusters == 0].mode().iloc[0]
label1 = y_train[train_clusters == 1].mode().iloc[0]
y_pred_km = np.where(test_clusters == 0, label0, label1)
print("Cluster 0 majority label:", label0, "Cluster 1 majority label:", label1)
print("Mapped accuracy:", accuracy_score(y_test, y_pred_km))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_km))
print(classification_report(y_test, y_pred_km, zero_division=0))

#Model Comparison (Higher = positive class)
print("------Model Comparison------")
names = ["Linear Regression", "Logistic Regression", "Decision Tree", "KNN", "K-Means"]
preds = [y_pred_linear, y_pred_logistic, y_pred_dt, y_pred_knn, y_pred_km]
accuracy = []
precision = []
recall = []
f1 = []
for p in preds:
    accuracy.append(accuracy_score(y_test, p))
    precision.append(precision_score(y_test, p, zero_division=0))
    recall.append(recall_score(y_test, p, zero_division=0))
    f1.append(f1_score(y_test, p, zero_division=0))
result = pd.DataFrame({"Model": names, "Accuracy": accuracy, "Precision": precision,
                       "Recall": recall, "F1 Score": f1})
print(result.round(4))
result.to_csv("Beach_Model_Comparison.csv", index=False)
print("Linear Regression is thresholded; K-Means is training-majority mapped.")
print("They are reference baselines, not standard supervised classifiers.")
baseline_label = y_train.mode().iloc[0]
print("Training-majority baseline label:", baseline_label)
print("Training-majority baseline test accuracy:", (y_test == baseline_label).mean())
#Both constant predictions are diagnostic baselines, not additional models.
print("Always-Lower accuracy:", (y_test == 0).mean())
print("Always-Higher accuracy:", (y_test == 1).mean())
print("Always-Higher F1:", f1_score(y_test, np.ones(len(y_test), dtype=int)))

#Accuracy comparison
plt.figure(figsize=(9, 5))
plt.bar(names, accuracy)
plt.title("Test Accuracy - Five Methods")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("Beach_Accuracy_Comparison.png", dpi=160)
plt.show()

#F1 comparison
plt.figure(figsize=(9, 5))
plt.bar(names, f1)
plt.title("Higher F1 Score - Five Methods")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("Beach_F1_Comparison.png", dpi=160)
plt.show()

#Confusion matrix for each method
for name, p in zip(names, preds):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, p), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Lower", "Higher"], yticklabels=["Lower", "Higher"])
    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

#K-Means illustration (the fit uses all five features)
plt.figure(figsize=(7, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_clusters, cmap="viridis", s=15, alpha=0.5)
plt.xlabel("Scaled Water Temperature")
plt.ylabel("Scaled Wave Height")
plt.title("K-Means Test Clusters (Two-Feature View)")
plt.tight_layout()
plt.savefig("Beach_KMeans_Clusters.png", dpi=160)
plt.show()

#Best method for detecting Higher turbidity: compare positive-class F1.
best = result.sort_values(by="F1 Score", ascending=False).iloc[0]
print("------Best Model------")
print("Best method for turbidity-level prediction:", best["Model"])
print("Higher F1:", round(best["F1 Score"], 4), "Accuracy:", round(best["Accuracy"], 4))
best_accuracy = result.sort_values(by="Accuracy", ascending=False).iloc[0]
print("Highest accuracy method:", best_accuracy["Model"], best_accuracy["Accuracy"])
print("F1 winner means best among these five, NOT necessarily better than constant predictions.")
print("This conclusion is limited to these sensors and this chronological test period.")
print("Higher/lower turbidity does NOT indicate whether swimming is safe.")
print("------END------")
