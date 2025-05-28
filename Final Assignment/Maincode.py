from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df= pd.read_csv("/content/drive/MyDrive/Coding/final_avian_dataset.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load your final dataset


# Ensure 'High_Risk' column exists
df['High_Risk'] = (df['Total_Cases'] >= 10).astype(int)

# Set style
sns.set(style="whitegrid")

# 1. Total Cases by Region
plt.figure(figsize=(10, 5))
region_cases = df.groupby('Region')['Total_Cases'].sum().sort_values(ascending=False)
region_cases.plot(kind='bar')
plt.title('Total Cases by Region')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("total_cases_by_region.png")
plt.show()
plt.close()

# 2. Boxplot - Total Cases by Diagnosis
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Diagnosis', y='Total_Cases')
plt.title('Total Cases Distribution by Diagnosis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_total_cases_by_diagnosis.png")
plt.show()
plt.close()

# 3. Monthly Trend of Outbreaks
plt.figure(figsize=(8, 5))
monthly_trend = df.groupby('Month')['Total_Cases'].sum()
monthly_trend.plot(marker='o')
plt.title('Monthly Outbreak Trend')
plt.xlabel('Month')
plt.ylabel('Total Cases')
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_trend_outbreaks.png")
plt.show()
plt.close()

# 4. Correlation Heatmap of All Numeric Features
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of All Numeric Features')
plt.tight_layout()
plt.savefig("correlation_heatmap_all_features.png")
plt.show()
plt.close()

# 5. High-Risk vs Low-Risk Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='High_Risk')
plt.title('Distribution of High-Risk vs Low-Risk Outbreaks')
plt.xticks([0, 1], ['Low Risk', 'High Risk'])
plt.tight_layout()
plt.savefig("high_risk_distribution.png")
plt.show()
plt.close()

# CLuste analysis 
# === Apply KM# Select features
categorical = ['Region', 'Diagnosis']
numerical = [
    'Total_Cases', 'Month', 'Avg_Temperature', 'Humidity',
    'Migration_Level', 'Biosecurity_Score', 'Vaccine_Rate', 'Poultry_Population_000s'
]

df = df[categorical + numerical].dropna()

# Pipeline for preprocessing + PCA
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2))
])

X_pca = pipeline.fit_transform(df)

# === Elbow Method ===
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.axvline(x=4, linestyle='--', color='red', label='Suggested K = 4')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.tight_layout()
plt.show()eans with optimal K ===
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init='auto')
clusters = kmeans_final.fit_predict(X_pca)

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=60)
plt.title("Clusters of Avian Disease Profiles (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Naive Bayes Classifier Model
df['High_Risk'] = (df['Total_Cases'] >= 10).astype(int)

# === 1. Train-Test Split ===
features = [
    'Vaccine_Rate',
    'Poultry_Population_000s',
    'Biosecurity_Score',
    'Month',
    'Migration_Level'
]
X = df[features]
y = df['High_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Train Naive Bayes Model ===
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# === 3. Evaluation Metrics ===
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# === 4. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Naive Bayes Classifier")
plt.grid(False)
plt.tight_layout()
plt.show()

# === 5. Feature Importance (using absolute log probability means) ===
importances = np.abs(nb_model.theta_[1] - nb_model.theta_[0])  # Difference in means between classes
feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance (abs diff in mean)': importances
}).sort_values(by='Importance (abs diff in mean)', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance (abs diff in mean)', y='Feature', data=feat_imp_df, hue='Feature', dodge=False, palette='coolwarm', legend=False)

plt.title("Naive Bayes Feature Importance (Approximate)")
plt.tight_layout()
plt.show()

# === 6. Decision Boundary (2D) ===
X_vis = df[['Poultry_Population_000s', 'Vaccine_Rate']]
y_vis = df['High_Risk']
nb_vis = GaussianNB()
nb_vis.fit(X_vis, y_vis)

x_min, x_max = X_vis['Poultry_Population_000s'].min() - 50, X_vis['Poultry_Population_000s'].max() + 50
y_min, y_max = X_vis['Vaccine_Rate'].min() - 5, X_vis['Vaccine_Rate'].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Poultry_Population_000s', 'Vaccine_Rate'])
Z = nb_vis.predict(grid_df).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap='Pastel2', alpha=0.6)
sns.scatterplot(x=X_vis['Poultry_Population_000s'], y=X_vis['Vaccine_Rate'], hue=y_vis, palette='Set1', edgecolor='k')
plt.title("Naive Bayes Classifier Decision Boundary")
plt.xlabel("Poultry Population (000s)")
plt.ylabel("Vaccine Rate (%)")
plt.legend(title="High Risk")
plt.tight_layout()
plt.show()
