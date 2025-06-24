import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load your dataset
df = pd.read_csv("your_dataset.csv")  # <- Replace with your file name

# ----------------- Step 1: Summary Statistics -----------------

print("\n--- Data Info ---")
print(df.info())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# ----------------- Step 2: Histograms and Boxplots -----------------

print("\nGenerating histograms...")
df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

print("\nGenerating boxplots...")
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.xticks(rotation=90)
plt.title("Boxplots of Numeric Features")
plt.show()

# ----------------- Step 3: Correlation Matrix -----------------

print("\nGenerating correlation heatmap...")
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ----------------- Step 4: Pairplot (if not too many features) -----------------

print("\nGenerating pairplot...")
sns.pairplot(df.select_dtypes(include='number'))
plt.show()

# ----------------- Step 5: Detecting Outliers -----------------

print("\nChecking for outliers...")
for col in df.select_dtypes(include='number').columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Check - {col}")
    plt.show()

# ----------------- Optional: Plotly Interactive Plot -----------------

print("\nGenerating interactive scatter matrix (Plotly)...")
fig = px.scatter_matrix(df.select_dtypes(include='number'))
fig.update_layout(title='Scatter Matrix', width=1000, height=800)
fig.show()
