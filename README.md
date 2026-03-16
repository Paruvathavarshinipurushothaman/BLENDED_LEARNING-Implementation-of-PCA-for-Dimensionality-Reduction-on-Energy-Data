# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect the energy dataset containing multiple features related to energy consumption or production.
2. Preprocess the data by handling missing values and standardizing the features.
3. Compute the covariance matrix of the standardized data to understand feature relationships.
4. Calculate eigenvalues and eigenvectors and select the principal components with the highest variance.
5. Transform the original dataset into the reduced-dimensional space using the selected principal components.


## Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())

# Select features
X = data[['Height(Inches)', 'Weight(Pounds)']]

# Plot original data
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create PCA dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot PCA results
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="914" height="754" alt="image" src="https://github.com/user-attachments/assets/831adcc3-9c62-4677-beaf-d6f8401e83ff" />
<img width="836" height="630" alt="image" src="https://github.com/user-attachments/assets/c7943cae-6863-4452-a9d3-036dc1da7c08" />
## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
