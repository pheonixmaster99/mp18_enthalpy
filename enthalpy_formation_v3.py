import json
import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Load the data from the .json file with error handling
try:
    with open('mp_e_form.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print("The JSON file was not found.")
    exit()

df = pd.DataFrame(data['data'], columns=data['columns'])
# print(df.head())

def extract_features(structure):
    """
    Extracts relevant features from a crystal structure.

    Args:
        structure: A dictionary representing the crystal structure.

    Returns:
        A dictionary containing the extracted features.
    """

    # Lattice Features
    lattice = structure['lattice']
    features = {
        'a': lattice['a'],
        'b': lattice['b'],
        'c': lattice['c'],
        'alpha': lattice['alpha'],
        'beta': lattice['beta'],
        'gamma': lattice['gamma'],
        'volume': lattice['volume']
    }

    # Elemental Composition
    elements = [site['species'][0]['element'] for site in structure['sites']]
    composition = Counter(elements)
    features.update({f'count_{element}': count for element, count in composition.items()})

    # Atomic Distances
    positions = np.array([site['xyz'] for site in structure['sites']])
    distances = pdist(positions)
    if distances.size > 0:
        features.update({
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        })
    else:
        features.update({
            'mean_distance': np.nan,
            'std_distance': np.nan,
            'min_distance': np.nan,
            'max_distance': np.nan
        })

    # Magnetic Moment (if available)
    magmom = [site['properties'].get('magmom', 0) for site in structure['sites']]
    features['mean_magmom'] = np.mean(magmom)

    return features

# Extract features for the dataset
features_list = [extract_features(struct) for struct in df['structure']]
features_df = pd.DataFrame(features_list)

# Add formation enthalpy as a target column
features_df['formation_enthalpy'] = df['e_form']

# Fill NaN values with the mean of their respective columns
features_df = features_df.fillna(features_df.mean())

# Function to drop highly correlated features
def drop_highly_correlated_features(df, threshold=0.6):
    """
    Removes features with a correlation above the specified threshold.

    Args:
        df: The DataFrame containing the features.
        threshold: The correlation threshold above which features are dropped.

    Returns:
        The DataFrame with highly correlated features removed.
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return df.drop(columns=to_drop)

# Apply the function to drop highly correlated features
features_df_reduced = drop_highly_correlated_features(features_df, threshold=0.6)

# Standardize the features before dimensionality reduction
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df_reduced.drop(columns='formation_enthalpy'))

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['formation_enthalpy'] = features_df_reduced['formation_enthalpy']

# Apply t-SNE for non-linear dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(tsne_components, columns=['tSNE1', 'tSNE2'])
tsne_df['formation_enthalpy'] = features_df_reduced['formation_enthalpy']

# Fit a Random Forest Regressor to determine feature importance
model = RandomForestRegressor()
X = features_df_reduced.drop(columns='formation_enthalpy')
y = features_df_reduced['formation_enthalpy']
model.fit(X, y)

# Get feature importances and plot them
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Create a figure for both PCA and t-SNE plots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# PCA Plot
pca_scatter = sns.scatterplot(ax=axs[0], data=pca_df, x='PC1', y='PC2', hue='formation_enthalpy', palette='viridis')
axs[0].set_title('PCA of Scaled Feature Set')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
fig.colorbar(pca_scatter.collections[0], ax=axs[0], label='formation enthalpy')  # Use collections[0] for colorbar

# t-SNE Plot 
tsne_scatter = sns.scatterplot(ax=axs[1], data=tsne_df, x='tSNE1', y='tSNE2', hue='formation_enthalpy', palette='viridis')
axs[1].set_title('t-SNE of Reduced Feature Set')
axs[1].set_xlabel('t-SNE Component 1')
axs[1].set_ylabel('t-SNE Component 2')
fig.colorbar(tsne_scatter.collections[0], ax=axs[1], label='formation enthalpy') 

# Save the PCA and t-SNE plots
plt.savefig('pca_tsne_plots.png') 
plt.close(fig) # Close the figure to avoid overlapping plots

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.savefig('feature_importance.png')
plt.close() # Close the figure

# Apply k-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the pca_df dataframe
pca_df['cluster'] = kmeans_labels 

# Plot the clusters using PCA for simplicity
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title('K-Means Clustering on PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Save the k-means plot
plt.savefig('kmeans_plot.png')
plt.close() # Close the figure

# Show all plots (if desired)
# plt.show()