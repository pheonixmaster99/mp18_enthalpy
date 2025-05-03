# mp18_enthalpy
This project explores the MP18 formation enthalpy dataset from Matbench using a custom pipeline for feature extraction, correlation analysis, dimensionality reduction, and clustering.

✅ What This Project Does:
Parses and processes the dataset in JSON format using pandas and Python.

Extracts structural features such as lattice parameters, atomic distances, and elemental counts.

Applies basic feature engineering to handle missing values and standardize input.

Performs feature correlation filtering to remove highly correlated features.

Uses PCA and t-SNE to visualize the feature space and explore separation by formation enthalpy.

Applies Random Forest Regressor to estimate feature importance.

Adds KMeans clustering to explore natural groupings in the data.

🛠 What I Learned:
How to extract and engineer features from structural data.

Techniques for visualizing high-dimensional data using PCA and t-SNE.

The importance of preprocessing and correlation analysis before ML modeling.

Basics of clustering and model interpretation with Random Forests.

📌 Notes:
The feature set currently relies on raw lattice parameters and composition, which provides a foundational baseline. Further work can include advanced featurization with matminer for improved model performance.

No train/test split or hyperparameter tuning was included yet — this would be a key next step to evaluate generalization.

Future updates may include using Matminer’s featurizers like ElementProperty or StructuralFeaturizer and testing performance with different ML models.

