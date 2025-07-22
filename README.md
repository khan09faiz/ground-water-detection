# Identifying Groundwater Potential Zones Machine Learning

This repository contains the project "Identifying Groundwater Potential Zone," submitted in partial fulfillment for the award of the Degree of B. Tech in Computer Science and Engineering (Artificial Intelligence and Machine Learning) at Manipal University Jaipur (2024-2025).

---

## 📋 Abstract

Groundwater is a critical resource in arid regions like Jodhpur, India. Due to over-extraction and climate change, its sustainable management has become imperative. This research presents a machine learning-driven approach to delineate Groundwater Potential Zones (GWPZ) across the administrative boundary of Jodhpur. Using **52 ground-truth locations** (34 well sites and 18 non-well sites) and **15 influential geospatial and hydrogeological parameters**, we trained and evaluated nine supervised classification algorithms. Our findings show that ensemble models, particularly **Random Forest (RF)** and **Logistic Regression (LR)**, offer superior accuracy in identifying GWPZs. A feature importance analysis highlighted that **Rainfall, NDVI, and Proximity to Waterbodies** are the most significant factors influencing groundwater availability. The final delineated map classifies the region into five zones—*Very Good, Good, Moderate, Poor, and Very Poor*—providing crucial insights for sustainable water resource management and policymaking.

---

## ✨ Key Features

* **Study Area**: Focused on the arid and semi-arid administrative boundary of Jodhpur, Rajasthan, India.
* **Comprehensive Dataset**: Utilizes 15 high-resolution geospatial predictors, including rainfall, NDVI, lithology, soil characteristics, and topographical data.
* **Multi-Model Comparison**: Implements and evaluates nine machine learning models:
    * Random Forest (RF)
    * Gradient Boosting (GB)
    * AdaBoost
    * Decision Tree (DT)
    * Logistic Regression (LR)
    * Support Vector Classifier (SVC)
    * Naïve Bayes (NB)
    * Multilayer Perceptron (MLP)
    * K-Nearest Neighbours (KNN)
* **Feature Importance**: Identifies the most influential factors for groundwater potential.
* **Actionable Insights**: Produces a spatially explicit GWPZ map to guide water exploration and policy.

---

## 🌍 Study Area & Data

The research focuses on Jodhpur, a region characterized by arid climatic conditions.A geospatial dataset with a 30m resolution was created using QGIS 3.34 for data preprocessing, reprojection, and sampling.


<br>

### **Influencing Factors (Thematic Layers)**

Fifteen thematic layers were used to train the models, representing key topographic, hydrological, geological, and biotic factors.

1.  **Rainfall**: Primary source of groundwater recharge.
2.  **Proximity to Water Bodies (DFR)**: Areas closer to rivers have higher recharge potential.
3.  **Drainage Density (DD)**: Lower density indicates greater recharge capacity.
4.  **Lithology**: Physical rock characteristics influencing porosity and permeability.
5.  **Geomorphology**: Landforms that control water flow and storage.
6.  **Topographical Wetness Index (TWI)**: Quantifies the potential for water accumulation.
7.  **Soil Depth**: Deeper soils offer more storage and enhance percolation.
8.  **Soil Slope**: Influences lateral water movement and infiltration rates.
9.  **Soil Texture**: Determines the proportion of sand, silt, and clay, affecting infiltration.
10. **Forest Cover**: Promotes infiltration and reduces surface runoff.
11. **Hill Shade**: Derived from the DEM to understand topography.
12. **Normalized Difference Vegetation Index (NDVI)**: Indicates vegetation density, which affects infiltration.
13. **Altitude (Elevation/DEM)**: Affects surface runoff and gravitational potential for recharge.
14. **Land Use/Land Cover (LULC)**: Categorizes human impact on groundwater systems.
15. **Slope**: Steepness of the ground surface, controlling runoff vs. infiltration.

---

## ⚙️ Technology and Code Explained 💻

This project leverages a combination of geospatial analysis tools and Python-based machine learning libraries to process data and build predictive models.

### **Technology Stack**

* **QGIS 3.34**: An open-source Geographic Information System (GIS) used for all spatial data preprocessing, including re-projecting, clipping, and raster analysis to derive factors like slope, drainage density, and TWI.
* **Python**: The core programming language for data analysis and machine learning.
* **Scikit-learn**: A fundamental library for implementing machine learning models, including RandomForest, GradientBoosting, AdaBoost, and for performing evaluations like train-test splits and cross-validation.
* **Pandas**: Used for data manipulation and analysis, primarily for handling the structured dataset extracted from the geospatial layers.
* **NumPy**: A library for numerical operations that underpins many of the data science libraries used.
* **Matplotlib & Seaborn**: Used for data visualization, including plotting feature importance, confusion matrices, and ROC curves.

### **Code Walkthrough (`ground water.ipynb`)**

The Jupyter Notebook provided contains the end-to-end workflow for training and evaluating the machine learning models.

1.  **Library Imports**: The first cell imports all necessary libraries for model building, data handling, and visualization.

2.  **Data Loading and Cleaning**: The dataset is loaded from a CSV file (`gw 34.csv`) into a Pandas DataFrame.The code checks for missing values using `.isna().sum()` and handles them using spline interpolation (`.interpolate()`), which is a method for estimating missing data points by fitting a smooth curve.

3.  **Data Splitting**: The dataset is split into features (`X`) and the target variable (`y`, 'Potential')[cite: 38]. [cite_start]`train_test_split` from Scikit-learn is used to create training and testing sets.The `stratify=y` argument is crucial as it ensures that the proportion of well sites to non-well sites is the same in both the training and testing sets, which is important for imbalanced datasets.

4. **Feature Scaling**: A `StandardScaler` is initialized to standardize the features by removing the mean and scaling to unit variance. This step is important for models that are sensitive to the scale of input features, such as SVC, Logistic Regression, and MLP.

5.  **Model Training and Evaluation**: Several classification models are trained on the data:
    * **Random Forest**: An ensemble model that builds multiple decision trees and merges them to get a more accurate and stable prediction.
    * **Gradient Boosting & AdaBoost**: Both are boosting algorithms that build models sequentially, where each new model corrects the errors of the previous one.
    * **Decision Tree**: A single tree model that makes decisions based on feature splits.
    * **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies a data point based on the majority class of its 'k' nearest neighbors.
    * **Logistic Regression**: A linear model used for binary classification.
    * **Support Vector Classifier (SVC)**: Finds the hyperplane that best separates the two classes in the feature space.
    * **Naïve Bayes (GaussianNB, BernoulliNB)**: A probabilistic classifier based on Bayes' theorem.
    * **Multilayer Perceptron (MLP)**: A feedforward artificial neural network.

6.  **Feature Importance Analysis**: After training the Random Forest classifier, the `.feature_importances_` attribute is used to determine which factors had the most significant impact on the model's predictions.These importances are then visualized using a horizontal bar chart, providing clear insights into the key drivers of groundwater potential.

7. **Performance Visualization**: The notebook generates confusion matrices and ROC curves for the models to visually assess their performance, showing how well they distinguish between "Potential" and "No Potential" zones.

***

## **Results 📊**

The models were trained on a 60-40 split of the ground-truth data with stratified sampling. **Logistic Regression** and **Random Forest** demonstrated the highest performance across all key metrics.

### **Model Evaluation Table**

The performance of the nine machine learning models was evaluated using Accuracy, Precision, Recall, F1 Score, and AUC.

| Models | Accuracy | Precision | Recall | F1 score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 90 | 89 | 93 | 90 | 85 |
| **Gradient Boost** | 86 | 84 | 86 | 84 | 86 |
| **Adaboost** | 86 | 84 | 86 | 84 | 86 |
| **Decision Tree** | 86 | 84 | 86 | 84 | 86 |
| **KNN** | 62 | 60 | 61 | 60 | 61 |
| **Logistic Regression**| **90** | **89** | **93** | **90** | **93** |
| **SVC** | 62 | 60 | 61 | 60 | 61 |
| **Gaussian NB** | 71 | 69 | 71 | 70 | 71 |
| **Bernoulli NB** | 67 | 33 | 50 | 40 | 50 |
| **MLP** | 67 | 33 | 50 | 40 | 50 |
*(Data sourced from the Model Evaluation Table in the project report)*

### **Feature Importance & ROC Curve**

The feature importance analysis confirms that **Rainfall, NDVI, and Proximity to Waterbodies** are the most critical factors.The ROC curve analysis visually confirms that **Logistic Regression (AUC = 0.93)** provides the best classification performance.


## **Conclusion 💡**

This study successfully demonstrates that an integrated machine learning framework, utilizing high-resolution geospatial data, can accurately delineate groundwater potential zones. **Logistic Regression** and **Random Forest** proved to be the most effective models, combining high accuracy with interpretability. The resulting GWPZ map provides a valuable decision-support tool for municipal water planning, guiding sustainable resource use, and enhancing drought resilience strategies in semi-arid regions like Jodhpur.

***

## **How to Use This Project 🚀**

Follow these steps to set up and run the project on your local machine.

### **Prerequisites**

* Python 3.8 or higher
* Jupyter Notebook or JupyterLab
* Git

### **Setup and Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided to install all necessary Python libraries.
    ```bash
    pip install -r requirements.txt
    ```

### **Running the Code**

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
    Or if you prefer JupyterLab:
    ```bash
    jupyter lab
    ```

2.  **Open and run the notebook:**
    Navigate to and open the `ground water.ipynb` file. You can run the cells sequentially to see the entire data processing, training, and evaluation workflow.

3.  **Data:**
    Ensure that the dataset `gw 34.csv` is located in the appropriate directory as referenced in the notebook.


