# ML-Groundwater Potential Detection
# Identifying Groundwater Potential Zones in Jodhpur, India using Machine Learning

[cite_start]This repository contains the project "Identifying Groundwater Potential Zone," submitted in partial fulfillment for the award of the Degree of B. Tech in Computer Science and Engineering (Artificial Intelligence and Machine Learning) at Manipal University Jaipur (2024-2025)[cite: 3, 4, 5].

## 📋 Abstract

[cite_start]Groundwater is a critical resource in arid regions like Jodhpur, India[cite: 15, 34]. [cite_start]Due to over-extraction and climate change, its sustainable management has become imperative[cite: 16]. [cite_start]This research presents a machine learning-driven approach to delineate Groundwater Potential Zones (GWPZ) across the administrative boundary of Jodhpur[cite: 17]. [cite_start]Using **52 ground-truth locations** (34 well sites and 18 non-well sites) and **15 influential geospatial and hydrogeological parameters**, we trained and evaluated nine supervised classification algorithms[cite: 18, 20]. [cite_start]Our findings show that ensemble models, particularly **Random Forest (RF)** and **Logistic Regression (LR)**, offer superior accuracy in identifying GWPZs[cite: 22]. [cite_start]A feature importance analysis highlighted that **Rainfall, NDVI, and Proximity to Waterbodies** are the most significant factors influencing groundwater availability[cite: 22]. [cite_start]The final delineated map classifies the region into five zones—*Very Good, Good, Moderate, Poor, and Very Poor*—providing crucial insights for sustainable water resource management and policymaking[cite: 23].

---

## ✨ Key Features

* [cite_start]**Study Area**: Focused on the arid and semi-arid administrative boundary of Jodhpur, Rajasthan, India[cite: 34, 46].
* [cite_start]**Comprehensive Dataset**: Utilizes 15 high-resolution geospatial predictors, including rainfall, NDVI, lithology, soil characteristics, and topographical data[cite: 18].
* [cite_start]**Multi-Model Comparison**: Implements and evaluates nine machine learning models[cite: 20, 47]:
    * Random Forest (RF)
    * Gradient Boosting (GB)
    * AdaBoost
    * Decision Tree (DT)
    * Logistic Regression (LR)
    * Support Vector Classifier (SVC)
    * Naïve Bayes (NB)
    * Multilayer Perceptron (MLP)
    * K-Nearest Neighbours (KNN)
* [cite_start]**Feature Importance**: Identifies the most influential factors for groundwater potential[cite: 22, 119].
* [cite_start]**Actionable Insights**: Produces a spatially explicit GWPZ map to guide water exploration and policy[cite: 23].

---

## 🌍 Study Area & Data

[cite_start]The research focuses on Jodhpur, a region characterized by arid climatic conditions[cite: 46]. [cite_start]A geospatial dataset with a 30m resolution was created using QGIS 3.34 for data preprocessing, reprojection, and sampling[cite: 76, 78].

**Instructions:** To display the images below, upload them to your GitHub repository (e.g., into an `assets` folder) and replace `path/to/your/image.png` with the actual path to the image.

![Map of Jodhpur Study Area](path/to/your/jodhpur_map.png)
[cite_start]*Figure 1: Map of the Jodhpur study area showing well locations and water bodies[cite: 85, 86].*

### Influencing Factors (Thematic Layers)

[cite_start]Fifteen thematic layers were used to train the models, representing key topographic, hydrological, geological, and biotic factors[cite: 18, 76].

1.  [cite_start]**Rainfall**: Primary source of groundwater recharge[cite: 99].
2.  [cite_start]**Proximity to Water Bodies (DFR)**: Areas closer to rivers have higher recharge potential[cite: 98].
3.  [cite_start]**Drainage Density (DD)**: Lower density indicates greater recharge capacity[cite: 96].
4.  [cite_start]**Lithology**: Physical rock characteristics influencing porosity and permeability[cite: 100].
5.  [cite_start]**Geomorphology**: Landforms that control water flow and storage[cite: 101].
6.  [cite_start]**Topographical Wetness Index (TWI)**: Quantifies the potential for water accumulation[cite: 96].
7.  [cite_start]**Soil Depth**: Deeper soils offer more storage and enhance percolation[cite: 105].
8.  [cite_start]**Soil Slope**: Influences lateral water movement and infiltration rates[cite: 107].
9.  [cite_start]**Soil Texture**: Determines the proportion of sand, silt, and clay, affecting infiltration[cite: 104].
10. [cite_start]**Forest Cover**: Promotes infiltration and reduces surface runoff[cite: 111].
11. [cite_start]**Hill Shade**: Derived from the DEM to understand topography[cite: 18].
12. [cite_start]**Normalized Difference Vegetation Index (NDVI)**: Indicates vegetation density, which affects infiltration[cite: 109].
13. [cite_start]**Altitude (Elevation/DEM)**: Affects surface runoff and gravitational potential for recharge[cite: 94].
14. [cite_start]**Land Use/Land Cover (LULC)**: Categorizes human impact on groundwater systems[cite: 113].
15. [cite_start]**Slope**: Steepness of the ground surface, controlling runoff vs. infiltration[cite: 95].

![Causative Factors 1](path/to/your/causative_factors_1.png)
[cite_start]*Figure 2: Distribution of causative factors: a) Soil Depth, b) Soil Texture, c) Soil Slope, d) NDVI, e) TWI, f) Land Use Land Cover[cite: 87, 88].*

![Causative Factors 2](path/to/your/causative_factors_2.png)
[cite_start]*Figure 3: Distribution of causative factors: a) Forest type, b) Altitude, c) Slope, d) Rainfall, e) Proximity to water bodies, f) Lithology[cite: 89, 90].*

---

## ⚙️ Results and Evaluation

[cite_start]The models were trained on a 60-40 split of the ground-truth data with stratified sampling[cite: 120]. [cite_start]Logistic Regression demonstrated the highest performance across all key metrics[cite: 123].

### Model Evaluation

[cite_start]The performance of the nine machine learning models was evaluated using Accuracy, Precision, Recall, F1 Score, and AUC[cite: 21].

| Models              | Accuracy | Precision | Recall | F1 score | AUC |
| ------------------- | :------: | :-------: | :----: | :------: | :---: |
| **Random Forest** |    90    |    89     |   93   |    90    |  85   |
| **Gradient Boost** |    86    |    84     |   86   |    84    |  86   |
| **Adaboost** |    86    |    84     |   86   |    84    |  86   |
| **Decision Tree** |    86    |    84     |   86   |    84    |  86   |
| **KNN** |    62    |    60     |   61   |    60    |  61   |
| **Logistic Regression**|  **90** |  **89** | **93** |  **90** |**93** |
| **SVC** |    62    |    60     |   61   |    60    |  61   |
| **Gaussian NB** |    71    |    69     |   71   |    70    |  71   |
| **Bernoulli NB** |    67    |    33     |   50   |    40    |  50   |
| **MLP** |    67    |    33     |   50   |    40    |  50   |
[cite_start]*(Data sourced from the Model Evaluation Table in the project report [cite: 122])*

### Feature Importance

[cite_start]The feature importance analysis from the Random Forest model shows that **Rainfall**, **NDVI**, and **Proximity to Waterbodies** are the most critical factors in determining groundwater potential in the study area[cite: 121].

![Feature Importance Graph](path/to/your/feature_importance.png)

### ROC Curve

[cite_start]The Receiver Operating Characteristic (ROC) curve analysis visually confirms that Logistic Regression (AUC = 0.93) provides the best classification performance among the tested models[cite: 122].

![ROC Curve Graph](path/to/your/roc_curve.png)

---

## 💡 Conclusion

[cite_start]This study successfully demonstrates that an integrated machine learning framework, utilizing high-resolution geospatial data, can accurately delineate groundwater potential zones[cite: 125]. [cite_start]**Logistic Regression** and **Random Forest** proved to be the most effective models, combining high accuracy with interpretability[cite: 22, 126]. [cite_start]The resulting GWPZ map provides a valuable decision-support tool for municipal water planning, guiding sustainable resource use, and enhancing drought resilience strategies in semi-arid regions like Jodhpur[cite: 128].

---
