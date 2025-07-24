PROBLEM STATEMENT

This project involves building a Deep Learning model to predict product reordering behaviour by users. The aim is to identify patterns in purchasing habits and predict whether a product will be reordered in the next purchase cycle. The project includes data pre-processing, exploratory data analysis, feature engineering, and model deployment using Streamlit.

🧼 DATA PREPROCESSING

Missing Values: Handled using MICE (Multiple Imputation by Chained Equations) for robust estimation. Outlier Detection: Outliers were identified and treated based on domain logic and distribution analysis. Collinearity Check: Multicollinearity among features was checked and mitigated to prevent redundancy. Top Products Filter: The top 25 products were selected based on reorder frequency for focused modeling. Sampling: Dataset size was reduced using sampling techniques to optimize model training time and resource usage.

📊 EXPLORATORY DATA ANALYSIS (EDA)

Performed comprehensive EDA to understand: Product-wise reorder frequency, Department and aisle contribution to reorder, User behavior and patterns over time, Correlation between engineered features.

🛠️ FEATURE ENGINEERING

Extracted meaningful features from product, department, and aisle names. Applied OneHotEncoding to categorical variables: product_name, department_name, aisle_name. Scaled numerical features using MinMaxScaler.

🧪 MODEL DETAILS

Architecture: Inner Layers: Activated using ReLU. Output Layer: Activated using Softmax. Loss Function: Categorical Cross-Entropy. Optimizer: Adam. Evaluation Metrics: F1 Score ✅ Accuracy ✅ Precision ✅ Recall ✅

🧠 TARGET VARIABLE

The target column is: Reordered (Binary Classification)

📈 RESULTS

Achieved high precision and F1 score on filtered top 25 product categories. Model successfully predicts reorder behaviour based on user-product patterns.
