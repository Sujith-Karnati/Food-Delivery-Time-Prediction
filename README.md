# 🚚 Food Delivery Time Prediction

> **An end-to-end Machine Learning project for predicting food delivery time using delivery, traffic, weather, distance, and order-related information.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-ff4b4b?logo=streamlit)
![Git](https://img.shields.io/badge/Git-Version%20Control-orange?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)

---

## 📌 About the Project

Food delivery time depends on several factors such as:

* 📍 Distance between restaurant and customer
* 🚦 Traffic conditions
* 🌦️ Weather conditions
* 🛵 Delivery-related information
* 🍔 Order characteristics
* 📌 Location/zone-related factors

The goal of this project is to build a Machine Learning model that can estimate the expected delivery time for a given order.

### 🎯 Objective

> **Build an accurate and reliable regression model that predicts `Time_taken_min` while avoiding data leakage and deploy the final model through a user-friendly Streamlit application.**

---

# 🧭 Project Workflow

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Data Cleaning  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │       EDA       │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │Feature Engineering│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Preprocessing   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Train/Test Split│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Training  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Evaluation│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Hyperparameter  │
                    │     Tuning      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Final Model    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Streamlit     │
                    │   Deployment    │
                    └─────────────────┘
```

---

# 🛠️ Tech Stack

### 💻 Programming Language

* Python

### 📚 Libraries & Frameworks

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit

### 🔧 Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📊 Exploratory Data Analysis

The dataset was analyzed to understand the relationship between different features and delivery time.

### 🔍 Key Areas Explored

| Analysis Area                     | Purpose                        |
| --------------------------------- | ------------------------------ |
| Distribution of delivery time     | Understand target distribution |
| Distance vs delivery time         | Analyze distance impact        |
| Traffic vs delivery time          | Understand traffic impact      |
| Weather vs delivery time          | Analyze weather effects        |
| Weekend vs weekday patterns       | Compare delivery patterns      |
| Categorical feature distributions | Understand category frequency  |
| Numerical correlations            | Identify relationships         |
| Outlier analysis                  | Detect unusual observations    |

### 💡 Example Business Insights

* Higher traffic conditions generally correspond to longer delivery times.
* Weather conditions can affect delivery duration.
* Delivery distance has an important relationship with delivery time.
* Delivery patterns can vary between weekdays and weekends.

---

# 🧹 Data Preprocessing

The following preprocessing techniques were applied where required:

* Handling missing values
* Removing duplicate records
* Numerical feature preprocessing
* Categorical feature encoding
* Feature scaling
* Train-test splitting
* Feature selection

A preprocessing pipeline was created using Scikit-learn tools such as:

```python
ColumnTransformer
OneHotEncoder
SimpleImputer
StandardScaler
SelectPercentile
```

---

# ⚠️ Data Leakage Prevention

## 🚨 Identifying Target Leakage

One of the most important findings in this project was **target leakage**.

The dataset contained the feature:

```text
Average_Speed_kmph
```

which was calculated using:

```text
Average_Speed_kmph = Road_Distance_km / Time_taken_min
```

Since `Time_taken_min` is the target variable, this feature indirectly contained the answer that the model was supposed to predict.

Including this feature would produce artificially high model performance and would not represent a realistic prediction scenario.

### ❌ Leaky Feature

```python
Average_Speed_kmph
```

### ✅ Solution

The feature was removed before model training:

```python
X = df.drop(columns=[
    "Time_taken_min",
    "Average_Speed_kmph"
])

y = df["Time_taken_min"]
```

The models were then trained and evaluated again without the leaked feature.

### 🎯 Why This Matters

This ensures that the final model uses only information that would realistically be available **before the delivery is completed**.

---

# 🤖 Machine Learning Models

Multiple regression algorithms were experimented with and compared.

### Models Used

* Linear Regression
* KNN Regression
* Decision Tree
* Support Vector Machine
* Random Forest Regressor
* AdaBoost Regressor
* Extra Trees Regressor
* Gradient Boosting
* XGBoost Regressor

The models were evaluated using standard regression metrics.

---

# 📏 Model Evaluation

The following metrics were used to evaluate the regression models.

## 1️⃣ MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted delivery time.

```text
MAE = average(|Actual - Predicted|)
```

**Lower MAE is better.**

---

## 2️⃣ MSE — Mean Squared Error

Penalizes larger prediction errors more heavily.

```text
MSE = average((Actual - Predicted)²)
```

**Lower MSE is better.**

---

## 3️⃣ RMSE — Root Mean Squared Error

The square root of MSE and expressed in the same unit as the target.

```text
RMSE = √MSE
```

**Lower RMSE is better.**

---

## 4️⃣ R² Score

Measures how much variance in delivery time is explained by the model.

```text
R² = 1 - (SS_res / SS_total)
```

A higher R² generally indicates better fit.

---

# 🔧 Hyperparameter Tuning

Hyperparameter optimization was performed to improve model performance.

### Techniques Explored

* `GridSearchCV`
* `RandomizedSearchCV`

These methods were used to search for better combinations of model hyperparameters while evaluating model performance systematically.

---

# 🌐 Streamlit Deployment

The final Machine Learning model was deployed using **Streamlit**.

The application provides an interactive interface where users can enter the required delivery information and receive a predicted delivery time.

### 🔄 Application Flow

```text
User Input
    ↓
Input Validation
    ↓
Preprocessing Pipeline
    ↓
Trained ML Model
    ↓
Predicted Delivery Time
```

### 📌 Example Output

```text
Predicted Delivery Time:
32.5 minutes
```

The deployed application makes the ML model accessible without requiring users to interact directly with the training notebook.

---

# 📁 Project Structure

```text
Food-Delivery-Time-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Building.ipynb
│   └── Model_Evaluation.ipynb
│
├── models/
│   └── final_model.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> **Note:** Adjust the folder/file names above to match your actual GitHub repository structure.

---

# 🚀 Run the Project Locally

Follow the steps below to run the project on your local machine.

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

## 2. Navigate to the Project

```bash
cd Food-Delivery-Time-Prediction
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🌍 Live Application

### 🚀 Streamlit App

## **Live Application:**

# 💡 Key Learnings

Through this project, I gained practical experience in:

* End-to-end Machine Learning workflow
* Regression problems
* Exploratory Data Analysis
* Feature engineering
* Numerical and categorical preprocessing
* Scikit-learn pipelines
* Model comparison
* Regression evaluation metrics
* Hyperparameter tuning
* Data leakage detection
* Preventing target leakage
* Model serialization
* Streamlit application development
* ML model deployment
* Git and GitHub project management

---

# ⚠️ Important Project Lesson

A major lesson from this project was that:

> **High model performance does not always mean a good Machine Learning model.**

During the project, `Average_Speed_kmph` produced target leakage because it was calculated using the target variable `Time_taken_min`.

After identifying the issue, the feature was removed and the model was retrained.

This reinforced an important ML principle:

> **Features used for prediction must be available at prediction time and must not contain information derived from the target.**

---

# 🔮 Future Improvements

Possible future improvements include:

* Collecting more real-world delivery data
* Improving feature engineering
* Adding real-time traffic information
* Adding live weather information
* Experimenting with advanced ensemble models
* Model monitoring after deployment
* Tracking prediction errors
* Adding an API layer for production use
* Containerizing the application using Docker

---

# 👨‍💻 Author

## Sujith Karnati 

**Machine Learning | Data Analytics | Python | AI/ML**

---

# ⭐ Support the Project

If this project helped you understand Machine Learning workflows, feel free to ⭐ the repository.

---

### 📌 Project Highlights

```text
🚚 Food Delivery Prediction
        │
        ├── 📊 EDA
        ├── 🧹 Data Preprocessing
        ├── ⚙️ Feature Engineering
        ├── 🤖 Multiple ML Models
        ├── 📏 Model Evaluation
        ├── 🔧 Hyperparameter Tuning
        ├── ⚠️ Leakage Prevention
        ├── 💾 Model Serialization
        └── 🌐 Streamlit Deployment
```
