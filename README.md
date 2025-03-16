# 🏡 House Price Prediction

## 📌 Project Overview
This project predicts house prices based on various features such as location, number of rooms, square footage, etc. The model uses **Linear Regression** to estimate the price of a house given its attributes.

## 🚀 Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training using **Linear Regression**
- Model evaluation and hyperparameter tuning
- Deployment of the trained model (Optional)

## 📂 Dataset
The dataset contains information on houses, including:
- **Square footage**
- **Number of bedrooms & bathrooms**
- **Location**
- **Year built**
- **Price** (target variable)

## 🔧 Technologies Used
- **Python**
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning - Linear Regression)
- **Flask/FastAPI** (Deployment, if applicable)

## 🏗️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train.py
   ```
4. If deploying, start the API:
   ```bash
   python app.py
   ```

## 📊 Model Training
1. Load and clean the dataset
2. Perform EDA (visualizations, correlations, missing values handling)
3. Split the dataset into training and testing sets
4. Train the model using **Linear Regression**
5. Evaluate the model using **RMSE, MAE, R² Score**
6. Optimize using **Feature selection and scaling**

## 🖥️ Deployment (Optional)
- Deploy the trained model using Flask/FastAPI
- Create a simple web interface using **Streamlit**

## 📜 Results
- Achieved an **R² score of ~0.75** using **Linear Regression**
- Model predicts house prices with reasonable accuracy

## 🔗 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 🤝 Contributing
Contributions are welcome! Feel free to open a pull request.

## 📜 License
This project is licensed under the MIT License.

