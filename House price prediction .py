import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  

df = pd.read_csv('//Enter your system path')
print(df.head())

df['Garage'] = df['Garage'].map({'No': 0, 'Yes': 1})

print(df.describe())

print(df.isnull().sum())

plt.title("Correlation Matrix")
plt.show()

print(df.columns)

X = df[['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt',
       'Location', 'Condition', 'Garage']]
y = df['Price']

categorical_features = ['Location', 'Condition']  # Include 'Condition' if it's categorical
numerical_features = ['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Garage']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numerical values with the mean
            ('passthrough', 'passthrough')]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with the most frequent
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), categorical_features)])

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

new_data = [["enter your input based on their values"]]
new_data_df = pd.DataFrame(new_data, columns=['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Location', 'Condition', 'Garage'])
predicted_price = model.predict(new_data_df)

print("Predicted Price:", predicted_price[0])














