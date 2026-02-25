# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Create polynomial features.
6. Set up pipelines for Ridge, Lasso, and ElasticNet models.
7. Fit the models on the training data.
8. Evaluate model performance using R² score and Mean Squared Error.
9. Compare the results.


## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: PUGAZH V
RegisterNumber:   212225240109
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
data =pd.read_csv("encoded_car_data (1) (2).csv")
data.head()

data=pd.get_dummies(data,drop_first=True)

x=data.drop('price',axis=1)
y=data['price']

scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(y.values.reshape(-1,1))
data =pd.read_csv("encoded_car_data (1) (2).csv")
data.head()

data=pd.get_dummies(data,drop_first=True)

x=data.drop('price',axis=1)
y=data['price']

scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(y.values.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
results = {}
for name, model in models.items():
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    
pipeline.fit(x_train, y_train)
 predictions = pipeline.predict(x_test)
    
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
results[name] = {'MSE': mse, 'R2 Score': r2}
print('Name:PUGAZH V')
print('Reg. No:212225240109')
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, R² Score: {metrics['R2 Score']:.2f}")



results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)


plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2 Score', data=results_df, palette='viridis')
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```


## Output:
<img width="545" height="62" alt="Screenshot 2026-02-25 094005" src="https://github.com/user-attachments/assets/c6eb8bd6-add8-4b20-8a37-6a5fea1ee34c" />


![WhatsApp Image 2026-02-25 at 9 32 59 AM](https://github.com/user-attachments/assets/9c372c9c-12b2-4e0c-848c-21b9867fab92)

![WhatsApp Image 2026-02-25 at 9 33 06 AM](https://github.com/user-attachments/assets/8c61d8a3-eb13-4afe-8c32-88248395f09a)



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
