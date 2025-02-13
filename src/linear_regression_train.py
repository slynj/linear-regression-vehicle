
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set_theme()

data = pd.read_csv("../data/vehicle_clean_final.csv")
target = data['Log Price']
input = data.drop(['Log Price'], axis=1)

scaler = StandardScaler()
scaler.fit(input)
input_scaled = scaler.transform(input)

x_train, x_test, y_train, y_test = train_test_split(input_scaled, target, test_size=0.2, random_state=42)

# Regression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_train)

plt.scatter(y_train, y_pred, alpha=0.2)

plt.xlabel('Target (Train Data)', size=18)
plt.ylabel('Prediction', size=18)

plt.xlim(6, 13)
plt.ylim(6, 13)

plt.show()


# Residuals Distribution
sns.displot(y_train - y_pred)
plt.title("Residuals PDF", size=18)

# Score
score = reg.score(x_train, y_train)

intercept = reg.intercept_
coefficient = reg.coef_

print("Score: ", score)
print("Intercept: ", intercept)
print("Coefficients: ", coefficient)

summary = pd.DataFrame(input.columns.values, columns=['Features'])
summary['Weights'] = reg.coef_

y_pred_test = reg.predict(x_test)

plt.scatter(y_test, y_pred_test, alpha=0.2)

plt.xlabel('Target (Test Data)', size=18)
plt.ylabel('Prediction', size=18)

plt.xlim(6, 13)
plt.ylim(6, 13)

plt.show()

# Actual Price & Difference%
final_summary = pd.DataFrame(np.exp(y_pred_test), columns=['Prediction'])
y_test = y_test.reset_index(drop=True)

final_summary['Target'] = np.exp(y_test)
final_summary['Residual'] = final_summary['Target'] - final_summary['Prediction']
final_summary['Difference%'] = np.absolute((final_summary['Residual'] / final_summary['Target']) * 100)
final_summary.to_csv("../data/final_summary.csv", index=False)
