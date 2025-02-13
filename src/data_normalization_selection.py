import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme()

# Ordinary Least Square Assumption
data = pd.read_csv("../data/vehicle_clean.csv")
data.head()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))

ax1.scatter(data['Year'], data['Price'])
ax1.set_title('Year vs Price')
ax1.set_xlabel('\nYear')

ax2.scatter(data['EngineV'], data['Price'])
ax2.set_title('EngineV vs Price')
ax2.set_xlabel('\nEngineV')

ax3.scatter(data['Mileage'], data['Price'])
ax3.set_title('Mileage vs Price')
ax3.set_xlabel('\nMileage')

plt.show()

# Log Transformation (Price)
price_log = np.log(data['Price'])
data['Log Price'] = price_log
data['Log Price'].describe()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))

ax1.scatter(data['Year'], data['Price'])
ax1.set_title('Year vs Price')
ax1.set_xlabel('\nYear')

ax2.scatter(data['EngineV'], data['Price'])
ax2.set_title('EngineV vs Price')
ax2.set_xlabel('\nEngineV')


ax3.scatter(data['Mileage'], data['Price'])
ax3.set_title('Mileage vs Price')
ax3.set_xlabel('\nMileage')

plt.show()

data = data.drop(['Price'], axis=1)
data.head()

# Multicollinearity
variables = data[['Mileage', 'Year', 'EngineV']]

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns

data = data.drop(['Year'], axis=1)
data.head()

# Dummification
data = pd.get_dummies(data, drop_first=True)

data.columns.values

cols = ['Log Price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
       'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota',
       'Brand_Volkswagen', 'Body_hatch', 'Body_other', 'Body_sedan',
       'Body_vagon', 'Body_van', 'Engine Type_Gas', 'Engine Type_Other',
       'Engine Type_Petrol', 'Registration_yes']

data = data[cols]

data.to_csv("../data/vehicle_clean_final.csv", index=False)
