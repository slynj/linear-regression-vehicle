import pandas as pd
import seaborn as sns
sns.set_theme()

raw_df = pd.read_csv('../data/vehicle_raw.csv')

# Drop Unnecessary Column
# remove model column (312 unique values are too much + unnecessary)
data = raw_df.drop(['Model'], axis=1)
data.head()

# Drop Rows with Null
# sum of null values => in percentage
(data.isnull().sum() / len(data)) * 100 

# lower than 5% => we'll just drop those rows
data.dropna(axis=0, inplace=True)

# Remove Price Outlier
data.info()

# Focus on Price, EngineV, Mileage bc they're numbers => examin PDFs
sns.displot(data['Price'])
data['Price'].describe()

top98 = data['Price'].quantile(0.98)
data_rm_outlier_price = data[data['Price'] < top98]
sns.displot(data_rm_outlier_price['Price'])

# Remove Mileage Outlier
sns.displot(data_rm_outlier_price['Mileage'])
data_rm_outlier_price['Mileage'].describe()
top99 = data_rm_outlier_price['Mileage'].quantile(0.99)
data_rm_outlier_mileage = data_rm_outlier_price[data_rm_outlier_price['Mileage'] < top99]
sns.displot(data_rm_outlier_mileage['Mileage'])

# Remove Enginge Volume Outlier
sns.displot(data_rm_outlier_mileage['EngineV'])
data_rm_outlier_mileage['EngineV'].describe()
top99 = data_rm_outlier_mileage['EngineV'].quantile(0.99)
data_rm_outlier_engineV = data_rm_outlier_mileage[data_rm_outlier_mileage['EngineV'] <= top99]
sns.displot(data_rm_outlier_engineV['EngineV'])

# Remove Year Outlier
sns.displot(data_rm_outlier_engineV['Year'])
data_rm_outlier_engineV['Year'].describe()
# note that it's skewed to the right -> remove bottom n% values
bott1 = data_rm_outlier_engineV['Year'].quantile(0.01)
data_rm_outlier_year = data_rm_outlier_engineV[data_rm_outlier_engineV['Year'] > bott1]
sns.displot(data_rm_outlier_year['Year'])

# Reset Index
data_rm_outlier = data_rm_outlier_year
data_rm_outlier.head(10) # There's no inex 5, 6, ...etc -> bc we removed these rows while preprocessing
data_clean = data_rm_outlier.reset_index(drop=True)
data_rm_outlier.head(10)

# Final Data
data_clean.describe(include='all')
data_clean.to_csv("../data/vehicle_clean.csv", index=False)


