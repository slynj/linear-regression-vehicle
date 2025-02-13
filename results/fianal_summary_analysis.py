
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error

sns.set_theme()

# Final Summary
data = pd.read_csv("../data/final_summary.csv")
data.describe()

# Regression Metrics
r2 = r2_score(data['Target'], data['Prediction'])
mse = mean_squared_error(data['Target'], data['Prediction'])
rmse = root_mean_squared_error(data['Target'], data['Prediction'])
mae = mean_absolute_error(data['Target'], data['Prediction'])

print(f"R^2 Score: {r2}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

'''
R^2 Score
R^2 tells us how well the model explains the variance in the target values. Since our score is 0.7449, 
it means that 74.49% of the variance in the target varible is explained by the model. It's not perfect, 
but it's fairly a strong fit.

MSE & RMSE
Mean Squared Error is the avg squared difference between the predicted and actual values. Higher => 
larger errors, However, it's hard to interpret because it's squared and the unit is not the same as 
the target variable ($ for price in our case). Root Mean Squared Error is the square root of MSE, 
making it easier to interpret because it's in the smae unit as the target variable. Our RMSE value is 
8042.68, meaning that on avverage, the prediction deviates from the actual values by $8042.68. 

MAE
Mean Absolute Error is the abs avg error between the predcition and the actual value. Unlike MSE, 
it doesn't square the errors => less sensitive to outliers. MAE is always < RMSE becuase it never 
squares the errors at the first place. The difference between RMSE and MAE is that RMSE penalizes 
larger errors more than MAE by squaring the differneces. In MAE they're all considered equal (= no
modification to the differences). So if the difference between RMSE and MAE are huge it tells us 
that some predictions were really off. Our MAE is $4534.62 which is half of RMSE, showing that 
we have outliers that are very off.

'''


# Prediction vs Target Value Scatter Plot

plt.figure(figsize=(10,10))
plt.scatter(data['Target'], data['Prediction'], alpha=0.2)

plt.plot([data['Target'].min(), data['Target'].max()],
         [data['Target'].min(), data['Target'].max()],
          color='red', linestyle='dashed')

plt.xlabel('Target ($)', size=18)
plt.ylabel('Prediction ($)', size=18)

plt.title('Prediction vs Target Scatter Plot')

print([data['Target'].min(), data['Target'].max()])
print([data['Target'].min(), data['Target'].max()])

plt.show()

# Residual Plot
sns.displot(data['Residual'])
plt.title("Residuals PDF", size=18)

# Residuals Box Plot
plt.figure(figsize=(5,8))
plt.boxplot(data["Residual"], vert=True, patch_artist=True, flierprops={'markerfacecolor': 'red', 'alpha': 0.5}, whiskerprops={'color': 'orange', 'linewidth': 2},  # 수염(Whiskers) 색상 설정
            capprops={'color': 'orange', 'linewidth': 2})

plt.ylabel("Residuals")
plt.title("Residuals Box Plot")

plt.show()


