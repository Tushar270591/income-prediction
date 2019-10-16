import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from scipy import stats
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,Imputer,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('./income_training/income_training.csv',low_memory=False)
dataset1 = pd.read_csv('./test_data.csv',low_memory=False)

dataset = dataset.fillna(method='ffill')
dataset1 = dataset1.fillna(method='ffill')
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])


for column in dataset1.columns:
    if dataset1[column].dtype == type(object):
        le = LabelEncoder()
        dataset1[column] = le.fit_transform(dataset1[column])

z = np.abs(stats.zscore(dataset))
# dataset = dataset[(z < 3).all(axis=1)]
features = ['Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]']
X = dataset[features].values     #reshape:arranged the data in ascending order.
y = dataset['Income in EUR'].values
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Income in EUR'])

X_train = dataset[features].values     #reshape:arranged the data in ascending order.
y_train = dataset['Income in EUR'].values
X_test = dataset1[features].values     #reshape:arranged the data in ascending order.
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, features, columns=['Coefficient'])
print(coeff_df)
y_pred = regressor.predict(X_test)
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df1 = df.head(50)


output = {'Instance': dataset1['Instance'].values,
        'Income': y_pred
        }

df = pd.DataFrame(output, columns= ['Instance', 'Income'])

export_csv = df.to_csv ('income_output.csv',index = None)
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))