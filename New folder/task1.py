import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
url= 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
s_data = pd.read_csv(url)
s_data.head(10)
s_data.shape
s_data.info()
print("\nMissing values :",s_data.isnull().sum().values.sum())
s_data.describe()
s_data.info()
s_data.plot(kind='scatter',x='Hours',y='Scores');
plt.show()
s_data.corr(method='pearson')
s_data.corr(method='spearman')
hours=s_data['Hours']
Scores=s_data['Scores']
sns.distplot(hours)
sns.distplot(Scores)
x=s_data.iloc[:, :-1].values
y=s_data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x, y)
plt.plot(x, line);
plt.show()
y_pred=reg.predict(x_test)
actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()
hours=np.array(9.25)
hours=hours.reshape(-1,1)
pred=reg.predict(hours)
print("no of hours={}",format(hours))
print("No of Hours={}",format(hours))
print("predicted score ={}",format(pred[0]))
from sklearn import metrics
print("mean absolute error:",metrics.mean_absolute_error(y_test,y_pred))
print('adjusted R square:',metrics.r2_score(y_test,y_pred))