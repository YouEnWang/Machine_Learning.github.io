import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/train_data_titanic.csv")
# 將資料統計
df.head()
df.info()
df.describe().T

# 篩選不要的資料
df.drop(['Name','Ticket'], axis=1, inplace=True)
df.head()
df.info()
# 繪製圖表
sns.pairplot(df[['Survived','Parch']], dropna=True)
# 用想要的標題來做平均
df.groupby('Survived').mean()
# data observing 各別計數想要的資料
df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()
df['Pclass'].value_counts()
# Handle missing values

df.isnull().sum()
len(df)
len(df)/2
df.isnull().sum()>(len(df)/2)
# Cabin has too many missing values
df.drop('Cabin', axis=1, inplace=True)
df.head()

sns.pairplot(df[['Survived','Fare']], dropna=True)
# sns.countplot(df['Fare'], hue=df['Survived'])


# Age is also have some missing values
df['Age'].isnull().value_counts()
df.groupby('Sex')['Age'].median().plot(kind='bar')
# 缺失值男生就用男生的中位數(29)、女生就用女生的中位數(27)來填補
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
df['Age'].isnull().value_counts()
df.groupby('Sex')['Age'].median()
# 用Age各性別的眾數來填補缺失值
# df['Age'].fillna(df['Age'].mode()[0],inplace=True)
# df.apply(lambda x: sum(x.isnull()),axis=0)
# 用Age個別情況的眾數來填補缺失值
# table = df.pivot_table(values='Age',index='Pclass',columns='Sex',aggfunc=df['Age'].mode()[0])
# def fage(x):
    # return table.loc[x['Pclass'],x['Sex']]
# df['Age'].fillna(df.apply(fage, axis=1),inplace=True)
# # 創造新的變數：家庭人數
# df['Family'] = df['SibSp'] + df['Parch'] + 1

# Survival_Rate = df[['Family','Survived']].groupby(by=['Family']).agg(np.mean)*100
# Survival_Rate.columns = ['Survival Rate(%)']
# Survival_Rate.reset_index()
# print(Survival_Rate)
# # 將Family做級別區分
# df['Family Class'] = np.nan
# df.loc[ df.Family==0, 'Family Class' ] = 2
# df.loc[ (df.Family>=1) & (df.Family<=3), 'Family Class' ] = 3
# df.loc[ (df.Family>=4) & (df.Family<=6), 'Family Class' ] = 2
# df.loc[ (df.Family>=7), 'Family Class' ] = 1
# # 用Age個別情況的中位數來填補缺失值
# table = df.pivot_table(values='Age',index='Family Class',columns='Sex',aggfunc=np.median)
# def fage(x):
#     return table.loc[x['Family Class'],x['Sex']]
# df['Age'].fillna(df.apply(fage, axis=1),inplace=True)

# 檢查還有哪些標題有缺失資料
df.isnull().sum()
# 發現還有Embarked還有缺2個
df['Embarked'].value_counts()
# 找出第一個次數最多的，發現是S
df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df['Embarked'].value_counts()
# 檢查還有哪些標題有缺失資料
df.isnull().sum()

# 將Sex, Embarked進行轉換
# Sex轉換成是否爲男生、是否爲女生，Embarked轉換爲是否爲S、是否爲C、是否爲Q
df = pd.get_dummies(data=df, columns=['Sex','Embarked'])
df.head()
# 是否爲男生與是否爲女生只要留一個就好，留下是否爲男生
df.drop(['Sex_female'], axis=1, inplace=True)
df.head()

# # Prepare training data
# df.corr()
# # 把Survived, Fare丟掉
# X = df.drop(['Survived','Pclass'],axis=1)
# y = df['Survived']
# # split to training data & testing data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)
# # using Logistic regression model
# from sklearn.linear_model import LogisticRegression
# # 顯示說ITERATIONS REACHED LIMIT,所以增加max_iter
# lr = LogisticRegression(max_iter=200)
# lr.fit(X_train, y_train)
# predictions = lr.predict(X_test)
# predictions

# using confusion_matrix to Evaluate
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# print(precision_score(y_test, predictions))
# print(recall_score(y_test, predictions))
# print(accuracy_score(y_test, predictions))


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
def Survived_model(model, data, predictors, outcome, t_size, rs_number):
    X = data[predictors]
    y = data[outcome]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=rs_number)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    print(f"Accuracy:{accuracy}")
    print(f"Recall:{recall}")
    print(f"Precision:{precision}")

# pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Survived','Predict Survived'], index=['True not Survived', 'True Survived'])

#1
# outcome_var ='Survived'
# model = LogisticRegression(max_iter=200)
# predictor_var = ['Pclass','Sex_male','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Embarked_Q','PassengerId']
# Survived_model(model, df, predictor_var, outcome_var, 0.3, 6)

#2
# outcome_var ='Survived'
# model2 = DecisionTreeClassifier()
# predictor_var = ['Pclass','Sex_male','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Embarked_Q','PassengerId']
# Survived_model(model2, df, predictor_var, outcome_var, 0.3, 6)

#3
outcome_var = 'Survived'
model3 = RandomForestClassifier(n_estimators=10)
predictor_var = ['Pclass','Sex_male','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Embarked_Q','PassengerId']
Survived_model(model3, df, predictor_var, outcome_var, 0.3, 6)



#Model Export
import joblib
joblib.dump(model3,'Titanic-LR-20220317.pkl',compress=3)