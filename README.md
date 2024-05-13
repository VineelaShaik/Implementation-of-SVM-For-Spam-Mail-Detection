# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vineela Shaik
RegisterNumber:  212223040243
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy


```

## Output:

Result Output

![328540245-887b0d7f-0083-48d4-89ad-d3e8b413aad3](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/aa78ff24-6b60-4332-8400-9e6211d17680)

data.head()

![328540415-6fd06372-cb47-4e6f-8ab1-8aa7fb447b58](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/e3669bfa-8f06-4c45-bf17-a9c4b6f24731)

data.info()

![328540623-671c942a-ced7-4c91-a4e0-81e3c3893822](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/33b98b3d-4bbd-4aee-a71c-85b565e0d981)

data.isnull().sum()

![328540830-4741dfff-dcf7-43e2-af2e-9c32937c2444](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/e6db3511-d83a-4ecd-b551-5ce46ecd7367)

Y_prediction Value

![328541019-5a9748ec-df64-4d8a-9f54-7db87b98d844](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/3ab8be8a-b304-41e9-9b60-616ced04498b)

Accuracy Value

![328541258-cc2ca4e7-26ae-4bb0-b52e-fb3a143118fa](https://github.com/23014226/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568974/3241b04a-f8b1-4825-9c92-67818cb1feac)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
