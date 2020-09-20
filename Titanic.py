# Titanic competition
# kaggle.com competition
# solved using Random forest classifier with 73.684% accuracy
# solved using ANN with with 76.555% accuracy
# https://www.kaggle.com/c/titanic/overview

# for any further questions, inquiries or communicating
# Mahmoud Mostafa Tayee
# mahmoud.tayee.1994@gmail.com

######################## section 1: importing libraraies ########################
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

######################## section 2: preprocessing the training data ########################
training_data = pd.read_csv('D:\\Not yours\\Titanic\\train.csv')

# to view the data characteristics
training_data.info()


# droppimg the useless data
titanic_data = training_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

# to see the number of missing data for each column
titanic_data.isnull().sum()

# imputing the missing data of the Age with number of parents and children aboard from 'Parch' column
parch_grouped = titanic_data.groupby(['Parch'])
parch_grouped.mean()

def age_approx(colms):
    age = colms[0]
    parch = colms[1]
    
    if pd.isnull(age):
        if parch == 0:
            return 32
        elif parch == 1:
            return 24
        elif parch == 2:
            return 17
        elif parch == 3:
            return 33
        elif parch == 4:
            return 45
        else:
            return 30
    else:
        return age
    
    
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis = 1)

# dropping the 2 missing records of Embarked 
titanic_data.dropna(inplace = True)

# resetting index after dropping some records
titanic_data.reset_index(inplace = True, drop = True)

# dealing with categorical data 'Sex'
le_sex = LabelEncoder()
titanic_data['Sex'] = le_sex.fit_transform(titanic_data['Sex'])

# dealing with categorical data 'Embarked'
le_embarked = LabelEncoder()
titanic_data['Embarked'] = le_embarked.fit_transform(titanic_data['Embarked'])
ohe = OneHotEncoder()
embarked_one_hot = ohe.fit_transform(titanic_data['Embarked'].values.reshape(-1,1)).toarray()
embarked_DF = pd.DataFrame(embarked_one_hot, columns = ['C', 'Q', 'S'])

# concatenating the one hotted embarked with our data
titanic_data.drop(['Embarked'], inplace = True, axis = 1)
titanic_data = pd.concat([titanic_data, embarked_DF], axis = 1)

# break the correlation between variables by dropping some columns.
# heatmap to view the correlation between variables
sb.heatmap(titanic_data.corr())
# dropping 'S' as its high correlation with 'C' and 'Fare' as its high correlation with 'Pclass'
titanic_data.drop(['S', 'Fare'], inplace = True, axis = 1)

# splitting data to training and testing
X_train, X_test, y_train, y_test = train_test_split(titanic_data.drop('Survived', axis = 1), titanic_data['Survived'], test_size = 0.2)

######################## section 3: creating the model ########################

#### 1st model: solve using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

prediction = classifier.predict(X_test)


#### 2nd model: solve using ANN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# create model
ANN_model = Sequential()

ANN_model.add(Dense(128, input_dim = 7 , activation = 'relu'))
ANN_model.add(Dense(64, activation = 'relu'))
ANN_model.add(Dense(32, activation = 'relu'))
ANN_model.add(Dense(16, activation = 'relu'))
ANN_model.add(Dense(8, activation = 'relu'))
ANN_model.add(Dense(1, activation = 'sigmoid'))


#compile model
ANN_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the data to the model
ANN_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10, verbose=2)

# scores = ANN_model.evaluate(X_test, y_test, verbose = 0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))

######################## section 4: preprocessing the data to submit ########################
# reading the data that need to be predicted
X_to_predict = pd.read_csv('D:\\Not yours\\Titanic\\test.csv')

# droppimg the useless data
X_to_predict = X_to_predict.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

# predicting missing values of age
X_to_predict['Age'] = X_to_predict[['Age', 'Parch']].apply(age_approx, axis = 1)

# dealing with categorical data 'Sex'
X_to_predict['Sex'] = le_sex.transform(X_to_predict['Sex'])

# dealing with categorical data 'Embarked'
X_to_predict['Embarked'] = le_embarked.transform(X_to_predict['Embarked'])
embarked_one_hot = ohe.transform(X_to_predict['Embarked'].values.reshape(-1,1)).toarray()
embarked_DF = pd.DataFrame(embarked_one_hot, columns = ['C', 'Q', 'S'])

# concatenating the one hotted embarked with our data
X_to_predict.drop(['Embarked'], inplace = True, axis = 1)
X_to_predict = pd.concat([X_to_predict, embarked_DF], axis = 1)

# break the correlation between variables by dropping some columns.
# dropping 'S' as its high correlation with 'C' and 'Fare' as its high correlation with 'Pclass'
X_to_predict.drop(['S', 'Fare'], inplace = True, axis = 1)

# taking the 'PassengerID' column
PassengerID = X_to_predict['PassengerId'].values
X_to_predict.drop('PassengerId', axis = 1 , inplace = True)

######################## section 5: prediction  ########################

# 1st prediction: using Random Forest classifier
prediction = classifier.predict(X_to_predict)

# 2nd prediction: using ANN

prediction = ANN_model.predict(X_to_predict)
prediction = pd.DataFrame(prediction)
prediction[prediction < 0.5] = 0
prediction[prediction >= 0.5] = 1
prediction = prediction.astype(int)


######################## section 6: putting the submission data in its format  ########################

final_value = pd.DataFrame(PassengerID, columns = ['PassengerId'])
final_value['Survived'] = prediction

final_value.to_csv('D:\\Not yours\\Titanic\\submit.csv', index= False)











