# Data-310-Lab-6

## Question 14: If we retain only two input features, such as "mean radius" and "mean texture" and apply the Gaussian Naive Bayes model for classification, then the average accuracy determined on a 10-fold cross validation with random_state = 1693 is (do not use the % notation, just copy the first 4 decimals):

x1 = pd.DataFrame(X['mean radius'])

x2 = pd.DataFrame(X['mean texture'])

x = pd.concat([x1,x2], axis = 1)

x = np.array(x)

from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

scale = StandardScaler()

pipe = Pipeline([('scale',scale),('Classifier',model)])

def validation(x,y,k,model):

  PA_IV = []
  
  PA_EV = []
  
  pipe = Pipeline([('scale',scale),('Classifier',model)])
  
  kf = KFold(n_splits = k, shuffle = True, random_state = 1693)
  
  for idxtrain, idxtest in kf.split(x):
  
    x_train = x[idxtrain,:]
    
    y_train = y[idxtrain]
    
    x_test = x[idxtest,:]
    
    y_test = y[idxtest]
    
    pipe.fit(x_train,y_train)
    
    PA_IV.append(accuracy_score(y_train,pipe.predict(x_train)))
    
    PA_EV.append(accuracy_score(y_test,pipe.predict(x_test)))
    
  return np.mean(PA_IV), np.mean(PA_EV)
  
validation(x,y,10,model) = (0.8840079800194932, 0.8805137844611528)

## Question 15: From the data retain only two input features, such as "mean radius" and "mean texture" and apply the Random Froest model for classification with 100 trees, max depth of 7 and random_state=1693; The average accuracy determined on a 10-fold cross validation with thesame random state is (do not use the % notation, just copy the first 4 decimals): 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)

scale = StandardScaler()

pipe = Pipeline([('scale',scale),('Classifier',model)])

validation(x,y,10,model):(0.9652404666179336, 0.8822368421052632)

## Question 16: 

x1 = pd.DataFrame(X['mean radius'])

x2 = pd.DataFrame(X['mean texture'])

x = pd.concat([x1,x2], axis = 1)

x = np.array(x)

scale = StandardScaler()

X = scale.fit_transform(x)

from keras.models import Sequential

from keras.layers import Dense 

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as acc

model = Sequential()

model.add(Dense(16,kernel_initializer='random_normal', activation='relu'))

model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))

model.add(Dense(4,kernel_initializer='random_normal', activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle=True,random_state=1693)

AC = []

for idxtrain, idxtest in kf.split(X):

  Xtrain = X[idxtrain]
  
  Xtest  = X[idxtest]
  
  ytrain = y[idxtrain]
  
  ytest  = y[idxtest]
  
  Xstrain = scale.fit_transform(Xtrain)
  
  Xstest  = scale.transform(Xtest)
  
  model.fit(Xtrain,ytrain,epochs=150,validation_split=0.25,batch_size=10,shuffle=False)
  
  AC.append(acc(ytest,model.predict_classes(Xstest)))
  
  print(acc(ytest,model.predict_classes(Xstest)))
  
  np.mean(AC) = 0.8858082706766917
