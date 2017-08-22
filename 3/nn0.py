# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sklearn.metrics
import gc

NB_SAMPLES = 1500000

# In[2]:

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:

print ('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print ('The test data has {} rows and {} columns'.format(test.shape[0],test.shape[1]))


# In[4]:

train.head()


# In[5]:

# imputing missing values
train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None",inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None",inplace=True)
test['devid'].fillna("None",inplace=True)


# In[7]:

# create timebased features

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['tweekday'] = train['datetime'].dt.weekday
test['tweekday'] = test['datetime'].dt.weekday

train['thour'] = train['datetime'].dt.hour
test['thour'] = test['datetime'].dt.hour

train['tminute'] = train['datetime'].dt.minute
test['tminute'] = test['datetime'].dt.minute


# In[141]:

# create aggregate features
site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_offer_count_test = test.groupby(['siteid','offerid']).size().reset_index()
site_offer_count_test.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_cat_count_test = test.groupby(['siteid','category']).size().reset_index()
site_cat_count_test.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

site_mcht_count_test = test.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count_test.columns = ['siteid','merchant','site_mcht_count']


# In[158]:

# joining all files
agg_df = [site_offer_count,site_cat_count,site_mcht_count]
agg_df_test = [site_offer_count_test,site_cat_count_test,site_mcht_count_test]

for x in agg_df:
    train = train.merge(x)
    
for x in agg_df_test:
    test = test.merge(x)


# In[28]:

# Label Encoding
from sklearn.preprocessing import LabelEncoder
for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))        


# In[163]:

# sample 10% data - to avoid memory troubles
# if you have access to large machines, you can use more data for training

print('Selecting random {} samples'.format(NB_SAMPLES))
rows = np.random.choice(train.index.values, NB_SAMPLES)
sampled_train = train.loc[rows]


# In[164]:

# select columns to choose
cols_to_use = [x for x in sampled_train.columns if x not in list(['ID','datetime','click'])]


# In[165]:

# standarise data before training
scaler = StandardScaler().fit(sampled_train[cols_to_use])

strain = scaler.transform(sampled_train[cols_to_use])
stest = scaler.transform(test[cols_to_use])


# In[167]:

# train validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, sampled_train.click, test_size = 0.5, random_state=2017)


# In[168]:

print (X_train.shape)
print (X_valid.shape)
print (Y_train.shape)
print (Y_valid.shape)


# In[169]:

# model architechture
def keras_model(train):
    
    input_dim = train.shape[1]
    classes = 2
    
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape = (input_dim,))) #layer 1
    model.add(Dense(30, activation = 'relu')) #layer 2
    model.add(Dense(classes, activation = 'sigmoid')) #output
    model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['accuracy'])

    print(model.summary)

    return model

# ### Now, let's understand the architechture of this neural network:
# 
# 1. We have 13 input features. 
# 2. We connect these 13 features with 100 neurons in the first hidden layer (call layer 1).
# 3. Visualise in mind this way: The lines connecting input to neurons are assigned a weight (randomly assigned).
# 4. The neurons in layer 1 receive a weighted sum (bias + woxo + w1x1...) of inputs while passing through `relu` activation function.
# 5. Relu works this way: If the value of weighted sum is less than zero, it sets it to 0, if the value of weighted sum of positive, it considers the value as is.
# 6. The output from layer 1 is input to layer 2 which has 30 neurons. Again, the input passes through `relu` activation function.   
# 7. Finally, the output of layer 2 is fed into the final layer which has 2 neurons. The output passes through `sigmoid` function. `Sigmoid` functions makes sure that probabilities stays within 0 and 1 and we get the output predictions.

# In[171]:

# one hot target columns
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)

ids = np.copy(test.ID.values)

del sampled_train
del train
del test
gc.collect()

# In[173]:

early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')

model_checkpoint = ModelCheckpoint('../data/nn.model', monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto')

# train model
model = keras_model(X_train)
model.fit(X_train,
          Y_train,
          batch_size=1000,
          epochs=500,
          callbacks=[early_stopping,model_checkpoint],
          validation_data=(X_valid, Y_valid),
          shuffle=True)

model.load_weights('../data/nn.model')

# In[177]:

# check validation accuracy
vpreds = model.predict_proba(X_valid)[:,1]
test_loss = sklearn.metrics.log_loss( y_true=Y_valid[:,1], y_pred=vpreds)
test_aucroc = sklearn.metrics.roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)
print('\ntest_logloss={} test_aucroc={}'.format(test_loss,test_aucroc))

# In[178]:

# predict on test data
test_preds = model.predict_proba(stest)[:,1]


# In[180]:

# create submission file
submit = pd.DataFrame({'ID':ids, 'click':test_preds})
submit.to_csv('../submissions/nn_loss={:<7.5f}_aucroc={:<7.5f}.csv'.format(test_loss,test_aucroc), index=False)


# In[ ]:



