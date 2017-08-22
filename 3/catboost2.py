# -*- coding: utf-8 -*-
'''
Соревнование:
https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-3/machine-learning/predict-ad-clicks/

(c) Koziev Elijah inkoziev@gmail.com


Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''


from __future__ import print_function
import pandas as pd
import numpy as np
import catboost
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics
import zipfile


LEARNING_RATE = 0.50
DEPTH = 7
NB_SAMPLES = 5000000



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('train.shape={}'.format(train.shape))
print('test.shape={}'.format(test.shape))


# In[17]:
#train.head()


# In[18]:
#test.head()


# In[3]:

# check missing values per column
print( 'missing values per column:', train.isnull().sum(axis=0)/train.shape[0] )


# In[4]:

# impute missing values

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)


# In[5]:

# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


# In[6]:

# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute


y_mean = np.mean(train['click'].values)
print('y_mean={}'.format(y_mean))




# In[7]:

cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')


cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

# catboost accepts categorical variables as indexes
cat_cols = [ cols_to_use.index(f) for f in ['category','merchant','tminute','countrycode','offerid','thour','siteid', 'devid', 'browserid'] ]

feature_names = cols_to_use


for col_name in ['devid','browserid','countrycode','siteid']:
    print('Applying LabelEncoder to {}'.format(col_name))
    le = preprocessing.LabelEncoder()
    le.fit( list(set(train[col_name].values) | set(test[col_name].values)) )
    train[col_name] = le.transform(train[col_name].values)
    test[col_name] = le.transform(test[col_name].values)


# modeling on sampled (1e6) rows
rows = np.random.choice(train.index.values, NB_SAMPLES)
sampled_train = train.loc[rows]


# In[15]:

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']


# In[17]:

X_train, X_val, y_train, y_val = train_test_split(trainX, trainY, test_size = 0.2)

D_train = catboost.Pool( X_train.values, label=y_train.values, cat_features=cat_cols, feature_names=feature_names )
D_val = catboost.Pool( X_val.values, label=y_val.values, cat_features=cat_cols, feature_names=feature_names )


print('Start training on {}, validating on {}...'.format(X_train.shape, X_val.shape))
model = catboost.CatBoostClassifier(depth=DEPTH,
                           iterations=10000,
                           learning_rate=LEARNING_RATE,
                           eval_metric='AUC',
                           random_seed=123456,
                           auto_stop_pval=1e-2,
                           rsm=0.70,
                           l2_leaf_reg=3,
                           thread_count=1,
                           bagging_temperature=1., # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
                           #class_weights=[1., y_mean]
                                    )

model.fit(D_train, eval_set=D_val, use_best_model=True, verbose=True )

nb_trees = model.get_tree_count()
print('nb_trees={}'.format(nb_trees))

y_pred = model.predict_proba(X_val)[:,1]
val_loss = sklearn.metrics.log_loss(y_val, y_pred )
val_aucroc = sklearn.metrics.roc_auc_score(y_val, y_pred)
print('val ==> logloss={} auc_roc={}'.format(val_loss, val_aucroc))

# -------------------------------------------------------------------------

X_submit = test[cols_to_use]
pred = model.predict_proba(X_submit)[:,1]


sub = pd.DataFrame({'ID':test['ID'],'click':pred})
submission_path = '../submissions/cb_loss={:<7.5f}_aucroc={:<7.5f} [lr={} depth={} nb_samples={}].csv'.format(val_loss,val_aucroc,LEARNING_RATE,DEPTH,NB_SAMPLES)
sub.to_csv( submission_path, index=False, float_format='%8.5f' )

#with zipfile.ZipFile(submission_path+'.zip', 'w') as zip_name:
#    zip_name.write(submission_path, submission_path )



