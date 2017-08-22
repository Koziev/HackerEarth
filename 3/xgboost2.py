# -*- coding: utf-8 -*-
'''
Соревнование:
https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-3/machine-learning/predict-ad-clicks/

(c) Koziev Elijah inkoziev@gmail.com

Справка по XGBoost:
http://xgboost.readthedocs.io/en/latest/
http://xgboost.readthedocs.io/en/latest/python/python_api.html

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''

from __future__ import print_function
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics
import zipfile
import collections
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama


NB_SAMPLES = 2000000
LEARNING_RATE = 0.9
MAX_DEPTH = 7


USE_HYPEROPT = False

# кол-во случайных наборов гиперпараметров
N_HYPEROPT_PROBES = 500

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest

colorama.init()

# -----------------------------------------------------

# <editor-fold desc="Load and prepare datasets">
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('train.shape={}'.format(train.shape))
print('test.shape={}'.format(test.shape))

# check missing values per column
print( 'missing values per column:', train.isnull().sum(axis=0)/train.shape[0] )


train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)


# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

y_mean = np.mean(train['click'].values)
print('y_mean={}'.format(y_mean))


#offerid_cnt = collections.Counter(train.offerid.values)

#cols = ['siteid','offerid','category','merchant']
#for x in cols:
#    train[x] = train[x].astype('object')
#    test[x] = test[x].astype('object')


cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

cat_cols = [ cols_to_use.index(f) for f in ['category','merchant','tminute','countrycode','offerid','thour','siteid', 'devid', 'browserid'] ]

for col_name in ['devid','browserid','countrycode','siteid']:
    print('Applying LabelEncoder to {}'.format(col_name))
    le = preprocessing.LabelEncoder()
    le.fit( list(set(train[col_name].values) | set(test[col_name].values)) )
    train[col_name] = le.transform(train[col_name].values)
    test[col_name] = le.transform(test[col_name].values)

print('Selecting random {} samples'.format(NB_SAMPLES))
rows = np.random.choice(train.index.values, NB_SAMPLES)
sampled_train = train.loc[rows]

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

ids = test['ID']
test = test[cols_to_use]

# -------------------------------------------------------------------------

for col_name in trainX.columns.values:
    print('Column {}: '.format(col_name), end='')
    cardinality = len(set(trainX[col_name].values))
    nan_count = trainX[col_name].isnull().sum()
    print('trainX ==> dtype={} cardinality={} nan_count={}'.format( trainX[col_name].dtype, cardinality, nan_count), end='' )

    cardinality = len(set(test[col_name].values))
    nan_count = test[col_name].isnull().sum()
    print(' test ==> dtype={} cardinality={} nan_count={}'.format( test[col_name].dtype, cardinality, nan_count) )

# -------------------------------------------------------------------------

print('Applying OneHotEncoder...')
ohe = sklearn.preprocessing.OneHotEncoder( n_values='auto',
                                           categorical_features=cat_cols,
                                           sparse=True,
                                           handle_unknown='error')

ohe.fit( pd.concat( [trainX, test] ) )

X_data = ohe.transform(trainX)
X_submit = ohe.transform(test)

print('X_data.shape=', X_data.shape)
print('X_submit.shape=', X_submit.shape)

# -------------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X_data, trainY, test_size = 0.2)

D_train = xgboost.DMatrix(X_train, y_train)
D_val = xgboost.DMatrix(X_val, y_val)
D_submit = xgboost.DMatrix(X_submit)
watchlist = [(D_train, 'train'), (D_val, 'valid')]

# ---------------------------------------------------------------
# </editor-fold>


# <editor-fold desc="hyperopt + XGBoost">

def get_xgboost_params(space):
    _max_depth = int(space['max_depth'])
    _min_child_weight = space['min_child_weight']
    _subsample = space['subsample']
    _gamma = space['gamma'] if 'gamma' in space else 0.01
    _eta = space['eta']
    _seed = space['seed'] if 'seed' in space else 123456
    _colsample_bytree = space['colsample_bytree']
    _colsample_bylevel = space['colsample_bylevel']
    booster = space['booster'] if 'booster' in space else 'gbtree'

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])

    xgb_params = {
        'booster': booster,
        'subsample': _subsample,
        'max_depth': _max_depth,
        'seed': _seed,
        'min_child_weight': _min_child_weight,
        'eta': _eta,
        'gamma': _gamma,
        'colsample_bytree': _colsample_bytree,
        'colsample_bylevel': _colsample_bylevel,
        'eval_metric': 'logloss', #'auc',  # 'logloss',
        'objective': 'binary:logistic',
        'silent': 1,
        #'scale_pos_weight':1.0/np.mean(y_data)
    }

    #xgb_params['updater'] = 'grow_gpu'
    xgb_params['eval_metric'] = ['logloss','auc']

    return xgb_params

log_writer = None
obj_call_count = 0
cur_best_loss = np.inf

def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count, cur_best_loss))

    xgb_params = get_xgboost_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str))

    model = xgboost.train(params=xgb_params,
                          dtrain=D_train,
                          num_boost_round=20000,
                          evals=watchlist,
                          verbose_eval=50,
                          early_stopping_rounds=100)

    print('nb_trees={} val_loss={:7.5f}'.format(model.best_ntree_limit, model.best_score))
    # loss = model.best_score
    nb_trees = model.best_ntree_limit
    y_pred = model.predict(D_val, ntree_limit=nb_trees)
    test_loss = sklearn.metrics.log_loss(y_val, y_pred)
    test_aucroc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    print('test_loss={:<7.5f} test_aucroc={:<7.5f}'.format(test_loss, test_aucroc))

    log_writer.write('loss={:<7.5f} aucroc={:<7.5f} Params:{} nb_trees={}\n'.format(test_loss, test_aucroc, params_str, nb_trees))
    log_writer.flush()

    if test_loss < cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)
        print('Computing submission...')
        y_submit = model.predict(D_submit, ntree_limit=nb_trees)

        sub = pd.DataFrame({'ID':ids,'click':y_submit})
        submission_path = '../submissions/xgb_loss={:<7.5f} aucroc={:<7.5f} [eta={} max_depth={} nb_samples={}].csv'.format(test_loss, test_aucroc, xgb_params['eta'], xgb_params['max_depth'], NB_SAMPLES)
        sub.to_csv( submission_path, index=False, float_format='%8.5f' )

    return {'loss': test_loss, 'status': STATUS_OK}

if USE_HYPEROPT:
    log_writer = open('../data/xgb-hyperopt-log.txt', 'w')

    space = {
        # 'booster': hp.choice( 'booster',  ['dart', 'gbtree'] ),
        'max_depth': hp.quniform("max_depth", 4, 9, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 40, 1),
        'subsample': hp.uniform('subsample', 0.75, 1.0),
        # 'gamma': hp.uniform('gamma', 0.0, 0.5),
        'gamma': hp.loguniform('gamma', -5.0, 0.0),
        # 'eta': hp.uniform('eta', 0.005, 0.018),
        'eta': hp.loguniform('eta', -2.3, -1.2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.70, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.70, 1.0),
        # 'seed': hp.randint('seed', 2000000)
    }

    trials = Trials()
    best = hyperopt.fmin(fn=objective,
                         space=space,
                         algo=HYPEROPT_ALGO,
                         max_evals=N_HYPEROPT_PROBES,
                         trials=trials,
                         verbose=1)

    print('-' * 50)
    print('The best params:')
    print(best)
    print('\n\n')
# </editor-fold>


# <editor-fold desc="Обучение одиночной модели XGBoost">
if not USE_HYPEROPT:
    print('Start training...')

    xgb_params = {
        'booster': 'gbtree',
        'subsample': 1.0,
        'max_depth': MAX_DEPTH,
        'seed': 123456,
        'min_child_weight': 1.0,
        'eta': LEARNING_RATE,
        'gamma': 0.01,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'eval_metric': 'auc',  # 'auc',  # 'logloss',
        'objective': 'binary:logistic',
        'silent': 1,
        #'scale_pos_weight': 1.0 / np.mean(y_data)
    }

    # xgb_params['updater'] = 'grow_gpu'
    xgb_params['eval_metric'] = ['auc', 'logloss']
    model = xgboost.train(params=xgb_params,
                          dtrain=D_train,
                          num_boost_round=10000,
                          evals=watchlist,
                          verbose_eval=50,
                          early_stopping_rounds=50)

    print('nb_trees={} val_loss={:7.5f}'.format(model.best_ntree_limit, model.best_score))
    # loss = model.best_score
    nb_trees = model.best_ntree_limit
    y_pred = model.predict(D_val, ntree_limit=nb_trees)
    val_loss = sklearn.metrics.log_loss(y_val, y_pred)
    val_aucroc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    print('val ==> logloss={} auc_roc={}'.format(val_loss, val_aucroc))

    # -------------------------------------------------------------

    print('Computing submission...')
    y_submit = model.predict(D_submit, ntree_limit=nb_trees)

    sub = pd.DataFrame({'ID':ids,'click':y_submit})
    submission_path = '../submissions/xgb_loss={:<7.5f} aucroc={:<7.5f} [eta={} max_depth={} nb_samples={}].csv'.format(val_loss, val_aucroc, xgb_params['eta'], xgb_params['max_depth'], NB_SAMPLES)
    sub.to_csv( submission_path, index=False, float_format='%8.5f' )

    #with zipfile.ZipFile(submission_path+'.zip', 'w') as zip_name:
    #    zip_name.write(submission_path, submission_path )
# </editor-fold>



