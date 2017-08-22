# -*- coding: utf-8 -*-
'''
Соревнование:
https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-3/machine-learning/predict-ad-clicks/

(c) Koziev Elijah inkoziev@gmail.com

Подбор гиперпараметров для LightGBM с помощью hyperopt.

Справка по LightGBM:
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#lightgbm-package

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''


from __future__ import print_function
import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics
import zipfile
import collections
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama


NB_SAMPLES = 1500000
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

X_train, X_val, y_train, y_val = train_test_split(X_data, trainY, test_size = 0.5)

D_train = lightgbm.Dataset(X_train, y_train)
D_val = lightgbm.Dataset(X_val, y_val)

# ---------------------------------------------------------------
# </editor-fold>


# <editor-fold desc="hyperopt + LightGBM">

def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['application'] = 'binary'
    lgb_params['metric'] = 'auc'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = space['max_depth'] if 'max_depth' in space else -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

    return lgb_params


log_writer = None
obj_call_count = 0
cur_best_loss = np.inf

def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nLightGBM objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = lightgbm.train(lgb_params,
                           D_train,
                           num_boost_round=100,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=100,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )

    nb_trees = model.best_iteration
    y_pred = model.predict(X_val, num_iteration=nb_trees)
    test_loss = sklearn.metrics.log_loss(y_val, y_pred)
    test_aucroc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    print('test_loss={:<7.5f} test_aucroc={:<7.5f}'.format(test_loss, test_aucroc))

    log_writer.write('loss={:<7.5f} aucroc={:<7.5f} Params:{} nb_trees={}\n'.format(test_loss, test_aucroc, params_str, nb_trees))
    log_writer.flush()

    if test_loss < cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)
        print('Computing submission...')
        y_submit = model.predict(X_submit, num_iteration=nb_trees)

        sub = pd.DataFrame({'ID':ids,'click':y_submit})
        submission_path = '../submissions/lgb_loss={:<7.5f} aucroc={:<7.5f} [eta={} max_depth={} nb_samples={}].csv'.format(test_loss, test_aucroc, xgb_params['eta'], xgb_params['max_depth'], NB_SAMPLES)
        sub.to_csv( submission_path, index=False, float_format='%8.5f' )

    return {'loss': test_loss, 'status': STATUS_OK}

if USE_HYPEROPT:
    log_writer = open('../data/lgb-hyperopt-log.txt', 'w')

    space = {
        'num_leaves': hp.quniform('num_leaves', 10, 200, 1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 1000, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -3.0, -1.2),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform('max_bin', 64, 512, 1),
        'bagging_freq': hp.quniform('bagging_freq', 1, 5, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10),
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


# <editor-fold desc="Обучение одиночной модели LightGBM">
if not USE_HYPEROPT:
    print('Start training...')

    lgb_params = {
        'num_leaves': 100,
        'min_data_in_leaf': 20,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.95,
        'learning_rate': 0.13,
        'min_sum_hessian_in_leaf': 1,
        'max_bin': 256,
        'bagging_freq': 1,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_depth': 10
                }

    # params = {
    #     'num_leaves' : 256,
    #     'learning_rate':0.03,
    #     'metric':'auc',
    #     'objective':'binary',
    #     'early_stopping_round': 40,
    #     'max_depth':10,
    #     'bagging_fraction':0.5,
    #     'feature_fraction':0.6,
    #     'bagging_seed':2017,
    #     'feature_fraction_seed':2017,
    #     'verbose' : 1
    # }


    lgb_params = get_lgb_params( lgb_params )
    model = lightgbm.train(lgb_params,
                           D_train,
                           num_boost_round=10000,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=100,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )

    nb_trees = model.best_iteration
    y_pred = model.predict(X_val, num_iteration=nb_trees)
    test_loss = sklearn.metrics.log_loss(y_val, y_pred)
    test_aucroc = sklearn.metrics.roc_auc_score(y_val, y_pred)
    print('test_loss={:<7.5f} test_aucroc={:<7.5f} nb_trees={}'.format(test_loss, test_aucroc, nb_trees))

    print('Compute submission using {} trees...'.format(nb_trees))
    y_submit = model.predict(X_submit, num_iteration=nb_trees)
    sub = pd.DataFrame({'ID':ids,'click':y_submit})
    submission_path = '../submissions/lgb_loss={:<7.5f} aucroc={:<7.5f} [lr={} nb_samples={}].csv'.format(test_loss, test_aucroc, lgb_params['learning_rate'], NB_SAMPLES)
    sub.to_csv( submission_path, index=False, float_format='%8.5f' )

    #with zipfile.ZipFile(submission_path+'.zip', 'w') as zip_name:
    #    zip_name.write(submission_path, submission_path )
# </editor-fold>



