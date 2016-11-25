import xgboost as xgb
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import KFold, train_test_split
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from xgb_regressor import XGBoostRegressor

class Stacker(object):
    def __init__(self, xgboost_func, mlp_func, train_path='train.csv', seed=0, test_size=0.25, **kwargs):
        self.seed = seed
        self.test_size = test_size
        self.xgboost_func = xgboost_func
        self.train_path = train_path
        self.mlp_func = mlp_func
        self.mlp_fit_kwargs = kwargs.get('mlp_fit_kwargs', {'nb_epoch': 30, 'batch_size': 128, 'verbose': 1})
        self.mlp_predict_kwargs = kwargs.get('mlp_predict_kwargs', {'batch_size': 256, 'verbose': 1})

    def stack_and_compare(self):
        xg_xtr, xg_xte, xg_ytr, xg_yte  = self.preprocess(encoding='label', transform_label=True)
        mlp_xtr, mlp_xte, mlp_ytr, mlp_yte = self.preprocess(encoding='one-hot', transform_label=False)
        
        assert mean_absolute_error(np.exp(xg_ytr), mlp_ytr) < 0.001 # Sanity check
        assert mean_absolute_error(np.exp(xg_yte), mlp_yte) < 0.001 # Sanity check

        xgb_folds = self.predict_folds(self.xgboost_func, xg_xtr, xg_ytr)
        mlp_folds = np.log(self.predict_folds(self.mlp_func, mlp_xtr, mlp_ytr, 
                           fit_kwargs=self.mlp_fit_kwargs, predict_kwargs=self.mlp_predict_kwargs))

        xgb_pred_hold = self.predict_holdout(self.xgboost_func, xg_xtr, xg_ytr, xg_xte)
        mlp_pred_hold = np.log(self.predict_holdout(self.mlp_func, mlp_xtr, mlp_ytr, mlp_xte, 
                               fit_kwargs=self.mlp_fit_kwargs, predict_kwargs=self.mlp_predict_kwargs))

        score_xgb, score_mlp = self.evaluate_estimators(xgb_pred_hold, mlp_pred_hold, xg_yte)
        print 'Single model performance:', 'xgb:', score_xgb, ',', 'mlp:', score_mlp

        stacker = self.stack(xgb_folds, mlp_folds, xg_ytr)
        score_stacker = self.evaluate_stacker(stacker, xgb_pred_hold, mlp_pred_hold, xg_yte)

        print {'xgb': score_xgb, 'mlp': score_mlp, 'stacker': score_stacker}
        return {'xgb': score_xgb, 'mlp': score_mlp, 'stacker': score_stacker}

    def preprocess(self, encoding='one-hot', transform_label=False):
        train = pd.read_csv(self.train_path)
        if transform_label: 
            train['loss'] = np.log(train['loss'])
        cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
        if encoding == 'one-hot':
            train = pd.get_dummies(data=train, columns=cat_features)
        elif encoding == 'label':
            for c in range(len(cat_features)): train[cat_features[c]] = train[cat_features[c]].astype('category').cat.codes
        else:
            raise Exception("Correct value of 'encoding' is required. Possible values of encoding=['one-hot', 'label']")
        features = [x for x in train.columns if x not in ['id','loss']]
        train_x = np.array(train[features])
        train_y = np.array(train['loss'])
        x_tr, x_te, y_tr, y_te = train_test_split(train_x, train_y, test_size=self.test_size, random_state=self.seed)
        return x_tr, x_te, y_tr, y_te

    def predict_folds(self, model_func, xtrain, ytrain, fit_kwargs={}, predict_kwargs={}):
        folds = KFold(len(ytrain), shuffle=False, n_folds=3)
        fold_preds = np.zeros(len(ytrain))
        for k, (train_index, test_index) in enumerate(folds):
            xtr = xtrain[train_index]
            ytr = ytrain[train_index]
            estimator = model_func(xtrain.shape[1])
            xte, yte = xtrain[test_index], ytrain[test_index]
            estimator.fit(xtr, ytr, **fit_kwargs)
            fold_preds[test_index] = estimator.predict(xte, **predict_kwargs)
        return fold_preds

    def predict_holdout(self, model_func, xtrain, ytrain, xtest, fit_kwargs={}, predict_kwargs={}):
        estimator = model_func(xtrain.shape[1])
        estimator.fit(xtrain, ytrain, **fit_kwargs)
        return estimator.predict(xtest, **predict_kwargs)

    def evaluate_estimators(self, xgb_pred, mlp_pred, test_y):
        return mean_absolute_error(np.exp(xgb_pred), np.exp(test_y)), mean_absolute_error(np.exp(mlp_pred), np.exp(test_y))

    def stack(self, xgb_oof, mlp_oof, oof_y):
        assert len(xgb_oof) == len(mlp_oof)
        oof_x = np.vstack((xgb_oof, mlp_oof)).T
        metaestimator = LinearRegression()
        metaestimator.fit(oof_x, oof_y)
        return metaestimator

    def evaluate_stacker(self, stacker, xgb_pred, mlp_pred, holdout_y):
        holdout_pred = np.hstack((xgb_pred.reshape(len(xgb_pred), 1), np.array(mlp_pred)))
        predictions = stacker.predict(holdout_pred)
        score = mean_absolute_error(np.exp(predictions), np.exp(holdout_y))
        return score