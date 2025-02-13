import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import pandas as pd

class LightGbmSnoop:
    def __init__(self):
        self.train_logs = []
        self.valid_logs = []
    def _callback(self, env):
        self.model = env.model
        self.train_logs.append( [b.eval_train()[0][2] for b in self.model.boosters] )
        self.valid_logs.append( [b.eval_valid()[0][2] for b in self.model.boosters] )
    def train_log(self):
        return pd.DataFrame(self.train_logs).add_prefix('train_')
    def valid_log(self):
        return pd.DataFrame(self.valid_logs).add_prefix('valid_')
    def logs(self):
        return pd.concat((self.train_log(), self.valid_log()), 1)
    def get_oof(self, n):
        oof = np.zeros(n, dtype=float)
        for i, b in enumerate(self.model.boosters):
            vs = b.valid_sets[0]  # validation data
            idx = vs.used_indices
            # Note: this uses all trees, not the early stopping peak count.
            # You can use b.rollback_one_iter() to drop trees :)
            p = b._Booster__inner_predict(1) # 0 = train; 1 = valid
            oof[idx] = p
        return oof



def train_lightgbm(X, y):
    params = {
        'num_leaves': 64,
        'objective': 'binary',
        'min_data_in_leaf': 10,
        'learning_rate': 0.01,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'max_cat_to_onehot': 128,
        'metric': 'auc',
        'num_threads': 8,
        'verbose': -1,
        'n_estimators': 5000,
    }
    
    folds = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(X, y))
    ds = lgb.Dataset(X, y, params=params)
    s = LightGbmSnoop()
    lgb_model = lgb.cv(
        params,
        ds,
        folds=folds,
        num_boost_round=1000,
        callbacks=[
            s._callback,
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    return lgb_model, s