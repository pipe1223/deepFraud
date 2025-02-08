import xgboost as xgb


def train_xgboost(X_train, y_train, X_val, y_val):
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1000,
        verbosity=0
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    return xgb_model