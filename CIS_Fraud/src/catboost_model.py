from catboost import CatBoostClassifier

# Train and Evaluate CatBoost
def train_catboost(X_train, y_train, X_val, y_val):
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        verbose=100,
        early_stopping_rounds=100
    )

    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
    return cat_model