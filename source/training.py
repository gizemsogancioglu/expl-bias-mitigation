from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
xgb_params = [{
    'max_depth': [5],
}]

mlp_params = [{
    'model__hidden_layer_sizes': [(100,)]
}]

svr_params = [{
    'model__kernel': ['linear']
}]

DATA_PATH = "../faircvtest/"


def blackbox_regressor(X_train, X_test, y_train, y_test):
    cv = 3

    grid = GridSearchCV(
        xgb.XGBRegressor(random_state=42), xgb_params, scoring="neg_mean_absolute_error",
        verbose=0, cv=cv)

    grid.fit(X_train, y_train)
    y_preds = grid.predict(X_test)
    print("Best score for grid estimators", grid.best_score_)

    print("Evaluation of model on test set: \n", mean_absolute_error(y_true=y_test, y_pred=y_preds))
    print("rounded ", round(mean_absolute_error(y_test, y_preds) * 100, 2))
    print("best params: ", grid.best_params_)
    return y_preds, grid.best_estimator_


def blackbox_classifier(X_train, X_test, y_train, y_test):
    cv = 3
    grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42), xgb_params, scoring="f1_macro",
        verbose=1, cv=cv)

    grid.fit(X_train, y_train)
    y_preds = grid.predict(X_test)
    print("Best score for grid estimators", grid.best_score_)
    # print("Best params for black-box clf: ", grid.best_params_)
    print("Evaluation of model on test set: \n", f1_score(y_true=y_test, y_pred=y_preds))

    return y_preds, grid.best_estimator_


def evaluate_reg_accuracy(y_test, y_preds):
    return round(mean_absolute_error(y_test, y_preds) * 100, 2)
