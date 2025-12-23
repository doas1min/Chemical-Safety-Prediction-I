from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

def cross_validate_rf(model, X, y, n_splits=5):
    cv = KFold(n_splits=n_splits)
    return cross_val_score(model, X, y, scoring="r2", cv=cv)

def grid_search_rf(model, param_grid, X, y, cv=5):
    gs = GridSearchCV(model, param_grid, cv=cv, scoring="r2")
    gs.fit(X, y)
    return gs

def train_and_evaluate_rf(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }
