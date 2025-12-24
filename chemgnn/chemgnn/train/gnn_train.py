from chemgnn.models.nfp_gnn import build_nfp_gnn

def train_and_evaluate_gnn(
    preprocessor,
    train_ds,
    valid_ds,
    test_ds,
    y_test,
    num_features,
    layers,
    lr,
    epochs,
):
    model = build_nfp_gnn(preprocessor, num_features, layers, lr)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        verbose=0,
    )

    test_pred = model.predict(test_ds).squeeze()
    return model, test_pred

