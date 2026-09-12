from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):
    """
    Melatih model Random Forest.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model