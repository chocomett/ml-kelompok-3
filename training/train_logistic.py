from sklearn.linear_model import LogisticRegression


def train_logistic(X_train, y_train):
    """
    Melatih model Logistic Regression.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model