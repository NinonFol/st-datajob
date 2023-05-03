

def obtain_target_features(df):
    # Select the target
    y = df['Q5']

    # Select the features
    X = df.drop('Q5', axis =1 )

    return X,y

def train_and_save_model(model, X, y, path_to_model='enreg_model.joblib'):
    from joblib import dump, load
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)
