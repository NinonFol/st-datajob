
#Liste des membres du groupe
def membres_groupe():
    return ['Vincent Abdou Chacourou', 'Jys Seyler', 'Ninon Fol', 'Encadrés par Robin Trinh']

# Importation du jeu de données 2020
def import_data(link_file):
    import pandas as pd
    df = pd.read_csv(link_file, on_bad_lines='skip')
    return df

#Texte à renseigner en introduction du projet
def introduction():
    introduction = "test introduction"
    return introduction

#Graphe distribution de la variable cible
def distribution(df):
    distribution = df.Q5.value_counts()
    return distribution



#MODELISATION
def train_model(model_choisi, X_train, y_train, X_test, y_test) :
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import svm
    model = LogisticRegression() #modèle choisi par défaut
    if model_choisi == 'Logistic Regression' : 
        model = LogisticRegression()
    elif model_choisi == 'Random Forest' : 
        model = RandomForestRegressor()
    elif model_choisi == 'Decision Tree' : 
        model = DecisionTreeClassifier()
    elif model_choisi == 'KNN' : 
        model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    dico_result = {'Score Train':score_train,
                   'Score Test':score_test,
                   'Predictions':y_pred,
                   'Model':model}
    return dico_result
