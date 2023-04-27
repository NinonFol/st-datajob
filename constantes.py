
#Liste des membres du groupe
def membres_groupe():
    return ['Vincent Abdou Chacourou', 'Jys Seyler', 'Ninon Fol', 'Encadrés par Robin Trinh']

# Importation du jeu de données 2020
def import_data():
    import pandas as pd
    df = pd.read_csv('C:/Users/ninon/Documents/2-DataScientest/3_Projet_Datajob/Data/streamlit_app/tabs/kaggle_survey_2020_responses.csv')
    return df

#Texte à renseigner en introduction du projet
def introduction():
    introduction = "test introduction"
    return introduction

#Graphe distribution de la variable cible
def distribution(df):
    distribution = df.Q5.value_counts()
    return distribution

#Preprocessing du DF
def pretraitement(df):
    import pandas as pd
    questions=pd.DataFrame(df.loc[0,:]).T
    df=df.iloc[1:,:]
    #Convertir la colonne des temps de réponse en nombres
    df['Time from Start to Finish (seconds)']=df['Time from Start to Finish (seconds)'].astype(float)

    #Filtrage des temps de réponse au questionnaire : garder uniquement les valeurs entre q1 et q3
    q1 = df['Time from Start to Finish (seconds)'].quantile(0.25)
    q3 = df['Time from Start to Finish (seconds)'].quantile(0.75)
    iqr = q3-q1
    low = q1 - 3*(iqr)
    high = q3 + 3*(iqr)
    df = df[df['Time from Start to Finish (seconds)'].between(low, high)]

    # Suppression de la colonne 'Duration'
    df=df.drop('Time from Start to Finish (seconds)',axis=1)

    #suppression des lignes vides de Q5
    df=df.dropna(subset=['Q5']) 

    # conserver uniquement lignes avec moins de 90% de nan
    df=df[df.isna().mean(axis=1)<0.90]




    return df

#MODELISATION

def train_model(model_choisi, X_train, y_train, X_test, y_test) :
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import svm
    if model_choisi == 'Random Forest' : 
        model = RandomForestRegressor()
    elif model_choisi == 'Decision Tree' : 
        model = DecisionTreeClassifier()
    elif model_choisi == 'KNN' : 
        model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score
