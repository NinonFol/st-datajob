#Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns
import io

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

from constantes import membres_groupe, import_data, introduction, distribution
from preprocessing_df import pretraitement, ordonner_colonnes, encodage, reduction
from model_ml import obtain_target_features, train_and_save_model

#Importer les données 2020 et 2021
df_2020 = import_data('kaggle_survey_2020_responses.csv')
df_2020 = ordonner_colonnes(df_2020)
df_2020.Q24=df_2020.Q24.replace(['$0-999'],['0-999']) #Parti pris de changer la catégorie 499999

df_2021 = import_data('df_2021.csv')
df_2021 = df_2021.drop(0,axis=0) #supprimer les questions de df 2021
df_2021.drop(df_2021.columns[[0,]], axis=1, inplace=True)

df_2021.Q24=df_2021.Q24.replace(['> $500,000','$0-999','>$1,000,000','$500,000-999,999','300,000-499,999'],
                                ['500,000','0-999','> $500,000','> $500,000','300,000-500,000'])  #Pour éviter les erreurs lors du traitement des Nan salaires
                                #pour les deux dernier montants -> regroupés comme dans 2020 pour plus de facilité

df_2021.Q4=df_2021.Q4.replace(['Bachelor\x19s degree', 'Master\x19s degree','Some college/university study without earning a bachelor\x19s degree'],
                                ['Bachelor’s degree', 'Master’s degree','Some college/university study without earning a bachelor’s degree'])


#Concaténer les df
df_union = pd.concat([df_2020, df_2021], axis = 0)
df_union = df_union.reset_index().drop('index',axis=1) #Pour éviter les erreurs lors du traitement des Nan salaires
df_union = df_union.drop(df_union.columns[[-1,]], axis=1) #Droper la colonne Unnamed

#Retraitement à appliquer au df complet
df = encodage(pretraitement(df_union))
# df = reduction(df)

#Split features and target
X,y = obtain_target_features(df)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)

# Label Encodage de la variable cible (sinon ne fonctionne pas)
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

#Instancier le modèle
model = LogisticRegression()
train_and_save_model(model,X,y)







#Titre du streamlit
# st.image('./assets/rapport_data.jpg')
st.title('PROJET DATAJOB - Les métiers de la Data en 2020')

#Side bar---------------------------------------------------------------------------
with st.sidebar:
    st.write("Sélection la section de votre choix")
    radio_btn = st.radio("",
                     options=('Présentation','Visualisation','Modélisation'))
    #  Affichage membres du groupe
    st.markdown('---')
    st.caption('Equipe projet')
    s = ''
    for i in membres_groupe():
        s += "- " + i + "\n"
    st.caption(s)
if radio_btn == 'Présentation':
 #---------------------------------------------------------------------------------- 

    #Affichage texte introduction
    st.markdown('---')
    st.markdown('## Introduction')
    st.markdown("<p style = 'text-align:justify;'>"+introduction()+"</p>",unsafe_allow_html=True)

    #Présentation des données
    st.markdown('---')
    st.markdown('## Présentation du jeu de données')
    if st.checkbox("Extrait du jeu de données initial") : 
        st.dataframe(df_union.head())
            #affichage ou non des taux de Nan
        # if st.checkbox("Afficher les taux de valeurs manquantes pour chaque variable") : 
        #     st.dataframe(round(df.isna().mean(),2))
    st.markdown(f'Le jeu de données est constitué de {df_union.shape[0]} lignes et de {df_union.shape[1]} variables.\nToutes les variables sont de type object et sont au format texte')
    

    # Affichage du DF après deuxième retraitement
    st.markdown('Après deuxième retraitement')
    if st.checkbox("Extrait du jeu de données après 2ème retraitement et encodage des données") : 
        st.dataframe(df.head())
        st.text(f'Shape: {df.shape[0]} lignes, {df.shape[1]} colonnes')

    #Distribution variable cible
    distrib = distribution(df)
    fig = plt.figure(figsize=(4,2))
    sns.barplot(y=distrib.values,x=distrib.index)
    plt.title('Répartition des classes de la variable cible')
    plt.xticks(rotation=90)
    st.write(fig)
    
elif radio_btn == 'Visualisation':
    st.markdown('graphes à afficher')
    fig = plt.figure()
    #*****
    st.write(fig)
else :
    st.markdown('## Modélisation et résultats')

    model = load('enreg_model.joblib') 
    st.write(model.score(X_test,y_test))
    st.write(model.predict(X_test)) #y_pred
    st.write(model.score(X_train, y_train))
