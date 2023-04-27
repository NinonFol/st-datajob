#Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns
import io

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

from constantes import membres_groupe, import_data, introduction, distribution,train_model, pretraitement
df = import_data()
df = pretraitement(df)


#Titre du streamlit
# st.image('./assets/rapport_data.jpg')
st.title('Les métiers de la Data en 2020')

#Side bar
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
  

    #Affichage texte introduction
    st.markdown('---')
    st.markdown('## Introduction')
    st.markdown("<p style = 'text-align:justify;'>"+introduction()+"</p>",unsafe_allow_html=True)

    #Présentation des données
    st.markdown('---')
    st.markdown('## Présentation du jeu de données')
    st.caption('Extrait du jeu de données initial')
    st.dataframe(df.head())
    st.markdown(f'Le jeu de données est constitué de {df.shape[0]} lignes et de {df.shape[1]} variables.\nToutes les variables sont de type object et sont au format texte')
    
    #Affichage du DF info - utile ?
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    #affichage ou non des taux de Nan
    st.caption('Données manquantes')
    if st.checkbox("Taux de valeurs manquantes") : 
        st.dataframe(round(df.isna().mean(),2))

    # # Lister les étapes de preprocess + les rendres cochables 
    # st.markdown('---')
    # st.markdown('# Preprocessing')
    # df = pretraitement(df)
    # st.markdown('---')


    st.write('Mettre distribution après preprocessing des données --> sur DF nettoyé')
    #Distribution variable cible
    distrib = distribution(df)
    fig = plt.figure(figsize=(6,4))
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
    st.markdown('modélisation et résultats')

    #Sélection des catégories professionnelles
    if st.button('Sélectionner uniquement les catégories professionnelles'):
        df=df.drop(df.loc[df['Q5']=='Student'].index,axis=0)
        df=df.drop(df.loc[df['Q5']=='Currently not employed'].index,axis=0)
        df=df.drop(df.loc[df['Q5']=='Other'].index,axis=0)

    st.write(df.Q5.unique())
    
    


    # Select the target
    y = df['Q5']

    # Select the features
    X = df.drop('Q5', axis =1 )

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state =0)
    
    model_list = ['Logistic regression', 'Random Forest','Tree Classifier', 'SVC']
    model_choisi = st.selectbox(label = "Select a model" , options = model_list)

    #Afficher le score du modèle sélectionné
    st.write("Score test : .....")#, train_model(model_choisi, X_train, y_train, X_test, y_test))

    # Selection seuil V de Cramer
    seuil_cramer = st.slider("Choix du seuil de V de Cramer", min_value = 0.0, max_value = 1.0, step = 0.1)

    #Optimisation du modèle
    st.markdown('Optimisation avec Grid Search')
    st.write('Uniquement score avec les best param ou bien slicer.liste pour faire varier les hyperparamètres?')
    st.write('manque partie de non ré-entrainement du modèle')


#MODELISATION
#-------Sauvegarder modèle après entrainement puis chargement sur ST-------------
#-------Eviter de réentrainer le modèle------------------------------------------


# # Enregistrement du modèle (à faire après l'entraînement)
# dump(model, 'nom_du_fichier.joblib') 

# # Chargement du modèle (à faire sur l'app Streamlit)
# model = load('nom_du_fichier.joblib') 

#--------------------------------------------------------------------------------