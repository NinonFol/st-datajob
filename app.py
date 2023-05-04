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

#Pour pouvoir charger un fichier >25mb sur git hub
A = import_data('df_2021_A.csv')
B = import_data('df_2021_B.csv')
df_2021 = pd.concat([A, B], axis = 0)

@st.cache_data  #éviter de recharger la page à chaque fois (gain de temsp)
def load_df(df_2020,df_2021):
    import random as rd
    
    df_2020 = ordonner_colonnes(df_2020)
    df_2020.Q24=df_2020.Q24.replace(['$0-999'],['0-999']) #Parti pris de changer la catégorie 499999

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

    return df_union
    
    
    #Retraitement à appliquer au df complet
@st.cache_data
def pretrait_encodage_reduction(df_union):
    df = encodage(pretraitement(df_union))
    df = reduction(df)
    return df

#Appliquer les transformations aux df importés
df_union = load_df(df_2020,df_2021)
df = pretrait_encodage_reduction(df_union)


#Titre du streamlit
# st.image('./assets/rapport_data.jpg')
st.image('illu.png')
st.title('PROJET DATAJOB ')
st.title('Les métiers de la Data en 2020')
#Side bar---------------------------------------------------------------------------
with st.sidebar:
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

   
elif radio_btn == 'Visualisation':
    st.markdown('graphes à afficher')

    #Distribution variable cible
    selector = st.checkbox('Distribution sur le jeu de données initial')
    distrib = distribution(df)
    if selector:
        distrib = distribution(df_union)
    fig = plt.figure(figsize=(4,2))
    sns.barplot(y=distrib.values,x=distrib.index, color='blue')
    plt.title('Répartition des classes de la variable cible', fontsize=6)
    plt.xticks(rotation=90,fontsize=4)
    plt.yticks(fontsize=4)
    st.write(fig)

    st.write('Observations: \n \
- Surreprésentation des étudiants \n \
- Part non-négligeable de personnes sans emploi ou appartenant à la catégorie Other')

    #Répartition des missions par métier
    normalise = repartion_missions(df_union)
    fig2 = plt.figure(figsize=(4,2))
    normalise.plot.bar(stacked=True,edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 0.2, 0.5, 0.5),fontsize=8,frameon=False)
    plt.title('Répartition des missions par métier',fontsize=6)
    plt.xticks(rotation=90,fontsize=4)
    plt.yticks(fontsize=4)
    st.pyplot(fig2)

    st.write('Observations: \n \
- Les “research scientist” passent 2x plus de temps à faire de la recherche sur du ML  \n\
- Les “business analyst”, “data analyst” passent 2x plus de temps à l’analyse et compréhension des data pour influencer des décisions  \n\
- Les “data scientist” et “machine learning engineer” passent 2x plus de temps à la construction et l’exécution d’un service de ML')

    #Utilisation des méthodes de machine learning
    fig3 = plt.figure(figsize=(4,2))
    st.write(fig3)

    st.write('Forte corrélation entre la méthode de ML utilisée et le métier data (cf métier de Research Scientist versus Software engineer)')




else :

    #Split features and target
    X,y = obtain_target_features(df)
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)

    #Simulation
    radio_btn = st.radio("Simulation", options=('1','2'))
    if radio_btn == '1':
        simu_1_df = load_df(df_2020,df_2021)
        simu_1 = pretrait_encodage_reduction(simu_1_df)
        simu_1 = pd.DataFrame(simu_1.loc[13691]).T

        values7 =['Python']
        values23 =['Analyze and understand data to influence product or business decisions',
                        ' Build and/or run the data infrastructure that my business uses for storing',
                        ' Experimentation and iteration to improve existing ML models']
        
        X_test, y_test = obtain_target_features(simu_1)

    if radio_btn == '2':
        simu_2_df = load_df(df_2020,df_2021)
        simu_2 = pretrait_encodage_reduction(simu_2_df)
        simu_2 = pd.DataFrame(simu_2.loc[33885]).T

        values7 =['Python','C++','Java Javascript','MATLAB']
        values23 =[' None of these activities are an important part of my role at work']

        X_test, y_test = obtain_target_features(simu_2)  

    # Label Encodage de la variable cible (sinon ne fonctionne pas)
    le=LabelEncoder()
    y_train=le.fit_transform(y_train)
    y_test=le.transform(y_test)

    #Instancier le modèle
    model = LogisticRegression()
    train_and_save_model(model,X_train,y_train)

    #Charger le model de reglog
    model = load('enreg_model.joblib')
    
    y_pred = model.predict(X_test)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test,y_test)

    #Désencoder la variable cible
    y_test = le.inverse_transform(y_test)  
    y_pred = le.inverse_transform(y_pred)
    y_train = le.inverse_transform(y_train)

    st.markdown('<h1 style = "text-align : center";>Questionnaire</h1>',unsafe_allow_html=True)
    with st.form('Form1'):      
        radio_country = st.radio('In wich country do you reside?',options=('Italy','Russia','Canada'))
        radio_years = st.radio('For how many years have you been writing code?',options=('10-20 years','5-10 years','3-5 years'))
        radio_compensation = st.radio('What is your current yearly compensation (approximate $USD)?',options=('40,000-49,999','50,000-59,999','70,000-79,999'))
        
        if radio_btn == '1':
            radio_country = 'Russia'
            radio_years = '10-20 years'
            radio_compensation = '50,000-59,999'
        if radio_btn == '2':
            radio_country = 'Canada'
            radio_years = '5-10 years'
            radio_compensation = '70,000-79,999'

        st.multiselect('What programming languages do you use on a regular basis? (Select all that apply)',
                       options=('Python',
                                'R',
                                'SQL',
                                'C',
                                'C++',
                                'Java Javascript',
                                'Julia',
                                'Swift',
                                'Bash',
                                'MATLAB',
                                'None',
                                'Other'),
                                default=values7)
        
        st.multiselect('Select any activities that make up an important part of your role at work: (Select all that apply)',
                        default = values23,
                        options=('Analyze and understand data to influence product or business decisions',
                        ' Build and/or run the data infrastructure that my business uses for storing',
                        ' analyzing',
                        ' and operationalizing data',
                        ' Build prototypes to explore applying machine learning to new areas',
                        ' Build and/or run a machine learning service that operationally improves my product or workflows',
                        ' Experimentation and iteration to improve existing ML models',
                        ' Do research that advances the state of the art of machine learning',
                        ' None of these activities are an important part of my role at work',
                        ' Other'))
        
        submitted = st.form_submit_button("Afficher le métier conseillé")
        if submitted:

            st.markdown(f"*Vos compétences sont celles d'un {y_pred[0]} selons les prédictions*")
            
    resultats = pd.concat([pd.DataFrame(y_pred,columns=['predicts']),pd.DataFrame(y_test,columns=['test'])],axis=1)

    if st.checkbox('Afficher les scores et la valeur réelle'):
        st.write(f'Score du jeu de test : {score_test}')
        st.write(f"Score du jeu d'entrainement : {score_train}")
        st.write(resultats)

    st.markdown("Conclusion : \n \
Utilité du modèle : \n \
- Les 4 métiers ciblés au fur et à mesure de l’étude sont les mieux prédits : Data Analyst, Data Scientist, Research Scientist et Software Engineer \n \
- Pour les autres métiers data, le résultat de la modélisation est plus aléatoire, d’où l’ajout d’un disclaimer invitant le répondant à valider le résultat par des ressources complémentaires \n \
Suggestions pour améliorer : \n \
- La qualité du dataframe : rendre des questions obligatoires, intégrer les questionnaires des autres années et des données externes (Glassdoor…) \n \
- L’analyse du dataframe : appliquer d’autres modèles de classification, comparer les résultats du modèle à la réalité du marché de l’emploi data")
    
