import streamlit as st
import pandas as pd
#Liste des membres du groupe
def membres_groupe():
    return ['Vincent Abdou Chacourou', 'Jys Seyler', 'Ninon Fol', 'Encadrés par Robin Trinh']

#Texte à renseigner en introduction du projet
def introduction():
    introduction = "Data Scientist, Data Analyst, Machine Learning Engineer, Statisticien…  \
Nombreux sont les métiers de la data apparus au cours des dernières années. Ces derniers sont de plus en plus plébiscités par les entreprises qui cherchent à faire parler leurs données de plus en plus nombreuses.\n \
A la nouveauté de ces métiers, s’ajoute la rapidité de développement du secteur, qui fait qu’il est difficile de définir quelles sont exactement les spécialités de chacun, quels sont les outils et les compétences de chacun : le périmètre de chacun de ces métiers évolue, et en particulier les outils utilisés? \n \
C’est ce qu’explique Arnold Haine, co-président de la commission data de Syntec Conseil : « La constitution très rapide du secteur a eu comme principal effet pervers qu'un même métier soit compris et perçu de manière différente selon les entreprises », \
Il donne l’exemple du métier de « data scientist » : « pour certaines entreprises c’est un statisticien, alors que pour d'autres, c'est un homme à tout faire de la data. » (https://start.lesechos.fr/travailler-mieux/classements/les-10-metiers-les-plus-prometteurs-du-monde-de-la-data-1248702) \n \
Ce projet “Data Job” est donc l’occasion de se pencher sur les spécificités de chacun des profils présents dans le paysage de la sphère “data”, et de permettre aux personnes intéressées par le domaine d’avoir une vision plus claire des outils et compétences à avoir en fonction du métier visé. \n\
Pour établir cette cartographie, nous avons utilisé les données issues des réponses au questionnaire annuel proposé aux utilisateurs du site Kaggle.\n \
Ces répondants sont issus de tous les pays, métiers, horizons… Afin d’affiner notre étude sur un échantillon plus homogène, nous avons choisi de nous concentrer sur les profils étant en poste au moment de répondre à l’enquête et sur une tranche d’âge allant de 30 à 44 ans, c’est-à-dire le cœur de la population active. On imagine que le modèle ainsi créé permettra d’aider les personnes actives qui souhaitent se reconvertir vers un métier de la data à identifier le métier le plus pertinent selon les compétences et outils maîtrisés à ce stade (ou qu’elles souhaitent maîtriser d’ici à leur reconversion)."   

#pd.read_csv("C:/Users/ninon/Documents/2-DataScientest/3_Projet_Datajob/Data/streamlit_app/tabs/introduction.txt")
    
    return introduction

#Graphe distribution de la variable cible

def distribution(df):
    distribution = df.Q5.value_counts()
    return distribution

def repartion_missions(df):
    #Graphe de répartition des missions par métier
    #1)Regrouper Q5 + sous-questions 
    taches=[i for i in df.columns if 'Q23' in i]
    taches.insert(0,'Q5')
    taches=df.loc[1:,taches]
    new_names=[]
    for col in taches.columns:
        new_names.append(taches[col].unique()[1])
        
    new_names[0]='Métier'
    taches.columns=new_names
    #2)Faire un crosstab croisant chaque métier avec chaque mission (1 crosstab par mission)
    count_taches=pd.DataFrame()
    for col in taches.columns[1:]:
        tcd=pd.crosstab(taches['Métier'],taches[col])
        count_taches[col]=tcd

    #3)Normaliser chaque valeur (ne fonctionne pas avec normalize=True dans le crosstab)   
    #Total de la ligne
    total_ligne={}
    for index in count_taches.index:
        total_ligne[index]=count_taches.loc[index,:].sum()

    #Diviser chaque valeur par total
    normalise={}
    for i in total_ligne.keys():
        ajout=[]
        for j in count_taches.columns:
            ajout.append(count_taches.loc[i,j]/total_ligne[i])
        normalise[i]=ajout
    normalise=pd.DataFrame(normalise,index=count_taches.columns).T

    return normalise
