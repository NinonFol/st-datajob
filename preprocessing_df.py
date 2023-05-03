
#Premier retraitement - Mettre les colonnes de df dans l'ordre
def ordonner_colonnes(df):
    import pandas as pd
    p1=df.iloc[:,:132]
    p2=df.iloc[:,132:144]
    p3=df.iloc[:,144:155]
    p4=df.iloc[:,155:173]
    p5=df.iloc[:,173:189]
    p6=df.iloc[:,189:198]
    p7=df.iloc[:,198:210]
    p8=df.iloc[:,210:221]
    p9=df.iloc[:,221:256]

    s1=df.iloc[:,256:268]
    s2=df.iloc[:,268:280]
    s3=df.iloc[:,280:291]
    s4=df.iloc[:,291:309]
    s5=df.iloc[:,309:324]
    s6=df.iloc[:,324:332]
    s7=df.iloc[:,332:344]
    s8=df.iloc[:,344:]

    df=pd.concat([p1,s1,p2,s2,p3,s3,p4,s4,p5,s5,p6,s6,p7,s7,p8,s8,p9],axis=1)
    return df

#Deuxième retraitement DF
def pretraitement(df):
    import pandas as pd
    import numpy as np
    # questions=pd.DataFrame(df.loc[0,:]).T
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

    #Conserver que les 4 classes les mieux prédites
    df = df[(df.Q5=='Data Scientist')|
        (df.Q5=='Data Analyst')|
        (df.Q5=='Software Engineer')|
        (df.Q5=='Research Scientist')]

    #conserver que les professionnels
    df=df.drop(df.loc[df['Q5']=='Student'].index,axis=0)
    df=df.drop(df.loc[df['Q5']=='Currently not employed'].index,axis=0)
    df=df.drop(df.loc[df['Q5']=='Other'].index,axis=0)

    #Conserver que les tranches d'âges les plus employables
    df=df[df['Q1'].isin(['30-34','35-39','40-44'])]


    # conserver uniquement lignes avec moins de 90% de nan
    df=df[df.isna().mean(axis=1)<0.90]

    #Traitement nan de la variable salaire
    # Pour éviter d'avoir les messages warning
    import warnings
    warnings.filterwarnings('ignore')
    

    l=df['Q24'].dropna().unique()
    df_temp=pd.DataFrame(l,columns=['range'])


    df_temp['A']=[j.split('-')[0].replace(",","") for j in df_temp['range']]

    df_temp['A'][df_temp['A']=='$0']=0
    df_temp['A'][df_temp['A']=='> $500000']=500000

    df_temp['A']=df_temp['A'].astype('int')
    df_temp=df_temp.sort_values(by='A')
    df_temp=df_temp.reset_index(drop=True)

    df_temp['B']=0
    df_temp['B'][0:24]=[j.split('-')[1].replace(",","") for j in df_temp['range'][0:24]]
    df_temp['B']=df_temp['B'].astype('int')

    df_temp['Mean']=0
    df_temp['Mean'][0:24]=[np.mean([i,j]) for i,j in zip(df_temp['A'][0:24],df_temp['B'][0:24])]
    df_temp['Mean'][df_temp['A']==500000]=500000

    #Définir la fonction qui renvoie moyenne par tranche de salaire
    def mean_salaire(x):        
        return df_temp[df_temp['range']==x]['Mean'].iloc[0]

    #Définir la fonction qui renvoie tranche de salaire par nombre
    def give_range(x):

        for i in df_temp.index:
            if x <= df_temp.loc[i,'B']:
                return df_temp.loc[i,'range']

    #Comptage des moyennes
    tab=df.loc[1:,['Q5','Q24']].dropna(subset=['Q24'])
    tab['avg']=[mean_salaire(j) for j in tab['Q24']]

    avg=tab.fillna('Missing').groupby(['Q5']).agg(['mean','median','count'])
    avg['NaN']=[df[df['Q5']==i]['Q24'].isna().sum() for i in avg.index]

    #Définir la fonction qui renvoie tranche de salaire par nombre
    def give_range(x):

        for i in df_temp.index:
            if x <= df_temp.loc[i,'B']:
                return df_temp.loc[i,'range']

    #Obtenir la tranche à partir de la moyenne
    avg['tranche']=[give_range(avg.iloc[j,0:1][0]) for j in range(len(avg.iloc[0:,0:1]))]
    #avg

    #Définir la fonction qui renvoie tranche de salaire par métier
    def job_range(x):
        for i,j in zip(avg.index,avg.tranche):
            if x==i:
                return j

    ###Remplacer les valeurs manquantes
    #Attribuer à chaque métier sans salaire, la moyenne de sa tranche
    l=[]
    for i,j in zip(df['Q5'],df['Q24']):
        if pd.isna(j)==True:
            l.append(job_range(i))
        if pd.isna(j)==False:
            l.append(j)

    df['Q24']=l


    #---Encodage 0/1 de toutes les questions QCM du DF---
    #Ne conserver que les intitulés de questions au format QCM (conserve aussi si QA ou QB) : 
    qcm = []
    for i in [i.split('_') for i in df.columns]:
        if len(i)>1:  
            if i[1] == 'A' or i[1] == 'B':
                qcm.append(i[0]+'_'+i[1])
            else :
                qcm.append(i[0])
    qcm = list(set(qcm))   #Supprimer doublons (set) + convertir en liste (list)  
    for question in qcm:
        for i in [i for i in df.columns if question in i]:
            col_name = f'{question}_{df[i].value_counts().index[0]}'   #Colonnes nommées 'Qxx_A/B_Choix'
            v=df[i].value_counts().index[0]
            
            
            if len(df[i].unique())>2:
                v2=df[i].value_counts().index[1]
                df=df.rename({i:col_name},axis=1)
                df[col_name]=df[col_name].replace({v:1,np.nan:0,v2:2})    #Encodage 0/1
            else : 
                df=df.rename({i:col_name},axis=1)
                df[col_name]=df[col_name].replace({v:1,np.nan:0})    #Encodage 0/1

    # #Renommer les colonnes avec la valeur unique de la colonne et remplacer les valeurs par 0/1:
    # for question in qcm:
    #     for i in [i for i in df.columns if question in i]:
    #         col_name = f'{question}_{df[i].value_counts().index[0]}'   #Colonnes nommées 'Qxx_A/B_Choix'
    #         v=df[i].value_counts().index[0]
    #         df=df.rename({i:col_name},axis=1)
    #         df[col_name]=df[col_name].replace({v:1,np.nan:0})    #Encodage 0/1
    
    #---Traitement des Nan - Questions nominales choix unique du DF---
    df=df.fillna({'Q8':'None','Q11':'None','Q13':'Never', 'Q15':'I do not use machine learning methods',
                       'Q30':'None','Q32':'None','Q38':'Other',
                       'Q22':'I do not know'})

    #Séparer variables à distribution ordinales des autres
    ordinals=['Q20','Q21','Q22','Q24','Q25']

    ###Remplacer les NaN par le mode du métier pour Taille entreprise, Nombre de respo ML, Budget ML ('Q20','Q21','Q25')
    # l=['Data Engineer', 'Software Engineer', 'Data Scientist','Data Analyst', 'Research Scientist', 'Statistician','Product/Project Manager', 'Machine Learning Engineer','Business Analyst', 'DBA/Database Engineer']
    l=df.Q5.unique() #plus flexible quand on ne garde que les classes les plus représentées
    
    #Pour chacune des trois colonnes
    for i in ['Q20','Q21','Q25']:
        #Définir un dictionnaire temporaire
        d={}
        #Clé : métier / Valeur : Mode du métier pour la question concernée
        for j in l:
            d[j]= df.loc[0:,i][df.loc[1:,'Q5']==j].dropna().mode()[0]
        

        #Pour chaque valeur de la colonne à fillna
        #Liste vide
        lt=[]
        for k,m in zip(df.loc[0:,i],df.loc[0:,'Q5']):
            #Ne rien faire si non-nulle
            if pd.isna(k)==False:
                lt.append(k)
            #Attribuer le mode du métier sinon
            else:
                lt.append(d[m])
                
        df.loc[0:,i]=lt

    return df

#Encodage des données
def encodage(df):
    
        ############################################One hot encoding des colonnes ['Q2','Q3','Q8', 'Q11','Q30','Q32','Q38']
    #Pourc chaque colonne
    for i in ['Q2','Q3','Q8', 'Q11','Q30','Q32','Q38']:
        #Pour chacune des modalités de la colonne
        for j in df[i].value_counts().index:
            #Instancier nom de la colonne dédiée à la modalité
            name=(str(i)+'_'+str(j).replace(" ","_"))
            l=[]
            #OHE
            for k in df[i]:
                if k==j:
                    l.append(1)
                else:
                    l.append(0)
                    
            df[name]=l

    #Suppression des colonnes encodées        
    df=df.drop(['Q2','Q3','Q8', 'Q11','Q30','Q32','Q38'],axis=1)

    ############################################Ordinal encoding pour ['Q1','Q4','Q6','Q13','Q15','Q20','Q21','Q22','Q24','Q25']
    #Ordinal encoding Q1#########################
    #df['Q1'].unique()

    d={}

    d['18-21']=0
    d['22-24']=1
    d['25-29']=2
    d['30-34']=3
    d['35-39']=4
    d['40-44']=5
    d['45-49']=6
    d['50-54']=7
    d['55-59']=8
    d['60-69']=9
    d['70+']=10


    df.loc[0:,'Q1']=[d[j] for j in df.loc[0:,'Q1']]
    #df.loc[0:,'Q1']

    #Ordinal encoding Q4#########################
    #df['Q4'].unique()
    d={}

    d['I prefer not to answer']=0
    d['No formal education past high school']=1
    d['Some college/university study without earning a bachelor’s degree']=2
    d['Professional degree']=3
    d['Bachelor’s degree']=4
    d['Master’s degree']=5
    d['Doctoral degree']=6
    d['Professional doctorate']=7

    df.loc[0:,'Q4']=[d[j] for j in df.loc[0:,'Q4']]
    #df.loc[0:,'Q4']

    #Ordinal encoding Q6#########################
    #df['Q6'].unique()

    d={}

    d['I have never written code']=0
    d['< 1 years']=1
    d['1-2 years']=2
    d['1-3 years']=3   #Rajouté suite ajout 2021
    d['3-5 years']=4
    d['5-10 years']=5
    d['10-20 years']=6
    d['20+ years']=7

    df.loc[0:,'Q6']=[d[j] for j in df.loc[0:,'Q6']]
    #df.loc[0:,'Q6']



    #Ordinal encoding Q13#########################
    #df['Q13'].unique()

    d={}


    d['Never']=0
    d['Once']=1
    d['2-5 times']=2
    d['6-25 times']=3
    d['More than 25 times']=4

    df.loc[0:,'Q13']=[d[j] for j in df.loc[0:,'Q13']]
    #df.loc[0:,'Q13']


    #Ordinal encoding Q15#########################
    #df['Q15'].unique()
    d={}

    d['I do not use machine learning methods']=0
    d['Under 1 year']=1
    d['1-2 years']=2
    d['2-3 years']=3
    d['3-4 years']=4
    d['4-5 years']=5
    d['5-10 years']=6
    d['10-20 years']=7
    d['20 or more years']=8


    df.loc[0:,'Q15']=[d[j] for j in df.loc[0:,'Q15']]
    #df.loc[0:,'Q15']



    #Ordinal encoding Q20#########################
    #df['Q20'].unique()

    d={}

    d['0-49 employees']=0
    d['50-249 employees']=1
    d['250-999 employees']=2
    d['1000-9,999 employees']=3
    d['10,000 or more employees']=4

    df.loc[0:,'Q20']=[d[j] for j in df.loc[0:,'Q20']]
    #df.loc[0:,'Q20']


    #Ordinal encoding Q21#########################
    #df['Q21'].unique()

    d={}

    d['0']=0
    d['1-2']=1
    d['3-4']=2
    d['5-9']=3
    d['10-14']=4
    d['15-19']=5
    d['20+']=6


    df.loc[0:,'Q21']=[d[j] for j in df.loc[0:,'Q21']]
    #df.loc[0:,'Q21']



    #Ordinal encoding Q22#########################
    #df['Q22'].unique()

    d={}

    d['I do not know']=0
    d['No (we do not use ML methods)']=1
    d['We use ML methods for generating insights (but do not put working models into production)']=2
    d['We are exploring ML methods (and may one day put a model into production)']=3
    d['We recently started using ML methods (i.e., models in production for less than 2 years)']=4
    d['We have well established ML methods (i.e., models in production for more than 2 years)']=5

    df.loc[0:,'Q22']=[d[j] for j in df.loc[0:,'Q22']]
    #df.loc[0:,'Q22']


    #Ordinal encoding Q24#########################
    #df['Q24'].unique()

    d={}

    d['0-999']=0   #d['$0-999']=0
    d['1,000-1,999']=1
    d['2,000-2,999']=2
    d['3,000-3,999']=3
    d['4,000-4,999']=4
    d['5,000-7,499']=5
    d['7,500-9,999']=6
    d['10,000-14,999']=7
    d['15,000-19,999']=8
    d['20,000-24,999']=9
    d['25,000-29,999']=10
    d['30,000-39,999']=11
    d['40,000-49,999']=12
    d['50,000-59,999']=13
    d['60,000-69,999']=14
    d['70,000-79,999']=15
    d['80,000-89,999']=16
    d['90,000-99,999']=17
    d['100,000-124,999']=18
    d['125,000-149,999']=19
    d['150,000-199,999']=20
    d['200,000-249,999']=21
    d['250,000-299,999']=22
    d['300,000-500,000']=23
    d['> $500,000']=24
    




    df.loc[0:,'Q24']=[d[j] for j in df.loc[0:,'Q24']]
    #df.loc[0:,'Q24']


    #Ordinal encoding Q25#########################
    #df['Q25'].unique()

    d={}


    d['$0 ($USD)']=0
    d['$1-$99']=1
    d['$100-$999']=2
    d['$1000-$9,999']=3
    d['$10,000-$99,999']=4
    d['$100,000 or more ($USD)']=5



    df.loc[0:,'Q25']=[d[j] for j in df.loc[0:,'Q25']]
    #df.loc[0:,'Q25']

    return df

#Réduction du nombre de variables explicatives --> juste V de Cramer à 0.2
def reduction(df):
    df = df[['Q5',
        'Q7_Python',
        'Q7_R',
        'Q7_SQL',
        'Q7_C',
        'Q7_C++',
        'Q7_Java',
        'Q7_Javascript',
        'Q7_Julia',
        'Q7_Swift',
        'Q7_Bash',
        'Q7_MATLAB',
        'Q7_None',
        'Q7_Other',
        'Q23_Analyze and understand data to influence product or business decisions',
        'Q23_Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
        'Q23_Build prototypes to explore applying machine learning to new areas',
        'Q23_Build and/or run a machine learning service that operationally improves my product or workflows',
        'Q23_Experimentation and iteration to improve existing ML models',
        'Q23_Do research that advances the state of the art of machine learning',
        'Q23_None of these activities are an important part of my role at work',
        'Q23_Other']]
    return df
