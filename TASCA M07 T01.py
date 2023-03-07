#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[75]:


title =['class','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
        'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline'] 


wine = pd.read_csv(r"C:\Users\34626\Documents\TRABAJO\TALENTO DIGITAL\DATA SCIENCE-BARCELONA ACTIVA\SPRINT 7.- CLASSIFICACIÓ. ALGORITMES DE APRENENTATGE SUPERVISAT\wineData.txt",
    sep=',', names = title)

wine


# In[16]:


wine.shape


# In[17]:


wine.info()
# Encara que a l'anunciat del dataset diu que no hi ha nuls, es comprova


# In[18]:


wine.describe()
# Les variables tenen ordre magnitud molt diferent


# In[10]:


col=wine.drop('class',axis='columns').columns

fig,axs =plt.subplots(4,4, figsize=(12,12),layout='constrained')
axs=axs.flat
fig.suptitle('Histograma de les varaibles dataset wine')

for count,elem in enumerate(col):
    sns.histplot(
        data = wine,
        x    = elem,
        kde  = True,
        ax   = axs[count])

#Per borrar els axes que estan buits (fas la llista sobre la que itera la i)
for i in [13,14,15]:
    fig.delaxes(axs[i])


# In[104]:


# Mirem les gràfiques de les variables per veure la distribució en funció de la varaibles categòrica
col=wine.drop('class',axis='columns').columns

fig,axs =plt.subplots(4,4, figsize=(12,12),layout='constrained')
axs=axs.flat
fig.suptitle('Relació class amb variables dataset wine')

for count,elem in enumerate(col):
    sns.barplot(
        data = wine,
        x    = 'class',
        y    = elem,
        ax   = axs[count])

#Per borrar els axes que estan buits (fas la llista sobre la que itera la i)
for i in [13,14,15]:
    fig.delaxes(axs[i])


# In[105]:


fig,axs =plt.subplots(4,4, figsize=(12,12),layout='constrained')
axs=axs.flat
fig.suptitle('Gràfic violí dataset wine')

for count,elem in enumerate(col):
    sns.boxplot(
        data = wine,
        x    = 'class',
        y    = elem,
        ax   = axs[count])

#Per borrar els axes que estan buits (fas la llista sobre la que itera la i)
for i in [13,14,15]:
    fig.delaxes(axs[i])


# In[21]:


# Correlació entre variables 
corr=wine.corr()

fig,axs =plt.subplots(figsize=(16,7))
sns.set(font_scale=1.1)
fig.suptitle('Matriu de correlació-WINE')

mask=np.zeros_like(corr)
mask[np.triu_indices_from(mask)]= True

sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='BrBG')

plt.show()


# In[107]:


from sklearn.model_selection import train_test_split # per separar en training-test

# Separo el dataset en 2 parts per poder entrenar i validar els models posterioment

X_train, X_test, y_train, y_test=train_test_split(
                                        wine.drop('class', axis='columns'),
                                        wine['class'],
                                        test_size=0.3,
                                        shuffle = True, # el dataset està ordenat i cal barrejar per a que sigui aleatori
                                        stratify = wine['class'],   # per a que al separar tinguem proporcions similars de y (no estan balancejades)
                                        random_state=25
                                        )


# In[132]:


from sklearn.preprocessing import StandardScaler

lista = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
        'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline'] 
scaler= StandardScaler()
X_train_tra = scaler.fit_transform(X_train)
X_test_tra = scaler.transform (X_test)
X_train_tra=pd.DataFrame(X_train_tra, columns=lista)
X_test_tra=pd.DataFrame(X_test_tra, columns=lista)
X_train_tra


# ## Exercici 1

# In[109]:


# Model 1  
# He escollit aquest model perquè és senzill i sembla bo per comparar amb els altres

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 15)
logreg.fit(X_train_tra, y_train)

y_pred_logreg = logreg.predict(X_test_tra)


# In[110]:


# Model 2

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tra, y_train)

y_pred_knn = knn.predict(X_test_tra)


# In[111]:


# Model 3

from sklearn.ensemble import GradientBoostingClassifier 

xbgb = GradientBoostingClassifier()
xbgb.fit(X_train_tra,y_train)

y_pred_xbgb = xbgb.predict(X_test_tra)


# ## Exercici 2

# In[112]:


from sklearn import metrics

print('Logistic Regression', metrics.accuracy_score(y_test, y_pred_logreg))
print('KNN', metrics.accuracy_score(y_test, y_pred_knn))
print('Gradient Boosting', metrics.accuracy_score(y_test, y_pred_xbgb))


# In[113]:


# Per calcular l'accuracy del train set
y_train_logreg = logreg.predict(X_train_tra)
y_train_knn = knn.predict(X_train_tra)
y_train_xbgb = xbgb.predict(X_train_tra)


# In[133]:


# Aquest seria el resultat si treballem amb el set train. Em surt una accuracy molt alta, que sembla sospitosa de que hi ha algo
# que no està bé. Però entenc que necessitem aquests parametres despres per avaluar els hiperparàmetres del model al següent
# exercici
print('Logistic Regression', metrics.accuracy_score(y_train, y_train_logreg))
print('KNN', metrics.accuracy_score(y_train, y_train_knn))
print('Gradient Boosting', metrics.accuracy_score(y_train, y_train_xbgb))


# In[115]:


clases_nombres =['Clase 1', 'Clase 2', 'Clase 3']

cm_logreg = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_logreg))

sns.heatmap(cm_logreg, annot=True, fmt='d')

plt.show()

print(metrics.classification_report(y_test, y_pred_logreg))


# In[116]:


cm_knn = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_knn))

sns.heatmap(cm_knn, annot=True, fmt='d')

plt.show()

print(metrics.classification_report(y_test, y_pred_knn))

# Class 0 TP = 20, FN = 0+0 (suma de las fila-excepto TP), FP = 1+0 (suma de las columna-excepto TP), TN = 18+3+0+12 (la suma de
# filas y columnas que no pertenecen a esa clase)
# Clase 1 TP = 18 FN = 1+3, FP =0+0, TN = 20+0+0+12
# Clase 2 TP = 12, FN = 0+0, FP = 3+0, TN = 20+0+1+18


# In[117]:


cm_xbgb = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_xbgb))

sns.heatmap(cm_xbgb, annot=True, fmt='d')

plt.show()

print(metrics.classification_report(y_test, y_pred_xbgb))


# ##### AUC

# In[118]:


y_pred_proba_logreg = logreg.predict_proba(X_test_tra)
print(metrics.roc_auc_score(y_test, y_pred_proba_logreg, multi_class ='ovr'))


# In[119]:


y_pred_proba_knn = knn.predict_proba(X_test_tra)
print(metrics.roc_auc_score(y_test, y_pred_proba_knn, multi_class ='ovr'))


# In[120]:


y_pred_proba_xbgb = xbgb.predict_proba(X_test_tra)
print(metrics.roc_auc_score(y_test, y_pred_proba_xbgb, multi_class ='ovr'))


# ## Exercici 3

# Segons https://medium.com/@data.science.enthusiast/logistic-regression-tune-hyperparameters-python-code-fintech-does-it-bring-any-value-619e172565e6 els principals hyperparàmtres per la regressió logistica son: solver, penalty y regularization strenght

# In[121]:


# Logistic Regression
param_logreg = {'solver' : ['liblinear', 'newton-cg'],
                'penalty': [ 'l2'],
                'C'      : [0.001,0.01,0.1,1,10,100,1000]
               }


# In[122]:


from sklearn.model_selection import GridSearchCV, RepeatedKFold
grid_logreg = GridSearchCV(
            estimator  = LogisticRegression(random_state = 15),
            param_grid = param_logreg,
            scoring    = 'accuracy',
            n_jobs     = 5,
            cv         = RepeatedKFold(n_splits = 5, n_repeats =5),
            return_train_score = True,
            error_score= 'raise')

result=grid_logreg.fit(X= X_train_tra, y= y_train)


# In[123]:


grid_logreg.best_params_  # Aquests serien els paràmetres si avaluem en funció del train set


# In[124]:


grid_logreg.best_score_  # Aquest resultat seria per al model aplicat al train set


# In[152]:


grid_logreg.scoring


# In[125]:


# Per aplicar el model òptim del fit al test set i fer prediccions
logreg_best = LogisticRegression(random_state = 15,
                                 C = 1000,
                                 penalty = 'l2',
                                 solver = 'newton-cg')
logreg_best.fit(X_train_tra, y_train)

y_pred_logreg_best = logreg_best.predict(X_test_tra)


# In[126]:


print('Logistic Regression', metrics.accuracy_score(y_test, y_pred_logreg_best))


# In[ ]:


# Amb l'aplicació d'aquest hiperparàmetres no s'ha millorat l'ajust del model (que ja era molt alt)


# Segons https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657 valors d'alguns hiperparàmetres que es poden utilitzar

# In[134]:


# KNN


param_knn = {'n_neighbors' : [3,5,7,9],
             'weights'    : ['uniform', 'distance'],
             'metric'     : ['euclidean', 'manhattan']
            }

grid_knn  = GridSearchCV(
            estimator  = KNeighborsClassifier(),
            param_grid = param_knn,
            scoring    = 'accuracy',
            n_jobs     = 5,
            cv         = RepeatedKFold(n_splits = 5, n_repeats =5),
            return_train_score = True,
            error_score= 'raise',
            verbose    = 1)

result_knn = grid_knn.fit(X_train_tra, y_train)


# In[135]:


grid_knn.best_params_


# In[136]:


grid_knn.best_score_
# Amb aquests hiperparàmetres no s'ha millorat l'accuracy del model. Mirem si al train té algun efecte


# In[137]:


knn_best = KNeighborsClassifier (metric      = 'manhattan',
                                 n_neighbors = 5,
                                 weights     = 'uniform')
knn_best.fit(X_train_tra, y_train)

y_pred_knn_best = knn_best.predict(X_test_tra)


# In[138]:


print('knn', metrics.accuracy_score(y_test, y_pred_knn_best))

# Aquí ha sortit un ajust millor al test set, però inferior al que es troba amb els paràmetres per defecte


# In[139]:


# Gradient Boosting

param_xbgb = {'learning_rate' : [0.01, 0.1, 1, 10, 100],
               'max_depth' :    [1, 3, 5, 7, 9],
               'n_estimators' : [5, 50, 250, 500]
             }
grid_xbgb  = GridSearchCV(
            estimator  = GradientBoostingClassifier(),
            param_grid = param_xbgb,
            scoring    = 'accuracy',
            n_jobs     = 5,
            cv         = RepeatedKFold(n_splits = 5, n_repeats =5),
            return_train_score = True,
            error_score= 'raise',
            verbose    = 1)

result_xbgb = grid_xbgb.fit(X_train_tra, y_train)       


# In[140]:


grid_xbgb.best_params_


# In[141]:


grid_xbgb.best_score_


# In[142]:


xbgb_best = GradientBoostingClassifier (learning_rate = 1,
                                        max_depth     = 1,
                                        n_estimators  = 500)
xbgb_best.fit(X_train_tra, y_train)

y_pred_xbgn_best = xbgb_best.predict(X_test_tra)


# In[143]:


print('Gradient Boosting', metrics.accuracy_score(y_test, y_pred_xbgb))
# El model s'ajusta molt be al train set però no funciona bé al test.


# ## Exercici 4

# In[88]:


# Cross validation score utiliza kfold para regresión logistica en el train
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
                    estimator = LogisticRegression(random_state = 15),
                    X         = X_train_tra,
                    y         = y_train,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[89]:


cv_scores.mean()


# In[92]:


# Cross validation score utiliza kfold para regresión logistica en el test
cv_scores = cross_val_score(
                    estimator = LogisticRegression(random_state = 15),
                    X         = X_test_tra,
                    y         = y_test,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[93]:


cv_scores.mean()


# El resultat amb el train set del **train/test** és 1.0, amb el mètode **cross-validation** és 0.9749999999999999. El resultat amb el test set del train/test és 0.9814814814814815, amb el mètode cross-validation és 1.0 .

# In[102]:


# Cross validation score utiliza kfold para KNN-neighbors a en el train

cv_scores = cross_val_score(
                    estimator = KNeighborsClassifier(),
                    X         = X_train_tra,
                    y         = y_train,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[95]:


cv_scores.mean()


# In[96]:


cv_scores = cross_val_score(
                    estimator = KNeighborsClassifier(),
                    X         = X_test_tra,
                    y         = y_test,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[97]:


cv_scores.mean()


# El resultat amb el train set del **train/test** és 0.9758064516129032, amb el mètode **cross-validation** és 0.9423076923076923. El resultat amb el test set del **train/test** és 0.9444444444444444, amb el mètode **cross-validation** és 0.9833333333333334. 

# In[98]:


# Cross validation score utiliza kfold para Gradient Boosting a en el train
cv_scores = cross_val_score(
                    estimator = GradientBoostingClassifier(),
                    X         = X_train_tra,
                    y         = y_train,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[99]:


cv_scores.mean()


# In[100]:


cv_scores = cross_val_score(
                    estimator = GradientBoostingClassifier(),
                    X         = X_test_tra,
                    y         = y_test,
                    scoring   = 'accuracy',   #metrica para medir la distancia entre modelo y datos. 
                    cv        = 10)  # numero de veces que hace la cross validatrion


# In[101]:


cv_scores.mean()


# El resultat amb el train set del **train/test** és 1.0, amb el mètode **cross-validation** és 0.9185897435897437. El resultat amb el test set del **train/test** és 0.9814814814814815, amb el mètode **cross-validation** 0.93. 

# ## Exercici 5

# Miro de veure quines són les variables més importants per al model, per tal de minimitzar el número de variables amb un funcionament òptim del model. Les gràfiques que he fet al principi, poden donar una idea de les variables que poden tenir més pes al model (les que presenten més diferenciació entre classes): alcohol, total phenols, flavonoids i color intensity.

# In[186]:


# Mètode K best features
# Els scoring mètodes són diferents per problemes de regressió o classificació, sino s'utilitzen malament no funiconen bé.
# Per a problemes regresió : f_regression
# Per a problemes de clasificació : chi2, f_classif

# He intentat utilitzar chi2 però no he pogut perquè hi ha varaibles que tenen negatius. Hi havia la possibilitat d'escollir un
# un altre score (el que he fet) o bé normalitzar el dataset en comptes d'escalar-lo (que ja ho tenia fet )

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


selector = SelectKBest(f_regression, k=6)

selector.fit(X_train_tra, y_train)
X_train_kbest = selector.transform(X_train_tra)
X_test_kbest = selector.transform(X_test_tra)

print(X_train_kbest.shape)


# In[189]:


new_X_train_tra = X_train_tra.iloc[:,selector.get_support()]
new_X_train_tra


# In[218]:


# He posat el k = 6 de forma arbitraria, però no tindria que ser així, sino amb una base raonada. Ho he fet amb el score accuracy
# per poder comparar amb el model fet previament

accuracy_score_list =[]

for k in range(1,14):   # Hi ha 3 varaibles possibles
    selector = SelectKBest(f_regression, k=k)   # Per avaluar per cada num de variables
    selector.fit(X_train_tra, y_train)
    
    sel_X_train = selector.transform(X_train_tra)  # Per tenir el X_train amb les millor variables segons el num. d'interaccions
    sel_X_test  = selector.transform(X_test_tra)
    
    logreg.fit(sel_X_train, y_train)  # Aplicar el model de regressio logistica per les millor varaibles seleccionades
    kbest_predic = logreg.predict(sel_X_test)   # Predicció de la y segons logreg aplicat
    
    accuracy_score_kbest =round(metrics.accuracy_score(y_test, kbest_predic),3) # score para evaluar
    
    accuracy_score_list.append(accuracy_score_kbest)
    
print (accuracy_score_list)

 
    


# In[234]:


fig, ax = plt.subplots(figsize=(12, 6))
x = ['1','2','3','4','5','6','7','8','9','10','11','12','13']
y = accuracy_score_list
plt.bar(x, y)
ax.set_xlabel('Nombre de varaibles(seleccionades per accuracy)')
ax.set_ylabel('Accuracy_score')

ax.set_ylim(0, 1.2)  # limit de l'eix y per que encabir els numeros
for index, value in enumerate(y):
    plt.text(x=index, y=value + 0.02, s=str(value), ha='center') # limit de l'eix y per que encabir els numeros
    


# In[236]:


# Amb 5 varaibles hi ha el mateix accuracy score que amb 13, per tant passariem a buscar les 5 millors variables

selector = SelectKBest(f_regression, k=5)

selector.fit(X_train_tra, y_train)
X_train_kbest = selector.transform(X_train_tra)
X_test_kbest = selector.transform(X_test_tra)

n5_X_train = X_train_tra.iloc[:,selector.get_support()]
n5_X_train


# Dels que jo havia predit com a variables importants veiens les gràfiqus : alcohol, total phenols, flavonoids i color intensity, només 2 d'elles (total phenols i flavonoids) semblen que són les importants pel model.

# In[ ]:




