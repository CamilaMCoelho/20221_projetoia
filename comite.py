import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from google.colab import drive
drive.mount('/content/drive/')

# Link do DF -> https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction?resource=download
df = pd.read_csv('/content/Employee.csv',sep =',')
df.drop(columns=['City'], axis=1, inplace=True)
df.head()

def converterInt(dataset, coluna, possibilidades):
  column = []
  for i in range(len(dataset)):
    for index, possibilidade in enumerate(possibilidades):
      if dataset[coluna][i] == possibilidade: column.append(index)
  return column

colunas = [[], [], []]

for i in range(len(df)):
  if df['Education'][i] == 'Bachelors': colunas[0].append(1)
  else: colunas[0].append(0)
  if df['Education'][i] == 'Masters': colunas[1].append(1)
  else: colunas[1].append(0)
  if df['Education'][i] == 'PHD': colunas[2].append(1)
  else: colunas[2].append(0)
  
df.drop(columns=['Education'], axis=1, inplace=True)
df['Bachelors'] = colunas[0]
df['Masters'] = colunas[1]
df['PHD'] = colunas[2]
df['Gender'] = converterInt(df, 'Gender', ['Male', 'Female'])
df['EverBenched'] = converterInt(df, 'EverBenched', ['No', 'Yes'])

df
X = df[['Bachelors', 'Masters', 'PHD', 'JoiningYear', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']]
y = df['LeaveOrNot']
X
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# APLICANDO O ALGORITMO DA ÁRVORE DE DECISÃO #
from sklearn import tree
modeloARV = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
modeloARV = modeloARV.fit(X_train, y_train)
predictionsARV = modeloARV.predict(X_test)

# MOSTRANDO OS RESULTADOS DO ALGORITMO DA ÁRVORE DE DECISÃO #

print("\nMatriz de confusão detalhada da Árvore de Decisão:\n",
      pd.crosstab(y_test, predictionsARV, rownames=['Real'], colnames=['Predito'],
      margins=True, margins_name='Todos'))

dic_arv = metrics.classification_report(y_test, predictionsARV, target_names=['Fica', 'Sai'],output_dict=True)
result_arv = dic_arv['accuracy']
print("Acurácia da Árvore de Decisão: ", result_arv)

# APLICANDO O ALGORITMO KNN #

from sklearn.neighbors import KNeighborsClassifier

modeloKNN = KNeighborsClassifier(n_neighbors=3)
modeloKNN.fit(X_train, y_train)
predictionsKNN = modeloKNN.predict(X_test)

# MOSTRANDO OS RESULTADOS DO ALGORITMO KNN #

print("\nMatriz de confusão detalhada do KNN:\n",
      pd.crosstab(y_test, predictionsKNN, rownames=['Real'], colnames=['Predito'],
      margins=True, margins_name='Todos'))

dic_knn = metrics.classification_report(y_test, predictionsKNN, target_names=['Fica', 'Sai'],output_dict=True)
result_knn = dic_knn['accuracy']
print("Acurácia do KNN: ", result_knn)

# APLICANDO O ALGORITMO SVM #

from sklearn import svm

modelosvm = svm.SVC()
modelosvm.fit(X_train.values, y_train)

predictionsSVM = modelosvm.predict(X_test.values)

# MOSTRANDO OS RESULTADOS DO ALGORITMO SVM #

print("\nMatriz de confusão detalhada do SVM:\n",
      pd.crosstab(y_test, predictionsSVM, rownames=['Real'], colnames=['Predito'],
      margins=True, margins_name='Todos'))

dic_svm = metrics.classification_report(y_test, predictionsSVM, target_names=['Sai', 'Fica'],output_dict=True)
result_svm = dic_svm['accuracy']
print("Acurácia do SVM: ", result_svm)

# COMITÊ DE CLASSIFICADORES #
'''[1, 2017, 0, 3, 25, 0, 0, 5]
   [1, 2017, 0, 3, 25, 1, 0, 5]'''

fica_sai = 0

pred_arv = (modeloARV.predict([[0, 1, 0, 2018, 2, 35, 1, 0, 5]]))
if pred_arv == 1 : fica_sai -= 1
else: fica_sai += 1
print(pred_arv)

pred_knn = (modeloKNN.predict([[0, 1, 0, 2018, 2, 35, 1, 0, 5]]))
if pred_knn == 1: fica_sai -= 1
else: fica_sai += 1
print(pred_knn)

pred_svm = (modelosvm.predict([[0, 1, 0, 2018, 2, 35, 1, 0, 5]]))
if pred_svm == 1: fica_sai -= 1
else: fica_sai += 1
print(pred_svm)

if fica_sai > 0:
  print("O comitê acredita que o individuo ficará.")
else:
  print("O comitê acredita que o individuo sairá.")

if abs(fica_sai)==3 :
  print("O comitê é unanime em sua opinião")