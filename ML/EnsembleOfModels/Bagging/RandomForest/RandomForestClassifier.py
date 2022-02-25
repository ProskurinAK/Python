import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Функция создания дата_сета для задач классификации и запись его в файл
def make_dataset():
    features, targets = make_classification(n_samples=200, n_features=9, random_state=0)

    features = np.around(features, decimals=7)

    file = open(r'C:\Users\Andrey\Desktop\dataset.txt', 'w')

    for i in range(200):
        for j in range(9):
            file.write(str("{:f}".format(features[i][j])) + "\t")
        file.write('\n')

    file.write(str(targets))
    file.close()

    print(features)
    print('-----------------------------------')
    print(targets)


# Функция чтения дата_сета из файла
def read_dataset(features, targets):
    file = open(r'D:\PyCharm Projects\EnsembleOfModels\Bagging\RandomForest\DataSet.txt', 'r')

    # чтение матрицы объектов признаков из файла
    for i in np.arange(200):
        line = file.readline().split()
        features[i] = line

    # чтение вектора ответов из файла
    counter = 0
    for i in np.arange(6):
        line = file.readline().replace('[', ' ').replace(']', ' ').split()
        for j in np.arange(len(line)):
            targets[counter] = line[j]
            counter += 1

    file.close()


X = np.zeros((200, 9))
Y = np.zeros(200)

# make_dataset()
read_dataset(X, Y)

# создание дата_сета в формате DataFrame
# --------------------------------------------------------------------------------------
data = pd.DataFrame(X)
target = pd.DataFrame(Y)

dataset = data.join(target, lsuffix='A', rsuffix='B')
dataset = dataset.rename(columns={'0A': 0, '0B': 9})

# предварительный анализ данных
# print(dataset)
# print('--------------------------------------')
# print(dataset.info())
# print('--------------------------------------')
# print(dataset.isna().sum())
# print('--------------------------------------')
# print(dataset.describe())
# --------------------------------------------------------------------------------------

# разделение дата_сета на тренировочные и тестовые данные
# --------------------------------------------------------------------------------------
X = dataset.drop(9, axis=1)
Y = dataset[9]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

# print(X_train)
# print('--------------------------------------')
# print(Y_train)
# --------------------------------------------------------------------------------------

# масштабирование данных
# --------------------------------------------------------------------------------------
ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print(Y_train)
# print(Y_test)
# --------------------------------------------------------------------------------------

# обучение модели
# --------------------------------------------------------------------------------------
rfc = RandomForestClassifier(n_estimators=9, random_state=0)
rfc.fit(X_train_scaled, Y_train)
print(rfc.score(X_train_scaled, Y_train))
print(rfc.predict(X_train_scaled))
# --------------------------------------------------------------------------------------
