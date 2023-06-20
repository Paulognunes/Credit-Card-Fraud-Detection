# Imports
import pickle
import seaborn as sns
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline


def cross_validation(X, y, model, sampler, splitter):
    accuracy_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()

    for train_index, test_index in splitter.split(X, y):
        # Treinando o modelo
        pipeline = make_pipeline(sampler, model)
        fit_model = pipeline.fit(X.iloc[train_index], y.iloc[train_index])

        # Calculando as métricas
        predict = model.predict(X_train.iloc[test_index])
        accuracy_list.append(accuracy_score(y_train.iloc[test_index], predict))
        recall_list.append(recall_score(y_train.iloc[test_index], predict))
        precision_list.append(precision_score(y_train.iloc[test_index], predict))
        f1_list.append(f1_score(y_train.iloc[test_index], predict))

    return fit_model, accuracy_list, recall_list, precision_list, f1_list


def save_with_pickle(file, directory, file_name):
    path = directory + file_name + '.pkl'
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_with_pickle(directory, file_name):
    path = directory + file_name + '.pkl'
    with open(path, "rb") as f:
        return pickle.load(f)


def calculate_metrics(name, model, X_test, y_test, df):
    y_predict = model.predict(X_test)
    line = [accuracy_score(y_test, y_predict), recall_score(y_test, y_predict),
            precision_score(y_test, y_predict), f1_score(y_test, y_predict)]
    df.loc[name] = line


def plot_confusion_matrix(model, name_title, X_test, y_test, ax, ax_index):
    # Calculando a matrix de confusão
    cf_matrix = confusion_matrix(y_test, model.predict(X_test))

    # Labels e contagem de instâncias em cada setor
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = []
    for i in range(2):
        soma_linha = cf_matrix[i].sum()
        group_percentages.append("{0:.2%}".format(cf_matrix[i][0] / soma_linha))
        group_percentages.append("{0:.2%}".format(cf_matrix[i][1] / soma_linha))

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plot usando heatmap
    sns.heatmap(cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis], annot=labels,
                fmt='', cmap='Purples', square=True, ax=ax[ax_index], annot_kws={"fontsize": 16})

    ax[ax_index].set_title(f'{name_title}\n', fontsize=18)
    ax[ax_index].set_xlabel('\nPredicted Values', fontsize=16)
    ax[ax_index].set_ylabel('True Values ', fontsize=16)
    ax[ax_index].tick_params(labelsize=12)
    ax[ax_index].xaxis.set_ticklabels(['False', 'True'])
    ax[ax_index].yaxis.set_ticklabels(['False', 'True'])
