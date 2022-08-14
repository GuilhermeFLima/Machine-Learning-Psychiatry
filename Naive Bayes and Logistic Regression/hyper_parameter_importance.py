import pandas as pd
import numpy as np
import filters
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.naive_bayes import MultinomialNB as mnb


def improve_dataframe_display():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.precision', 2)
    return None


def LogReg_accuracy_delta(groups1: list, groups2: list, verbal_fluencies: list, C_list: list):
    print('Evaluation of accuracy and hyper-params for Logistic Regression model:')
    print('')

    iterate_list = [(vf, g1, g2) for vf in verbal_fluencies
                    for (i, g1) in enumerate(groups1)
                    for (j, g2) in enumerate(groups2)
                    if i < j]

    results = pd.DataFrame(columns=['Groups', 'VF', 'max acc', 'min acc', 'acc delta', 'acc mean', 'acc std'])
    for (i, (vf, g1, g2)) in enumerate(iterate_list):
        accuracies = []
        for C in C_list:
            X, y = filters.filtered_features_and_target(group1=g1, group2=g2, vf_type=vf)
            model = logreg(C=C)

            # simple train-test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            accuracies.append(acc)
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        delta_acc = max_acc - min_acc
        results.loc[i] = [g1 + ' & ' + g2, vf, max_acc, min_acc, delta_acc, mean_acc, std_acc]
    print(results)
    print(results.describe())
    return None


def MNB_accuracy_delta(groups1: list, groups2: list, verbal_fluencies: list, alpha_list: list):
    print('Evaluation of accuracy and hyper-params for Multinomial Naive Bayes model:')
    print('')

    iterate_list = [(vf, g1, g2) for vf in verbal_fluencies
                    for (i, g1) in enumerate(groups1)
                    for (j, g2) in enumerate(groups2)
                    if i < j]

    results = pd.DataFrame(columns=['Groups', 'VF', 'max acc', 'min acc', 'acc delta', 'acc mean', 'acc std'])
    for (i, (vf, g1, g2)) in enumerate(iterate_list):
        accuracies = []
        for alpha in alpha_list:
            X, y = filters.filtered_features_and_target(group1=g1, group2=g2, vf_type=vf)
            model = mnb(alpha=alpha)

            # simple train-test
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            accuracies.append(acc)
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        delta_acc = max_acc - min_acc
        results.loc[i] = [g1 + ' & ' + g2, vf, max_acc, min_acc, delta_acc, mean_acc, std_acc]
    print(results)
    print(results.describe())
    return None


if __name__ == '__main__':
    groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
    vfs = ['courage', 'debut', 'douleur', 'fcat', 'flib', 'flit', 'piscine', 'royaume', 'serpent']
    param_range = [10 ** x for x in np.arange(-2, 3, 1.0)]
    improve_dataframe_display()
    LogReg_accuracy_delta(groups1=groupnames, groups2=groupnames, verbal_fluencies=vfs, C_list=param_range)
    MNB_accuracy_delta(groups1=groupnames, groups2=groupnames, verbal_fluencies=vfs, alpha_list=param_range)





