import pandas as pd
import numpy as np
import groups
import vectorize
import filters
from os import listdir
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB as mnb


def improve_dataframe_display():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.precision', 2)
    return None


def cross_val(groups1: list, groups2: list, verbal_fluencies: list, alpha: float):
    print('Cross Validation mean scores and standard deviations.')
    print('')
    results = pd.DataFrame(columns=['Groups', 'VF', 'acc', 'precision', 'recall', 'F1'])
    iterate_list = [(vf, g1, g2) for vf in verbal_fluencies
                    for (i, g1) in enumerate(groups1)
                    for (j, g2) in enumerate(groups2)
                    if i < j]

    for (i, (vf, g1, g2)) in enumerate(iterate_list):
        X, y = filters.filtered_features_and_target(group1=g1, group2=g2, vf_type=vf)
        model = mnb(alpha=alpha)

        # cross validation
        scores = cross_val_score(model, X, y, cv=5)
        f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
        precision = cross_val_score(model, X, y, cv=5, scoring='precision')
        recall = cross_val_score(model, X, y, cv=5, scoring='recall')
        result_row = [g1 + ' & ' + g2, vf, scores.mean(), precision.mean(), recall.mean(), f1.mean()]
        n = len(groups1)
        row_number = i
        results.loc[row_number] = result_row
    print(results)
    print('Average accuracy: {:.2f}'.format(results['acc'].mean()))
    print('Accuracy SD: {:.2f}'.format(results['acc'].std()))
    print('Max accuracy: {:.2f}'.format(results['acc'].max()))
    print('')
    return None


def simple_train_test(groups1: list, groups2: list, verbal_fluencies: list, alpha: float):
    print('Simple train-test scores.')
    print('')
    for (i, g1) in enumerate(groups1):
        for (j, g2) in enumerate(groups2):
            if i < j:
                for vf in verbal_fluencies:
                    X, y = filters.filtered_features_and_target(group1=g1, group2=g2, vf_type=vf)
                    model = mnb(alpha=alpha)

                    # simple train-test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
                    model.fit(X_train, y_train)
                    result_string = 'Groups: {} & {}, VF: {}, score: {:.2f}'
                    print(result_string.format(g1, g2, vf, model.score(X_test, y_test)))
    print('')
    return None


def merged_cross_val(merged_groups, verbal_fluencies, alpha: float):
    print('Cross-Val merging:', *merged_groups)
    print('')
    for vf in verbal_fluencies:
        X, y = filters.filtered_features_and_target_merged_groups(merged_groups=merged_groups, vf_type=vf)
        model = mnb(alpha=alpha)

        # cross validation
        scores = cross_val_score(model, X, y, cv=5)
        f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
        precision = cross_val_score(model, X, y, cv=5, scoring='precision')
        recall = cross_val_score(model, X, y, cv=5, scoring='recall')
        result_string = 'VF: {}, mean CV score: {:.2f}, std: {:.2f}, recall: {:.2f}, precision: {:.2f}'
        print(result_string.format(vf, scores.mean(), scores.std(), recall.mean(), precision.mean()))
    print('')
    return None


def gridsearch_crossval(groups1: list, groups2: list, verbal_fluencies: list):
    print('Grid Search Cross Validation.')
    print('')

    iterate_list = [(vf, g1, g2) for vf in verbal_fluencies
                    for (i, g1) in enumerate(groups1)
                    for (j, g2) in enumerate(groups2)
                    if i < j]

    param_range = [10 ** x for x in np.arange(-2, 3, 1.0)]
    param_grid = {'alpha': param_range}

    results = pd.DataFrame(columns=['Groups', 'VF', 'acc', 'F1', 'MCC', 'alpha', 'best cv'])

    for (i, (vf, g1, g2)) in enumerate(iterate_list):
        X, y = filters.filtered_features_and_target(group1=g1, group2=g2, vf_type=vf)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Initialize GridSearchCV class
        grid_search = GridSearchCV(mnb(), param_grid, cv=5, return_train_score=True)
        # Fit to training data
        model = grid_search.fit(X_train, y_train)

        # Extracting parameters and kernel and best scores
        best_params = dict.fromkeys(['alpha'])
        best_params.update(grid_search.best_params_)
        best_alpha = best_params['alpha']
        #best_penalty = best_params['penalty']
        best_cv = grid_search.best_score_

        # Predictions
        # positive_label = group_to_number(group1)
        pred_model = model.predict(X_test)
        # Extracting accuracy, F1, MCC from predictions
        conf_matrix = confusion_matrix(y_test, pred_model)
        F1 = f1_score(y_test, pred_model)
        MCC = matthews_corrcoef(y_test, pred_model)
        test_score = grid_search.score(X_test, y_test)

        # Appending to results dataframe
        results.loc[i] = [g1 + ' & ' + g2, vf, test_score, F1, MCC, best_alpha, best_cv]

    results.to_csv('Multinomial_Naive_Bayes_GSCV_results.csv', index=False)
    print(results)


if __name__ == '__main__':
    groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
    vfs = ['courage', 'debut', 'douleur', 'fcat', 'flib', 'flit', 'piscine', 'royaume', 'serpent']

    # alpha = 1
    # merged_cross_val(merged_groups=groupnames[1:], verbal_fluencies=vfs, alpha=alpha)
    #simple_train_test(groups1=['control'], groups2=groupnames, verbal_fluencies=['flib'], alpha=alpha)
    improve_dataframe_display()
    # cross_val(groups1=groupnames, groups2=groupnames, verbal_fluencies=['flib'], alpha=alpha)
    # gridsearch_crossval(groups1=groupnames, groups2=groupnames, verbal_fluencies=vfs)
    df = pd.read_csv('Multinomial_Naive_Bayes_GSCV_results.csv')
    select = df[df['Groups'] == 'mania & mixed mania']
    select.to_csv('MNB_mania_&_mixed mania.csv')
    print(select)












