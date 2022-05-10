import pandas as pd
from Features import Features
from os import listdir
import numpy as np


def files_by_task(path: str) -> dict:
    """
    Returns a dictionary where the keys are the tasks and
    the values are the files for that task.
    """
    tasks = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent', 'fcat', 'flib', 'flit']
    allfiles = listdir(path)
    csvfiles = sorted([x for x in allfiles if '.csv' in x])
    d = {task: [file for file in csvfiles if task in file] for task in tasks}
    return d


def extract_by_task(task: str, path: str, save_path: str):
    files = files_by_task(path)
    task_files = files[task]
    feature_matrix = []
    for file in task_files:
        features = Features(task_file=file, task_path=path)
        arr = np.array([features.patient_number(),
                        features.patient_group(),
                        features.group_number(),
                        features.unique_entries(),
                        features.repeat_entries(),
                        features.repeat_words(),
                        features.average_global_similarity(),
                        features.average_neighbour_similarity()
                        ])
        feature_matrix.append(arr)
    feature_cols = ['number',
                    'group',
                    'group number',
                    'unique_entries',
                    'repeat_entries',
                    'repeat_words',
                    'avg_global_sim',
                    'avg_neigh_sim']
    df = pd.DataFrame(feature_matrix, columns=feature_cols)
    new_filename = task + '_features.csv'
    df.to_csv(save_path + new_filename, index=False)
    return None


def extract_all(tasks, path, save_path):
    for (i, task) in enumerate(tasks):
        extract_by_task(task=task, path=path, save_path=save_path)
        if i == 0:
            print('Extracted', end=' ')
        print(task, end=' ')


if __name__ == '__main__':
    Tasks = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent', 'fcat', 'flib', 'flit']
    Path = "Data/Verbal Tasks 1-3/"
    Save_path = "Data/Verbal Tasks features/"

    extract_all(tasks=Tasks, path=Path, save_path=Save_path)
