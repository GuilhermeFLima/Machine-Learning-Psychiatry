import pandas as pd
import numpy as np
import fasttext.util
import re
from groups import number_to_group
from scipy import spatial


# fasttext.util.download_model('fr', if_exists='ignore')
print('Loading French FastText model.')
ft = fasttext.load_model('Feature-Extraction/cc.fr.300.bin')
print('Done.')


class Features:
    def __init__(self, task_file, task_path):
        self.task_file = task_file
        self.task_path = task_path
        self.task_series = pd.read_csv(self.task_path + self.task_file)['0']

    @staticmethod
    def get_number(filename: str) -> str:
        pattern = r"(^\d+)"
        regex = re.compile(pattern)
        list_with_number = regex.findall(filename)
        return list_with_number[0]

    def patient_number(self):
        return self.get_number(self.task_file)

    def patient_group(self):
        return number_to_group(self.patient_number())

    def group_number(self):
        """
        Given a group name, returns its corresponding number, ie:
        control -> 0; mania -> 1; etc...
        """
        group_names = ['control',
                       'mania',
                       'mixed mania',
                       'mixed depression',
                       'depression',
                       'euthymia']
        return group_names.index(self.patient_group())

    def repeat_entries(self):
        return self.task_series.duplicated().sum()

    def unique_entries(self):
        return len(self.task_series.unique())

    def repeat_words(self) -> int:
        all_words = []
        for (_, phrase) in self.task_series.iteritems():
            words_list = phrase.split('_')
            all_words += words_list
        s = pd.Series(all_words)
        repeats = s.duplicated().sum()
        return repeats

    @staticmethod
    def get_vec(phrase: str):
        word_list = phrase.split('_')
        vec_list = [ft.get_word_vector(word) for word in word_list]
        final_vec = np.sum(vec_list, axis=0)
        return final_vec

    def vectors(self):
        return self.task_series.map(self.get_vec)

    def unique_vectors(self):
        unique = pd.Series(self.task_series.unique())
        return unique.map(self.get_vec)

    @staticmethod
    def cos_sim(v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def average_global_similarity(self):
        if len(self.unique_vectors()) < 2:
            return 0.0
        else:
            sim_list = []
            for (i, vec1) in self.unique_vectors().iteritems():
                for (j, vec2) in self.unique_vectors().iteritems():
                    if i < j:
                        sim = self.cos_sim(vec1, vec2)
                        sim_list.append(sim)
            return np.mean(sim_list)

    def average_neighbour_similarity(self):
        if len(self.vectors()) < 2:
            return 0.0
        else:
            similarities = []
            v0 = None
            for (i, v1) in self.vectors().iteritems():
                if not (v0 is None):
                    sim = self.cos_sim(v0, v1)
                    similarities.append(sim)
                    v0 = v1
                else:
                    v0 = v1
            return np.mean(similarities)


if __name__ == '__main__':
    
    # Uncomment to test.
    #
    # tf = '101 courage words.csv'
    # tp = 'Data/Verbal Tasks 1-3/'
    #
    # test = Features(task_file=tf, task_path=tp)
    # print('Patient number:', test.patient_number())
    # print('Patient group:', test.patient_group())
    # print('Group\'s number:', test.group_number())
    # print('Repeat entries:', test.repeat_entries())
    # print('Unique entries:', test.unique_entries())
    # print('Repeat words:', test.repeat_words())
    # print('Global Similarity:', test.average_global_similarity())
    # print('Neighbour Similarity:', test.average_neighbour_similarity())
