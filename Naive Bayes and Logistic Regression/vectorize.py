import pandas as pd
pd.set_option('display.max_rows', None)


def find_type(filename: str) -> str:
    vf_list = ['courage', 'debut', 'douleur', 'fcat', 'flib', 'flit', 'piscine', 'royaume', 'serpent']
    for type_ in vf_list:
        if type_ in filename:
            return type_
        else:
            pass


def vectorize_vf(vf_file):
    """
    This function takes a verbal fluency (vf) task in csv format and
    returns the vector of word counts for that type of vf.
    :param vf_file:
    :return:
    """
    vf_type = find_type(vf_file)

    s = pd.read_csv(vf_file, squeeze=True)
    counts = s.value_counts()

    all_words = pd.read_csv('All Words/' + vf_type + '.csv', squeeze=True)
    all_words = all_words.apply(str)
    all_words = pd.Series(0, index=all_words)
    all_words.update(counts)
    return all_words


if __name__ == '__main__':
    path = '../Data/Verbal Tasks redux split/'
    file = '05 flib words.csv'
    print(vectorize_vf(path+file))

