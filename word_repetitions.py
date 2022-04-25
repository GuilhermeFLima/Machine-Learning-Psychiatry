import pandas as pd


def word_repeat(series) -> int:
    all_words = []
    for (_, phrase) in series.iteritems():
        words_list = phrase.split('_')
        all_words += words_list
    s = pd.Series(all_words)
    repeats = s.duplicated().sum()
    return repeats




