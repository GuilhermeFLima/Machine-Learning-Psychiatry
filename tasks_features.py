task_features = {'courage': ['courage_unique_entries', 'courage_repeat_entries', 'courage_repeat_words',
                             'courage_avg_anch_sim', 'courage_avg_global_sim', 'courage_avg_neigh_sim'],
                 'debut': ['debut_unique_entries', 'debut_repeat_entries', 'debut_repeat_words', 'debut_avg_anch_sim',
                           'debut_avg_global_sim', 'debut_avg_neigh_sim'],
                 'douleur': ['douleur_unique_entries', 'douleur_repeat_entries', 'douleur_repeat_words',
                             'douleur_avg_anch_sim', 'douleur_avg_global_sim', 'douleur_avg_neigh_sim'],
                 'piscine': ['piscine_unique_entries', 'piscine_repeat_entries', 'piscine_repeat_words',
                             'piscine_avg_anch_sim', 'piscine_avg_global_sim', 'piscine_avg_neigh_sim'],
                 'royaume': ['royaume_unique_entries', 'royaume_repeat_entries', 'royaume_repeat_words',
                             'royaume_avg_anch_sim', 'royaume_avg_global_sim', 'royaume_avg_neigh_sim'],
                 'serpent': ['serpent_unique_entries', 'serpent_repeat_entries', 'serpent_repeat_words',
                             'serpent_avg_anch_sim', 'serpent_avg_global_sim', 'serpent_avg_neigh_sim'],
                 'fcat': ['fcat_unique_entries', 'fcat_repeat_entries', 'fcat_repeat_words',
                          'fcat_avg_global_sim', 'fcat_avg_neigh_sim'],
                 'flib': ['flib_unique_entries', 'flib_repeat_entries', 'flib_repeat_words',
                          'flib_avg_global_sim', 'flib_avg_neigh_sim'],
                 'flit': ['flit_unique_entries', 'flit_repeat_entries', 'flit_repeat_words',
                          'flit_avg_global_sim', 'flit_avg_neigh_sim']}

if __name__ == '__main__':
    anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
    anchor_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']
    simple_fluence_list = ['fcat', 'flib', 'flit']
    simple_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']
    task_features = {}

    for anchor in anchor_list:
        feature_list = [anchor + '_' + col for col in anchor_cols]
        task_features[anchor] = feature_list

    for sf in simple_fluence_list:
        feature_list = [sf + '_' + col for col in simple_cols]
        task_features[sf] = feature_list

    print(task_features)
