### April 1
Reformated the csv files so they are vertical and have header (a single column name which is just "0").
- Gonna check the verbal tasks individually now -> too cumbersome, will clean using functions.
- Removed unwanted periods.
- Removed underscores.
- Remove double underscores.
- Checked vector addition for nouns + adjectives. Seems good.

### April 4

- Removed prepositions and pronouns, might need to use more comprehensive list. This is in the Verbal Tasks redux folder.
- Performed some manual cleaning.
- There are still complex phrases present. Might just eliminate these using word counting.

### April 7

- added comments and "if \_\_name\_\_ == '\_\_main\_\_' ".
- Script to reduce number of words in each verbal task entry.
- Finished neighbour similarity function for pandas Series.
- corrected: piscines -> piscine.

### April 10

To do: refactor the similarity scripts so that vectors are extracted
only once. Done.

- changed similarity scripts to expect a series with vectors, and not phrases. The series of vectors will be created in the get_vec.py script.
- First draft of feature extract done. Need to loop over all verbal tasks and over all subjects, then join the dataframes accordingly. Also need to be careful with flit and flib.
- Looped over all anchors for feature extraction.
- Concatenated all csv files into one in Verbal Tasks joined features.

### April 13

- Support Vector Machine classification successful: 82% accuracy on test set for separating depressed from manic, but failed with other groups.

Todo: Grid search, cross validation, and merge classes (e.g mania & mixed mania).

### April 14

- Finished merge function.

### April 21

- Finished Grid Search with Cross Validation:
  - with all features;
  - with merged classes;
  - with each verbal fluency task separate.

- Logistic Regression with L1 penalty
- Logistic Regression by task: not so great

### April 22

- removed feature 'repeat_words' and algorithms worsened... Reason for removing was that the had weight = 0 in LogReg with L1 penalty

### April 24

- renamed unique_words and repeat_words to unique_features and repeat_features.
- finished word repeat function. Need to include word repetitions in features.

### April 25

- introduced repeated words as feature: SVM improved!

- New ideas:
  - remove features to see if SVM alg improves.
  - summarize in dataframe.
  - Decision Trees, then Random Forests.

### April 26
- SVM works better on some features when avg_anch_sim is removed!

### May 2
- Random Forests not that great...

## To do:
- Refactor Grid Search Cross Vals
- change feature repeat_words to (repeat_words - repeat_phrases).

### May 4

- Started refactoring:
  - removing avg_anch_sim. DONE
  - 
