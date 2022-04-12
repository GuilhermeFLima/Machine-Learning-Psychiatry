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
only once.

- changed similarity scripts to expect a series with vectors, and not phrases. The series of vectors will be created in the get_vec.py script.
- First draft of feature extract done. Need to loop over all verbal tasks and over all subjects, then join the dataframes accordingly. Also need to be careful with flit and flib.
- Looped over all anchors for feature extraction.
- Concatenated all csv files into one in Verbal Tasks joined features.