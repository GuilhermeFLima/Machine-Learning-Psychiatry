### Quick Overview

This is a supervised machine learning project where we attempt to predict certain patient subclasses (mania, mixed mania, euthymia, mixed depression, depression) by applying classifiers to features extracted from verbal fluency tasks which were performed by patients with bipolar disorder (BD).

### Overview

A total of 140 subjects with BD were recruited from inpatient and outpatient clinics at the University Hospital of Strasbourg. Nine verbal fluency tasks were administered in French. Six of these were associational, i.e. the patients were given an initial cue word (courage, debut, douleur, royaume, serpente). 

The remaining three consisted of 
- a free fluency trial (flib) where they had to produce as many words as possible; 
- a letter fluency trial (flit), where all words should start with the letter 'p'; 
- and a category fluency trial (fcat), where patients had to produce as many words that fit under the semantic category 'animal'. 

Patients were given two minutes to perform all tasks except the free fluency trial, where they were given 150 seconds. 

From each task, five features were extracted:
- *unique_entries*: the number of unique entries;
- *repeat_entries*: the number of repeated entries ;
- *repeat_words*: the number of repeated words,since some entries consisted of more than one word;
- *avg_global_sim*: the mean similarity between all unique words;
- *avg_neigh_sim*: the mean similarity between every pair of words in sequence.

The similarities between words was extracted using a pre-trained [fastText](https://fasttext.cc/) model for the French language trained on [Common Crawl](https://commoncrawl.org/).

For each task, we ran the data through a support vector machine classifier as follows. In order to perform a grid search with cross-validation, the data was first split into two parts: a training set and a test set. The training set was split via cross validation (k=10) in order to perform a grid search for the best kernel (linear, radial-basis-function) and best parameters (C, gamma). Once the best parameters were found, the model was retrained and evaluated on the test set. The evaluation metrics where accuracy, F1 score, and Matthews correlation coefficient.

### Results


|    task |  acc |   F1 |   MCC | kernel |      C |  gamma | best cv |
|--------:|-----:|-----:|------:|-------:|-------:|-------:|--------:|
| courage | 0.91 | 0.93 |  0.81 |    rbf |   10.0 |    1.0 |    0.75 |
|   debut | 0.55 | 0.67 | -0.04 |    rbf |    1.0 | 316.23 |    0.68 |
| douleur | 0.73 | 0.80 |  0.39 |    rbf |  100.0 |    1.0 |    0.68 |
| piscine | 0.73 | 0.80 |  0.39 |    rbf |  31.62 |    1.0 |    0.71 |
| royaume | 0.36 | 0.46 | -0.26 |    rbf | 316.23 |    1.0 |    0.75 |
| serpent | 0.82 | 0.89 |  0.52 |    rbf |    1.0 |  100.0 |    0.65 |
|    fcat | 0.73 | 0.82 |  0.24 | linear |  100.0 |        |    0.81 |
|    flib | 0.45 | 0.40 |  0.29 |    rbf |  100.0 |    1.0 |    0.67 |
|    flit | 0.36 | 0.46 | -0.26 |    rbf |  100.0 |    1.0 |    0.64 |


Average accuracy: 0.63

Accuracy SD: 0.20

Max accuracy: 0.91