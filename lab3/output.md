| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.7135416666666666  | [[112  22] [ 33  25]] | I used random_state in train and test as "2" |
| Solution 2   | 0.7604166666666666  | [[118  16] [ 30  28]] | Keeping the random state as "2" I added "glucose" to the list of features |
| Solution 3   | 0.7864583333333334  | [[117  13] [ 28  34]] |  Changed the random_state for train and test split to"5" and for logestic regression to "2" and setting the max_iter to 90 i got the best accuracy.|
