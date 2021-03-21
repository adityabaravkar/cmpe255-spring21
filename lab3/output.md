| Experiement | Accuracy           | Confusion Matrix      | Comment                                                                             |
| ----------- | ------------------ | --------------------- | ----------------------------------------------------------------------------------- |
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |                                                                                     |
| Solution 1  | 0.7135416666666666 | [[116  14] [ 41  21]] | Insulin has 374 0 values hence replaced it with pedigree which has no 0 values      |
| Solution 2  | 0.7864583333333334 | [[117  13] [ 28  34]] | Replacing pedigree feature by glucose as it has greater value in correlation matrix |
| Solution 3  | 0.796875           | [[121   9] [ 30  32]] | Adding pedigree feature and scaling the features to between 0 and 1                 |
