# Statistical Learning Hackathon 2023 - CERN Machinery Data Analysis
## Overview
Welcome to the Statistical Learning Hackathon 2023, where we dive into the fascinating world of CERN's machinery data! This year, we've provided a comprehensive dataset based on the equipment at CERN, Geneva, focusing on the operational aspects of their compressors. Your challenge is to predict the alert levels of these machines using the provided dataset, enhancing our understanding and predictive maintenance capabilities.

## Dataset
The dataset comprises several features, structured as follows:

- id: Row index (not to be used for prediction).
- main.pos: Code representing the main position of the compressor.
- fun.pos: Code denoting the functional position of the compressor within the main position.
- when: The year and month of the measurement.
- y: Target variable (alert level) to predict.
From column 6 onwards, the dataset features 24 vectorized time series for each measurement point, each with a length of 2048.

## Evaluation
### Evaluation Metric
The primary evaluation metric for this competition is the Balanced Accuracy Score (BAS). The BAS ranges from 0 to 1, with 0 indicating random performance. This metric is particularly useful for dealing with imbalanced datasets, ensuring that our evaluation is fair and comprehensive.

## Definitions:
Precision (p): True positives / (True positives + False positives)
Recall (r): True positives / (True positives + False negatives)
The BAS is essentially the macro-average of recall scores per class or the raw accuracy weighted according to the inverse prevalence of each true class. In balanced datasets, the BAS equals conventional accuracy.

### For binary classification:

Balanced Accuracy = (Sensitivity + Specificity) / 2
In cases where the classifier performs equally well on both classes, this equates to conventional accuracy.
Submission Guidelines
Please submit your predictions in a CSV format, with each row corresponding to the id in the test set and a single column y for your predicted alert levels. Ensure that your submission file is in the correct format and contains the correct number of entries.

