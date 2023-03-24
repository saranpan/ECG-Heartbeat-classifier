# ECG-classification

The datasets used to train the models for both the Arhythmic and Myocardial Infarction tasks were obtained from the Heartbeat dataset on Kaggle (https://www.kaggle.com/datasets/shayanfazeli/heartbeat). The dataset files themselves are not included in this repository, as we utilized the Kaggle API to download and access them.

To avoid the risk of permanently deleting the dataset on Kaggle, it is recommended to download it and store it in your repository. The models used for training were implemented on Colab, and you can observe that the path always starts with "content/drive". If you prefer not to retrain the models on Colab, you may need to modify the path to make it compatible with your local environment.

## Performance of the models

(Task AR) Weighted F1 score on test set : 98.76%
(Task MI) Binary F1 score on test set : 98.44%
