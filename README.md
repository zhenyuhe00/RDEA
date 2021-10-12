# RDEA
Source code for SIGIR 2021 paper **Rumor Detection on Social Media with Event Augmentations**

## Dependencies
python 3.7

pytorch 1.8.1

pytorch_geometric 1.7.0


## Usage
create "Twitter15graph" folder and "Twitter16graph" folder in the data folder
```
python ./Process/getTwittergraph.py Twitter15 # pre-process the Twitter15 dataset
python ./Process/getTwittergraph.py Twitter16 # pre-process the Twitter16 dataset

python ./Model/train.py Twitter15 100 # Run RDEA for 100 iterations on Twitter15 dataset
python ./Model/train.py Twitter16 100 # Run RDEA for 100 iterations on Twitter16 dataset
```

## Dataset
We use Twitter15 and Twitter16 dataset for the experiment.
To learn more about the dataset, please refer to [Bi-GCN](https://github.com/TianBian95/BiGCN) for more details.

## About
If you have any question, please contact zhenyu dot h at outlook dot com 

**If you find this code useful, please cite our paper.**



