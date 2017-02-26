# tensorflow-example(Now support text classfication)

## using tf-record(suggeted, only tested with tf.__version__ 1.0.0)  
see README.md in ./examples  
incase not find dependence, make sure set PYTHONPATH to include tensorflow_example/util so we can find gezi and melt

## without tf-record(depreciated)  
cd ./depreciated  

basic tensorflow examples of doing binary classification
will show auc result for each epoch
It can deal with both dense or sparse input(like 3:2.5 1234:6.7)

#binary classification of dense input data using logistic regression 
python ./binary_classification.py  --train ./data/feature.normed.rand.12000.0_2.txt --test ./data/feature.normed.rand.12000.1_2.txt 
python ./binary_classification.py  --train ./data/feature.normed.rand.12000.0_2.txt --test ./data/feature.normed.rand.12000.1_2.txt  --method mlp

#binary classification of sparse input data using logistic regression 
python ./binary_classification.py  --train ./data/feature.trate.0_2.normed.txt --test ./data/feature.trate.1_2.normed.txt  
python ./binary_classification.py  --train ./data/feature.trate.0_2.normed.txt --test ./data/feature.trate.1_2.normed.txt --method mlp

python ./binary_classification.py --tr corpus/feature.trate.0_2.normed.txt --te corpus/feature.trate.1_2.normed.txt --batch_size 200 --method mlp --num_epochs 1000

... loading dataset: corpus/feature.trate.0_2.normed.txt

0

10000

20000

30000

40000

50000

60000

70000

finish loading train set corpus/feature.trate.0_2.normed.txt

... loading dataset: corpus/feature.trate.1_2.normed.txt

0

10000

finish loading test set corpus/feature.trate.1_2.normed.txt

num_features: 4762348

trainSet size: 70968

testSet size: 17742

batch_size: 200 learning_rate: 0.001 num_epochs: 1000

I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 24

I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 24

I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 24

I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 24

0 auc: 0.503701159392 cost: 0.69074464019

1 auc: 0.574863035489 cost: 0.600787888115

2 auc: 0.615858601208 cost: 0.60036152958

3 auc: 0.641573172518 cost: 0.599917832685

4 auc: 0.657326531323 cost: 0.599433459447

5 auc: 0.666575623414 cost: 0.598856064529
