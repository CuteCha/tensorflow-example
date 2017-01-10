* ./classification/  dense classification example 
* ./text-classification/ sparse classification example   
* ./sparse-tensor-classification/ sparse classification old, contains code without melt dependence 
* ./tf-record/ write and read tfrecord example 

only tested on tensorflow_version: 0.12.1

How to do sparse clasfication training   
1. gen tfrecord   
goto ./tf-record/sparse/ and do 'sh run.sh' #you may need to modify input data dir, any libsvm format or tlc format(with num_features after label) file is ok(one train and one test)  
2. train and validate  
goto ./text-classification/    
sh run.sh  
you may need to modify tfrecord data dir  
you may also need to change NUM_CLASSES = 34 NUM_FEATURES = 324510 according to your data, I will make these options to flags in future version.  

the result looks like below:    
  loss: 3.498 precision@1: 0.031  
  sess run eval_ops start  
  sess run eval_ops duration: 0.0119028091431  
  ['loss:3.476', 'precision@1:0.067']  
  loss: 1.895 precision@1: 0.547  
  loss: 1.545 precision@1: 0.578  
  loss: 1.345 precision@1: 0.703  
  loss: 1.075 precision@1: 0.734  
  loss: 0.766 precision@1: 0.734  
  loss: 1.008 precision@1: 0.703  

