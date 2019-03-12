# Simple baseline 
## features<br>
1. with a simplified resnet-18 model(decrease channels)<br>
   (Just using simple adaptive pooling to aggregate feature)
2. with linear spectrum magnitude as  feature
3. CE loss

## perfomance
We just trained 40 epoches,and no any hyperparam 
optimzation,Later we will add some tricks

| trainset | testset | EER|
|:----:|:----|:---:|
|vox2 train| vox1 test| 82.5(verify@thres:0.6)|
|vox2 train| vox1 test extended| TODO|
|vox2 train| vox1 test hard| TODO|
|vox2 train| vox2 val| 96.5(clc top1)|

## train
just look at train.py<br>
TODO: details about preprocessing and train

## predict
just look at predict.py<br>
 TODO: details

## Cost
5.7G memory, about 10 hour on 2080ti