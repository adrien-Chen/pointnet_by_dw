# PointNet

This repo is implementation for PointNet by Pytorch. The model can be trained by ModelNet40 or ShapeNet. You can also download the trained model for test.

  # Usage:w

  ## Training 

  - you should set the task ： `--task cls <cls | seg>`  
  - you should set the outf：`--outf <name output folder>`
  - Use `--feature_transform` to set whether use feature_transform or not

------

  Classification:

- Can use ModelNet40 or ShapeNet do classification：

```
cd pointnet
python train_net.py --task cls --dataset <dataset path> --nepoch <number epochs> --batch_size <batchsize> --dataset_type <modelnet40 | shapenet>
```

  

Segmentation:

- Only use ShapeNet do segmentation：

```
cd pointnet
python train_net.py --task seg --dataset <dataset path> --nepoch <number epochs> --batch_size <batchsize> --dataset_type shapenet
```



## Testing



  ## Evaluation

Classification：

```
python eval.py --task cls --dataset <dataset path> --model <model path> --dataset_type <modelnet40 | shapenet>
```

- On ModelNet40

|                                              | Overall Acc | trained |
| :------------------------------------------: | :---------: | ------- |
|           Original implementation            |    89.2     |         |
| this implementation（w/o feature transform） |    86.5     |         |
| this implementation（w/ feature transfoem）  |             |         |



------

To be continuing：

- On ShapeNet：

Segmentation：



