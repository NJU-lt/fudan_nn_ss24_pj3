# 任务一

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## 运行自监督学习算法并使用该算法在自选的数据集上训练ResNet-18

```
python run.py
python linear_classification.py
```

## 从0开始/从imagenet开始训练ResNet-18
```
python resnet18.py

```

## 实验效果

Feature evaluation is done using a linear model protocol. 

实验结果及对比结果如下所示：

| 是否采用自监督算法 | 自监督学习（监督学习）数据集 | 测试集     | top1 accuracy |
|--------------------|-----------------------------|------------|---------------|
| 是                 | CIFAR-10                    | CIFAR-100  | 38%           |
| 是                 | STL10                       | CIFAR-100  | 28%           |
| 是                 | CIFAR-100                   | CIFAR-100  | 45%           |
| 否                 | ImageNet                    | CIFAR-100  | 24%           |
| 否                 | CIFAR-100                   | CIFAR-100  | 51%           |
