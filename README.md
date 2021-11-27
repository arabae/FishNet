# FishNet

## Task Description

- task: Image Classification
- data: CIFAR-10
- matric: top-1, top-5 accuracy

## Command Line Interface

- model training

```bash
python main.py
```

- model inference
```bash
python main.py --isTrain False
```

## Directory structure

```bash
FishNet
├── README.md
├── .gitignore
├── main.py        # run task
│
├── models/        # related model
│   ├── blocks.py  # blocks of model modules
│   └── fishnet.py # model architecture
│
└── modules        # related training             
    ├── args.py    # arguments
    ├── dataset.py # custom dataset for FishNet
    ├── loss.py    # define label smoothing loss
    ├── trainer.py # define trainer
    └── utils.py   # file load, augmentation, etc.

```

## Results
- Table of best model performance on the train, test and validation.

|Dataset|Top1 Acc|Top5 Acc|
|-------|--------|--------|
| train | 100.00 | 100.00 |
| valid | 100.00 | 100.00 |
| test  | 100.00 | 100.00 |


- The figure show the training and validation top1 accuracy over 100 epochs(x-axis).

![image](https://user-images.githubusercontent.com/46676700/143670757-0bdbd9e1-5b83-477e-9e73-8aafcf23bb22.png)
