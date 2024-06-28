# Graph Neural Network for Subgraph Counting

## Quick Start

+ To train a model, use the following command:

```sh
python train.py -c ./config.json
```

+ To test the saved model, run the following command. The results will be available in the `./saved/result` directory.

```sh
python test.py -r ./saved/models/path_to_your_trained_model_best.pth
```

+ Feel free to modify the `config.json` configuration file to experiment with different model architectures, datasets, and hyperparameters.


## Folder Structure

  ```
pytorch-template/
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
│
├── test_all_model.py - run all models on all datasets and save the experiment result.
├── plot_result.py - plot the experiment result
│
├── config.json - holds configuration for training
├── parse_config.py - class to handle config file and cli options
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/ - anything about data loading goes here
│   ├── data_loaders.py - dataloader for subgraph counting task
│   ├── data_utils.py - utility functions for data processing
│   └── dataset.py - dataset for queries and their counting
│
├── data/ - directory for storing data, containing data graph, queries, and countings.
│
├── model/ - models, losses, and metrics
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   ├── log/ - default logdir logging output
│   └── result/ - default test result directory by running test.py
│
├── trainer/ - trainers
│   └── trainer.py - model training, validation, checkpoint, etc.
│
├── logger/ - module for logging
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│  
└── utils/ - small utility functions
    ├── util.py
    └── ...
  ```



## Progress

+ 20240512: Init the repository from [pytorch-template](https://github.com/victoresque/pytorch-template).
+ 20240513: Upload example dataset `hprd`.
+ 20240514: Implement dataset, dataloader, train, GCN.
+ 20240515: Implement model evaluation. (Finish the entire workflow.) 
+ 20240522: Implement GIN, GAT and GraphSAGE.
+ 20240524: Exp & Plot.

## Thanks

Our project utilizes some outside source code. Thanks for the following repositories.

| Description      | Github Link                                     |
| ---------------- | ----------------------------------------------- |
| Pytorch Template | https://github.com/victoresque/pytorch-template |

 