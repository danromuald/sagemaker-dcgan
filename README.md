# sm-ml-project-scaffold

sm-ml-project-scaffold is a Data Science and deep learning project structure. An environment to develop your entire machine (deep) learning project, from exploratory data analysis to production-ready code running in a Docker container.

# Project structure

The folders are organized as follows:

```
sm-ml-project-scaffold
├── README.md
├── algorithm
│   ├── README.md
│   ├── __init__.py
│   ├── custom_losses.py
│   ├── custom_metrics.py
│   ├── data_models
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── get_data.py
│   ├── lib
│   │   ├── README.md
│   │   └── __init__.py
│   ├── model.py
│   ├── optimizers.py
│   ├── schemas
│   │   ├── README.md
│   │   ├── default_hyperparameters.json
│   │   ├── input_schema.json
│   │   └── output_schema.json
│   ├── sm_algo.py
│   ├── sm_serve.py
│   ├── sm_train.py
│   └── util
│       ├── README.md
│       └── __init__.py
├── bin
│   └── docker-entrypoint.sh
├── buildspec.yml
├── data
│   ├── eval
│   ├── test
│   └── train
├── doc
├── docker_build
│   ├── Dockerfile
│   └── build_and_push.sh
├── examples
├── notebooks
│   ├── README.md
│   ├── eda.ipynb
│   └── outline.ipynb
├── opt
│   └── ml
│       ├── input
│       │   ├── config
│       │   └── data
│       │       ├── eval
│       │       ├── examples
│       │       └── train
│       ├── model
│       └── output
├── package.info
├── requirements.txt
├── scripts
│   └── __init__.py
├── templates
│   ├── docker_buildspec_template.yml
│   └── mxnet_hyperparameters_template.json
└── test

27 directories, 33 files
```

# Usage

1. Clone this repo.
2. Use README files in each folder to guide your project.
3. Get rid of what you don't need.
4. Send me suggestions or pull requests.

# sagemaker-dcgan
# sagemaker-dcgan
# sagemaker-dcgan
