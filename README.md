# Allstate Claims Severity Project

## Structure

This project provides a sample solution to [Allstate Claims Severity competition](https://www.kaggle.com/c/allstate-claims-severity) on Kaggle. It's been initially developed for the Capstone (a final project) on Udacity Machine Learning Nanodegree Program, but (with slight modifications) it's now available for everyone. Rather than achieving the best score on Kaggle, which requires good hardware and a considerate amount of time, this project aims to guide through the process of training and optimizing two models (XGBoost and MLP) and stacking the results using a linear regression. Such a structure makes it easy for beginners to get the basic concepts, and then to apply them in practice.

The project is divided into four sections, each section is described in a corresponding Jupyter notebook. Your feedback on improving notebooks is welcome!

* **[Part 1: Data Discovery](part1_data_discovery.ipynb)** — we get accustomed with Allstate's dataset and do basic data analysis: we collect basic statistics, plot a correlation matrix, compare train and test distributions.
* **[Part 2: XGBoost model training and tuning](part2_xgboost.ipynb)** — we try to solve the regression problem with [XGBoost](xgboost.readthedocs.io/en/latest/), a powerful and popular gradient boosting library. We start with a simple model, develop a framework for hyper-parameters optimization and train optimized models.
* **[Part 3: Multilayer Perceptron model training and tuning](part3_mlp.ipynb)** — we span the space with feed-forward neural networks. This section will be done with [TensorFlow](https://www.tensorflow.org/) (acting as a backend) and [Keras](https://keras.io/) (acting as a frontend). We introduce (and show by example) the concept of overfitting, use K-Fold cross-validation to compare the performance of our models, tune hyper-parameters (number of units, dropout rates, optimizers) via Hyperopt and select the best model.
* **[Part 4: Linear regression stacking and results validation](part4_stacking.ipynb)** — we combine predictions of XGBoost and MLP using a linear regression, observe the results and prove their statistical significance.

You can also read a [Capstone Report](report.md) which summarizes the implementation as well as the methodology of the whole project without going deep into details.

## Requirements

### Dataset

The dataset needs to be downloaded separately (21 MB). Just unzip it in the same directory with notebooks. The dataset is available for free on [Kaggle's competition page](https://www.kaggle.com/c/allstate-claims-severity/data).

### Pretrained models

To get the results quickly, the default option is to use pretrained models. At the beginning of XGBoost and MLP notebooks, there's a flag: `USE_PRETRAINED = True` which can be set to `False` to enable calculations. The default option (`True`)  just loads ready-to-use models from `pretrained` directory.

### Software

This project uses the following software (if version number is omitted, latest version is recommended):

* **Python stack**: python 2.7.12, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **XGBoost**: multi-threaded xgboost should be compiled, xgboost python package is also required.
* **Deep Learning stack**: CUDA 8.0.44, cuDNN 5.1, TensorFlow 0.11.0rc (compiled with GPU flags), Keras.
* **Hyperopt for hyper-parameter optimization:** hyperopt, networkx python packages, MongoDB 3.2, pymongo python driver.

## Guide to running this project

### Using AWS instances (recommended)

**Step 1. Launch EC2 instance**
The best option to run the project is to use EC2 AWS instances:

* `c4.8xlarge` CPU optimized instance for XGBoost calculations (best for Part 2).
* `p2.xlarge` GPU optimized instance for MLP and ensemble calculations (best for Part 3, Part 4). If you run an Ireland-based spot instance, the price will be about $0.15-0.2 per hour.

Please make sure you run Ubuntu 14.04. For Ireland region you can use this AMI: **ami-ed82e39e**. Also, add 30 GB of EBS volume to your instance. Your security group should be configured to allow incoming connections on port `8888`  which is used by Jupyter.

**Step 2. Clone this project**

`sudo apt-get install git`

`cd ~; git clone https://github.com/dnkirill/allstate_capstone.git`

`cd allstate_capstone`

**Step 3. Deploy configuration**
Deployment scripts (XGBoost-only and XGBoost + MLP) for Ubuntu instances are provided in `config` directory of this project.

Option 1: `bootstrap_xgb_hyperopt.sh` configures an instance (`c4.8xlarge` is recommended) for XGBoost and Hyperopt calculations. It installs essential libraries, python 2.7.12, xgboost, mongodb, hyperopt and python stack: numpy, scipy, pandas, sklearn, etc. Run this if you don't plan to train MLP or ensembles. Part 1 and Part 2 notebooks don't require anything beyond the scope of this script.

Option 2 (full): `bootstrap_all.sh` is the full deployment script and it also installs CUDA, cuDNN, TensorFlow and Keras. This is required to run Part 3 and Part 4 notebooks. **Optional, but important:** to speed up calculations, download cuDNN library (tarball) into your home directory before running this script. cuDNN 5.1 works best with this configuration.

It will take about 20 minutes to configure the instance. After that, all the packages are installed, Jupyter server is ready and you can connect to it via your browser: `{instance_public_dns}:8888`.

### Using your own hardware

Of course, it's possible to test the project on your local machine. Here are the suggested steps:

* Make sure you run a modern Python 2.7. Python 2.7.12. is preferred.

* For neural networks part, you need to install CUDA and cuDNN (provided you'll run the computations on GPU). If you just test the code using CPU, you can simply skip this part.

  CUDA (Ubuntu 14.04) can be downloaded and installed from NVIDIA website https://developer.nvidia.com/cuda-downloads.

  Also, you need to install cuDNN (deep learning library for CUDA). Uncompress it and add it to your CUDA directory:

  `tar -zxf cudnn-8.0-linux-x64-v5.1.tgz`

  `sudo cp cuda/lib64/* /usr/local/cuda/lib64/`

  `sudo cp cuda/include/cudnn.h /usr/local/cuda/include/`

* Then, you can install TensorFlow 0.11.0rc. Make sure you installed GPU-optimized version: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#optional-linux-enable-gpu-support

* Install multi-threaded XGBoost: https://xgboost.readthedocs.io/en/latest/get_started/index.html

* Install MongoDB 3.2: https://www.mongodb.com/download-center#community, then install the driver: `sudo pip install pymongo`.

* Install Hyperopt: `sudo pip install hyperopt networkx`.

This should be enough to run this project, provided that you have basic python packages (`numpy scipy sklearn pandas matplotlib`) installed. Now you can start Jupyter server and run notebooks.