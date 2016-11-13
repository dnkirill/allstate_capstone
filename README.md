# Allstate Claims Severity Capstone Project

## Structure

This Capstone project provides a sample solution to [Allstate Claims Severity competition](https://www.kaggle.com/c/allstate-claims-severity) on Kaggle. There are several sections in this project:

* **[Data Discovery](part1_data_discovery.ipynb)** — the part where we understand the Allstate's dataset we have and the patterns in it.
* **[XGBoost model training and tuning](part2_xgboost.ipynb)** — we try to solve the regression problem with [XGBoost](xgboost.readthedocs.io/en/latest/), a powerful and popular among kagglers gradient boosting library.
* **[Multilayer Perceptron model training and tuning](part3_mlp.ipynb)** — we span the space with feed-forward neural networks. This section will be done with [TensorFlow](https://www.tensorflow.org/) (acting as a backend) and [Keras](https://keras.io/) (acting as a frontend).
* **[Linear regression stacking and results validation](part4_ensemble.ipynb)** — we combine predictions of XGBoost and MLP using a linear regression and observe the results.

**[Capstone Report](report.md) is available as a Markdown document (recommended) as well as a [standalone PDF document](report.pdf).**

## Requirements

### Software

This project uses the following software (if version number is omitted, latest version is recommended):

* **Python stack**: python 2.7.12, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **XGBoost**: multi-threaded xgboost should be compiled, xgboost python package is also required.
* **Deep Learning stack**: CUDA 8.0.44, cuDNN 5.1, TensorFlow 0.11.0rc (compiled with GPU flags), Keras.
* **Hyperopt for hyper-parameter optimization:** hyperopt, networkx python packages, MongoDB 3.2, pymongo python driver.

This project was built on two OS: Ubuntu 14.04 (used for computation), Mac OS X (reporting and writing). It is strongly recommended to install and run the project on Ubuntu since compiling GPU-based TensorFlow for Mac OS X is not very straightforward.

### Hardware

I tried to reduce the complexity and computation time of my models (I succeeded with XGBoost), but some parts are still compute-heavy: 

* MLP Hyperopt optimization
* Ensemble training and validation

Unfortunately, there is no simple answer to how to compute and optimize neural networks in minutes and not to lose some of their important properties. Nevertheless, some computations are heavy and I came up to the solution to save reviewer's time. For all my notebooks which require heavy calculations, I provide the flag `USE_PRETRAINED = True` which uses precalculated models from `./pretrained` directory. This can be set to `False` to run all the calculations on your machine. 

It will take up to 2 hours on a reasonably modern PC to complete XGBoost calculations.

Another option would be to use Amazon AWS instances:

* `c4.8xlarge` CPU-optimized instance for XGBoost calculations.
* `p2.xlarge` GPU-optimized instance for MLP and ensemble calculations.

Deployment scripts for these instances are provided in `./config` directory of this project.

## Guide to running this project

### Using AWS instances (recommended)

This project requires a number of dependencies which may complicate things. The best way is to set up a `p2.xlarge` spot instance on AWS and run the following steps on Amazon machine. Please make sure you set up a Ubuntu 14.04 AMI: **ami-ed82e39e**. Also, add 32 GB of EBS volume to your instance. Your security group should be configured to allow incoming connections on port `8888`  which is used by Jupyter.

If you run an Ireland-based spot instance, the price will be about $0.15-0.2 per hour.

Next, deploy the configuration:

1. `sudo apt-get install git`
2. `git clone https://github.com/dnkirill/allstate_capstone.git`
3. `cd allstate_capstone; sh config/deploy_ml_stack.sh`

It will take about 20 minutes to configure the instance. After that, all the packages are installed, Jupyter server is ready and you can connect to it via your browser: `{instance_public_dns}:8888`. Now you are ready to run notebooks.

### Using your own hardware

Of course, it's possible to test the project on your machine. I assume you already cloned this repository. Next, you can follow these steps:

* Make sure you run a modern Python 2.7. Python 2.7.12. is preferred.

* For neural networks part, you need to install CUDA (provided you'll run the computations on GPU). If you just test the code using CPU, you can simply skip this part

  CUDA (Ubuntu 14.04) can be downloaded and installed from NVIDIA website https://developer.nvidia.com/cuda-downloads.

  Also, you need to install cuDNN (deep learning library for CUDA). I included the Linux library:

  `wget https://s3-eu-west-1.amazonaws.com/kd-allstate/cudnn-8.0-linux-x64-v5.1.tgz`

  `tar -zxf cudnn-8.0-linux-x64-v5.1.tgz`

  `sudo cp cuda/lib64/* /usr/local/cuda/lib64/`
  `sudo cp cuda/include/cudnn.h /usr/local/cuda/include/`

  Add env variables:

  ```
  echo >> .bashrc
  echo "export CUDA_HOME=/usr/local/cuda" >> .bashrc
  echo "export CUDA_ROOT=/usr/local/cuda" >> .bashrc
  echo "export PATH=$PATH:/usr/local/cuda/bin:$HOME/bin" >> .bashrc
  echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> .bashrc
  ```

  Then, you can install TensorFlow 0.11.0rc. Make sure you installed GPU-optimized version: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#optional-linux-enable-gpu-support

* Install multi-threaded XGBoost: https://xgboost.readthedocs.io/en/latest/get_started/index.html

* Install MongoDB 3.2: https://www.mongodb.com/download-center#community, then install the driver: `sudo pip install pymongo`.

* Install Hyperopt: `sudo pip install hyperopt networkx`.

This should be enough to run this project, provided that you have basic python packages (`numpy scipy sklearn pandas matplotlib`) installed. Now you can start Jupyter server and run notebooks.