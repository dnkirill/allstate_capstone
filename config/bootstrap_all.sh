echo
echo "This installation sets up a P2 instance on AWS (Ubuntu), installs all necessary packages"
echo "which are required for XGBoost, TensorFlow, Keras and Hyperopt."
echo "About 2 GB of data will be downloaded. The whole installation time is about 15-20 minutes."
echo

read -p "Do you wish to continue (yes/no)? " yn
case $yn in
    [Nn]* ) exit;;
esac

cd ~
# Install dependencies
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn zip
sudo pip install -U pip

# Install latest python 2.7.12
sudo apt-add-repository ppa:fkrull/deadsnakes-python2.7 -y
sudo apt-get update
sudo apt-get --assume-yes install python2.7 python2.7-dev

# Install CUDA
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda

# Install cuDNN
CUDNN_FILE=$(ls | grep cudnn)
if [ -f "$CUDNN_FILE" ]; then
    tar -zxf $CUDNN_FILE
    sudo cp cuda/lib64/* /usr/local/cuda/lib64/
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
fi

# Add environmental variables
echo >> .bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> .bashrc
echo "export CUDA_ROOT=/usr/local/cuda" >> .bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin:$HOME/bin" >> .bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> .bashrc

# Install TensorFlow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
sudo pip install --upgrade keras numpy scipy sklearn pandas

# Install mongodb for hyperopt
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
echo "deb http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo service mongod restart
sudo pip install pymongo

# Install hyperopt
sudo pip install hyperopt networkx

# Install jupyter
sudo pip install jupyter
jupyter notebook --generate-config
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
nohup jupyter notebook & >> jupyter.log
echo

# Install XGBoost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make
sh ./build.sh
sudo pip install xgboost
cd ~

# Completed. Some notes to remember:
echo
echo "CONFIGURATION COMPLETE"
echo "Add to the imports of your python file to resolve problems with Keras:"
echo
echo ">> import tensorflow as tf"
echo ">> tf.python.control_flow_ops = tf"
echo
echo "Use nvidia-smi to check the installation and load of gpu"
