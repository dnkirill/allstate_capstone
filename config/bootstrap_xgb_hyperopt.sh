sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn zip
sudo pip install -U pip

sudo apt-add-repository ppa:fkrull/deadsnakes-python2.7 -y
sudo apt-get update
sudo apt-get --assume-yes install python2.7 python2.7-dev

sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
echo "deb http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo service mongod restart
sudo pip install pymongo

sudo pip install hyperopt networkx

git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make
sh ./build.sh
sudo pip install xgboost tinys3
cd ~

sudo pip install --upgrade pandas
