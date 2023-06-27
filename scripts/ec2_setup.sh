set -o xtrace

sudo yum update

# install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip install --upgrade pip

# install conda
sudo yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver -y
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
sh Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc

# download cuda requirements
conda install -c conda-forge -y cudatoolkit=11.8.0

# download deployment
aws s3api get-object --bucket bidi-enc-rep-trnsformers-everywhere --key v1/pretraining/4_persentence_pretrain_mlm_15p/ec2_deployment.tar.gz ec2_deployment.tar.gz
tar -xf ec2_deployment.tar.gz

# download python requirements
cd workspace
pip install -r requirements.txt
pip install -r aws_requirements.txt

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

ln -s $CONDA_PREFIX/lib/libcusolver.so.10 $CONDA_PREFIX/lib/libcusolver.so.11 # workaround

# test tensorflow gpu setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
