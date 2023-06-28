set -o xtrace

sudo yum update -y

# install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip install --upgrade pip

# install conda
sudo yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver -y
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
sh Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3
tee -a ~/.bashrc << END
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ec2-user/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ec2-user/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ec2-user/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ec2-user/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
END
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
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib

# workaround
mkdir $CONDA_PREFIX/lib/nvvm
ln -s $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/libdevice.10.bc
ln -s $CONDA_PREFIX/lib/libcusolver.so.10 $CONDA_PREFIX/lib/libcusolver.so.11

# test tensorflow gpu setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
