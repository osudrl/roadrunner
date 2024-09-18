mode="$1"
varname="$2"

if [ -z "$mode" ]; then
    read -p "[u]pgrade or [c]reate new conda env: " mode
fi


echo $mode
if [ "$mode" == "u" ]; then
    conda env update --file environment.yaml --prune
    exit 0
elif [ "$mode" == "c" ]; then
    if [ -z "$varname" ]; then
        echo enter name for new conda environment
        read varname
    fi
    #build conda env
    conda env create -n $varname --file environment.yaml
    eval "$(conda shell.bash hook)"
    #activate conda env
    conda activate $varname
    if ! test -d ~/.mujoco/mujoco210; then
        echo "Installing Mujoco210."
        pwd="$PWD"
        echo installing mujoco v2.10
        mkdir -p ~/.mujoco
        cd ~/.mujoco 
        rm -rf mujoco210-linux-x86_64.tar.gz
        wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
        tar -xvzf mujoco210-linux-x86_64.tar.gz
        rm -rf mujoco210-linux-x86_64.tar.gz
        cd $pwd
    fi
else
    echo Choice not recognized.
    exit 1
fi
exit 0
