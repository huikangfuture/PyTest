```py
print('Hello World!')
```
```py
nano ~/.config/user-dirs.dirs

XDG_DESKTOP_DIR="$HOME/Desktop"
XDG_DOWNLOAD_DIR="$HOME/Downloads"
XDG_TEMPLATES_DIR="$HOME/Templates"
XDG_PUBLICSHARE_DIR="$HOME/Public"
XDG_DOCUMENTS_DIR="$HOME/Documents"
XDG_MUSIC_DIR="$HOME/Music"
XDG_PICTURES_DIR="$HOME/Pictures"
XDG_VIDEOS_DIR="$HOME/Videos"
```
```py
# --------------------------------------------------------------------------------- #
pip list -o --format=columns
pip install --ignore-installed --upgrade tensorflow

[global]
timeout = 60
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
# --------------------------------------------------------------------------------- #
conda info -e

conda create -n pytorch python=3.6
conda create -n tensorflow python=3.6

conda remove -n pytorch --all
conda remove -n tensorflow --all

conda install -n pytorch pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -n tensorflow tensorflow-gpu

conda env create -f D:\Apps\Conda\pytorch.yml
conda env create -f D:\Apps\Conda\tensorflow.yml

https://conda.anaconda.org/pytorch/win-64/pytorch-1.1.0-py3.6_cuda100_cudnn7_1.tar.bz2
# --------------------------------------------------------------------------------- #
https://www.google.cn/intl/zh-CN/chrome/?standalone=1
# --------------------------------------------------------------------------------- #
from keras.utils import to_categorical
image = cv2.imread('image.png')
image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)

import os
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

while (cap.isOpened()):
    retval, frame = cap.read()
    cv2.imshow('frame', frame)
    if (cv2.waitKey(33) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
# --------------------------------------------------------------------------------- #
sudo modprobe -r ideapad_laptop
sudo nano /etc/modprobe.d/blacklist.conf
blacklist ideapad_laptop

sudo apt update && sudo apt upgrade

ubuntu-drivers devices
sudo add-apt-repository -y ppa:graphics-drivers/ppa && sudo apt update
sudo apt install -y nvidia-driver-430

sudo sh cuda_10.0.130_410.48_linux.run
sudo sh cuda_10.0.130.1_linux.run

sudo nano ~/.bashrc
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc

tar -zxvf cudnn-10.0-linux-x64-v7.6.1.34.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
sudo chmod a+r /usr/local/cuda/include/cudnn.h

sudo apt install system76-cuda-10.0
sudo apt install system76-cudnn-10.0

curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt install -y nodejs && sudo apt install -y build-essential gdb

sudo add-apt-repository -y ppa:git-core/ppa && sudo apt update
sudo apt install -y git

rm -rf ~/.vscode-server/
rm -rf ~/.vscode-server/extensions/
mkdir -p ~/.vscode-server/extensions/
# --------------------------------------------------------------------------------- #
```
```py
name: dev
channels:
  - pytorch
  - simpleitk
  - defaults
dependencies:
  - python=3.6
  - pip
  - tqdm
  - numpy
  - scipy
  - opencv
  - pandas
  - scrapy
  - jupyter
  - autopep8
  - openpyxl
  - simpleitk
  - xmltodict
  - matplotlib
  - scikit-image
  - scikit-learn
  - beautifulsoup4
  - pytorch
  - torchvision
  - tensorflow-gpu
  - cudatoolkit=10.0
  - pip:
    - pynrrd
```
```json
{
    "editor.accessibilitySupport": "off",
    "editor.renderWhitespace": "all",
    "explorer.autoReveal": true,
    "explorer.openEditors.visible": 0,
    "extensions.autoUpdate": false,
    "files.eol": "\n",
    "files.trimTrailingWhitespace": true,
    "git.autofetch": true,
    "terminal.integrated.cursorBlinking": true,
    "terminal.integrated.cursorStyle": "line",
    "terminal.integrated.scrollback": 100000,
    "update.mode": "manual",
    "update.showReleaseNotes": false,
    "workbench.colorTheme": "One Dark Pro",
    "workbench.iconTheme": "vscode-icons",
    "workbench.settings.editor": "json",
    "workbench.settings.openDefaultKeybindings": true,
    "workbench.settings.openDefaultSettings": true,
    "workbench.settings.useSplitJSON": true,
    "workbench.startupEditor": "none",
    "code-runner.runInTerminal": true,
    "python.linting.enabled": false,
    "python.terminal.activateEnvironment": false
}
```
