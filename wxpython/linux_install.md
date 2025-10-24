linux安装方法（linux mint已测试可用，其他系统未测试）
```
curl -L -o sensevox.py https://raw.githubusercontent.com/dapanggougou/sensevox/main/wxpython/sensevox_wx_gtcrn.py

curl -LsSf https://astral.sh/uv/install.sh | sh

uv v -p 3.13.9

source .venv/bin/activate

uv pip install wxpython pyperclip miniaudio keyboard sherpa_onnx opencc numpy

sudo apt install xclip
sudo apt install wl-clipboard

sudo .venv/bin/python sensevox.py
```
（构建wxpython或许要10分钟以上，如果报错可以安装下面的依赖，还是报错的话问ai，缺啥补啥，这几个python库能安装上就能跑）
```
sudo apt update
sudo apt install -y \
    libgtk-3-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libexpat1-dev \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxrender-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxcomposite-dev \
    libxfixes-dev \
    libxtst-dev \
    libasound2-dev \
    libpulse-dev \
    libnotify-dev \
    libsm-dev \
    build-essential \
    pkg-config \
    python3-dev
```
