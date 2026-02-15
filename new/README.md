**环境搭建**

**注意：** 项目路径请勿包含中文字符，否则会导致配置加载失败。

**可选：** 使用 `uv` 或直接安装均可。

```bash
uv venv --python=3.12.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

*注：更高版本的 Python 也可运行，但 OpenCC 和 miniaudio 缺少预编译轮子，编译后正常使用。*

**运行**
```bash
python sensevox.py
```

**打包**
```bash
pyinstaller -w sensevox.py -i icon.ico --hidden-import=_cffi_backend
```

**模型配置**

1. 下载模型：[model.onnx](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/blob/main/model.onnx)
2. 放置路径：`sensevox\_internal\assets\sensevoicesmallonnx\model.onnx`

---

**Environment Setup**

**Note:** The project path must not contain Chinese characters, as this will cause configuration loading to fail.

**Optional:** You can use `uv` or install directly.

```bash
uv venv --python=3.12.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

*Note: Higher versions of Python can also be used, but OpenCC and miniaudio lack precompiled wheels; compiling afterward should work normally.*

**Run**  
```bash
python sensevox.py
```

**Package**  
```bash
pyinstaller -w sensevox.py -i icon.ico --hidden-import=_cffi_backend
```

**Model Configuration**  

1. Download model: [model.onnx](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/blob/main/model.onnx)
2. Place path: `sensevox\_internal\assets\sensevoicesmallonnx\model.onnx`

