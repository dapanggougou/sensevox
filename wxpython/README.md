**感想**

先用`qwen-3-235b-a22b-instruct-2507`写wxpython界面，然后让`gemini2.5pro`把flet的功能照搬过来，多模型版本由`qwen-3-coder-480b`修改。

wxpython优点是简洁点、流畅点、打包快得多、体积内存占用小点、也不会有flet首次启动要向回环地址联网的问题、wx代码比flet少了200多行、对ai来说wx的知识比flet多得多，面向ai友好。

缺点是载入模型的几秒会阻塞主界面，不过不影响后续使用，懒得改。

FireRed模型太大了，也懒得加。

***	
**打包**

安装依赖 `pip install numpy pyaudio keyboard sherpa-onnx wxPython`

打包 `pyinstaller --windowed --name=sensevox_wx sensevox_wx.py`

打包多模型版 `pyinstaller --windowed --name=sensevox_wx sensevox_wx_multi_model.py`
***	
模型下载地址：`https://huggingface.co/csukuangfj/models`

dolphin模型放在 dolphinmodel 文件夹

paraformer模型放在 paraformermodel 文件夹

sensevoice位置不变
***	
<img src="./sensevox_wx.jpg" alt="SenseVox 微信图片" width="400" />

<img src="./sensevox_wx_multi_model.jpg" alt="SenseVox 微信图片" width="400" />
