# 神色语音 SenseVox

**神色语音 SenseVox** 是一款基于 Windows 的离线语音识别工具，利用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[FunASR SenseVoice](https://github.com/FunAudioLLM/SenseVoice) 和 [flet](https://github.com/flet-dev/flet/) 实现。本项目旨在为用户提供高效、便捷的离线语音识别体验。

<img src="https://github.com/user-attachments/assets/84f46047-d144-4cc3-976b-24670f66e463" alt="示例图片" width="250"/>

## 特性

- **离线语音识别**：保障数据隐私
- **多语言支持**：中文、粤语、英文、日语、韩语
- **内置200MB模型**：支持替换更大的模型以提高识别准确率
- **简单易用**：下载、解压、启动即可
- **技术基础**：基于 sherpa-onnx 和 FunASR SenseVoice 技术
- **兼容性**：支持 Windows 10，其他版本未测试

## 使用方法

1. **下载压缩包**
   从本项目的 [Releases](https://github.com/dapanggougou/sensevox/releases) 页面下载最新版本的压缩包。

2. **解压**
   解压下载的压缩包至任意目录（注意：路径中请避免使用中文）。

3. **启动应用**
   运行 **神色语音sensevox.exe**。

4. **快捷键设置**
   - 鼠标点击 Set Hotkey，再敲下键盘按键。
   - 默认快捷键为 **空格键 (Space)**
   - 建议更改为 **大小写切换键 (Caps Lock)**
   - 快捷键可以是组合键。

5. **最小化托盘/后台**
   - 可使用第三方软件如 [Traymond](https://github.com/fcFn/traymond) 或 [RBTray](https://sourceforge.net/projects/rbtray/) 将程序最小化至托盘。
   - 使用 **Win+Tab** 或者 **触摸板多指上划** 将软件移动到另一个桌面。

## 切换录音设备

在 Windows 设置中，可以通过以下步骤选择设备：
1. 打开 **Windows 设置**。
2. 选择 **系统** > **声音**。
   或者右击扬声器图标，选择 **打开声音设置** 以快速进入该界面。

没有设备，未测试，效果未知。

## 模型文件说明

- 压缩包中默认包含200MB模型。
- 若需要更大模型以提升识别准确率，可以下载900MB模型：
  - **[模型链接](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)**
- 下载后替换 `model.onnx` 文件：
  - 存放路径：`_internal/assets/sensevoicesmallonnx/model.onnx`

---

### 其他
会支持多模型吗？不会。

whisper 测试过 whisper.cpp vulkan版，即使是smallq5的模型延迟也很高有0.6秒，做不到抬手秒输入。显卡加速的不太通用，还可能导致打包体积大。不考虑。

paraformer-zh 中文似乎好一点，但普通话ok的话，其实效果差不多，都很准。所以也不打算加。

考虑过使用 sensevoice.cpp，貌似更快，但是不知道怎么用在python里。

---

再次感谢 FunASR SenseVoice、sherpa-onnx、CapsWriter-Offline、flet、Gemini 2.5 Pro。
