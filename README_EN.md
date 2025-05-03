# SenseVox

**SenseVox** is an offline speech recognition tool for Windows, built using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [FunASR SenseVoice](https://github.com/FunAudioLLM/SenseVoice), and [flet](https://github.com/flet-dev/flet/). This project aims to provide users with an efficient and convenient offline speech recognition experience.

<img src="https://github.com/user-attachments/assets/84f46047-d144-4cc3-976b-24670f66e463" alt="Sample Image" width="250"/>

## Features

- **Offline Speech Recognition**: Ensures data privacy
- **Multilingual Support**: Chinese, Cantonese, English, Japanese, Korean
- **Built-in 200MB Model**: Supports swapping with larger models for improved accuracy
- **User-Friendly**: Download, unzip, and run
- **Technical Foundation**: Based on sherpa-onnx and FunASR SenseVoice
- **Compatibility**: Supports Windows 10 (other versions untested)

## Usage

1. **Download the Package**
   Download the latest version from the [Releases](https://github.com/dapanggougou/sensevox/releases) page.

2. **Unzip**
   Extract the package to any directory **(Note: Path must be in English)**.

3. **Launch the Application**
   Run **sensevox.exe**.

4. **Hotkey Setup**
   - Click *Set Hotkey* and press a key.
   - Default hotkey: **Spacebar**
   - Recommended: **Caps Lock**
   - Combination keys are supported.

5. **Minimize to Tray/Background**
   - Use third-party tools like [Traymond](https://github.com/fcFn/traymond) or [RBTray](https://sourceforge.net/projects/rbtray/) to minimize to the system tray.
   - Use **Win+Tab** or **multi-finger swipe up on touchpad** to move the app to another desktop.

## Switching Recording Devices

To select a device in Windows Settings:
1. Open **Windows Settings**.
2. Go to **System** > **Sound**.
   Alternatively, right-click the speaker icon and select **Open Sound Settings** for quick access.

No devices tested; results unknown.

## Model Files

- The package includes a default 200MB model.
- For higher accuracy, download the 900MB model:
  - **[Model Link](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)**
- Replace the `model.onnx` file:
  - Path: `_internal/assets/sensevoicesmallonnx/model.onnx`

---

### Miscellaneous
Will multiple models be supported? No.

Whisper was tested with whisper.cpp Vulkan version. Even the smallq5 model had a 0.6-second delay, making instant input impossible. GPU acceleration is not universal and may increase package size. Not considered.

Paraformer-zh performs slightly better for Chinese. If the Mandarin and microphone quality are good, results are similar and highly accurate. No plans to add it.

Considered using sensevoice.cpp, which seems faster, but unsure how to integrate with Python.

---

Thanks again to FunASR SenseVoice, sherpa-onnx, CapsWriter-Offline, flet, and Gemini 2.5 Pro.