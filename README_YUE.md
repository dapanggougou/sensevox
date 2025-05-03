# 神色語音 SenseVox

**神色語音 SenseVox** 係一款基於 Windows 嘅離線語音識別工具，用咗 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[FunASR SenseVoice](https://github.com/FunAudioLLM/SenseVoice) 同 [flet](https://github.com/flet-dev/flet/) 實現。呢個項目旨在為用戶提供高效、方便嘅離線語音識別體驗。

<img src="https://github.com/user-attachments/assets/84f46047-d144-4cc3-976b-24670f66e463" alt="示範圖片" width="250"/>

## 功能特點

- **離線語音識別**：保障數據私隱
- **多語言支援**：中文、粵語、英文、日文、韓文
- **內置200MB模型**：支持替換更大模型以提高識別準確率
- **簡單易用**：下載、解壓、啟動就得
- **技術基礎**：基於 sherpa-onnx 同 FunASR SenseVoice 技術
- **兼容性**：支援 Windows 10，其他版本未測試

## 使用方法

1. **下載壓縮包**
   喺項目嘅 [Releases](https://github.com/dapanggougou/sensevox/releases) 頁面下載最新版本壓縮包。

2. **解壓**
   將下載嘅壓縮包解壓到任意目錄 **(注意：路徑必須全英文)**。

3. **啟動應用**
   運行 **神色語音sensevox.exe**。

4. **快捷鍵設置**
   - 用滑鼠點擊 Set Hotkey，再撳鍵盤按鍵。
   - 默認快捷鍵係 **空格鍵 (Space)**
   - 建議更改為 **大小寫切換鍵 (Caps Lock)**
   - 快捷鍵可以係組合鍵。

5. **最小化到系統托盤/後台**
   - 可以用第三方軟件好似 [Traymond](https://github.com/fcFn/traymond) 或者 [RBTray](https://sourceforge.net/projects/rbtray/) 將程序最小化到托盤。
   - 用 **Win+Tab** 或者 **觸控板多指上滑** 將軟件移動到另一個桌面。

## 切換錄音設備

喺 Windows 設置度，可以跟以下步驟選擇設備：
1. 打開 **Windows 設置**。
2. 選擇 **系統** > **聲音**。
   或者右擊喇叭圖標，選擇 **打開聲音設置** 快速進入呢個界面。

冇設備，未測試，效果未知。

## 模型文件說明

- 壓縮包默認包含200MB模型。
- 如果需要更大模型提升識別準確率，可以下載900MB模型：
  - **[模型下載連結](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)**
- 下載後替換 `model.onnx` 文件：
  - 存放路徑：`_internal/assets/sensevoicesmallonnx/model.onnx`

---

### 其他
會唔會支援多模型？唔會。

whisper 測試過 whisper.cpp vulkan版，就算係 smallq5 嘅模型延遲都好高有成0.6秒，做唔到撳掣即刻輸入。顯卡加速嘅唔太通用，仲可能令打包體積變大。唔考慮。

paraformer-zh 中文好似好啲，如果普通話同咪收音都OK嘅話，其實效果差唔多，都好準。所以都唔打算加。

考慮過用 sensevoice.cpp，好似快啲，但係唔知點樣喺python度用。

---

再次感謝 FunASR SenseVoice、sherpa-onnx、CapsWriter-Offline、flet、Gemini 2.5 Pro。