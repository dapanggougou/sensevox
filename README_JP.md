# SenseVox

**SenseVox** は、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[FunASR SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、および[flet](https://github.com/flet-dev/flet/)を利用したWindowsベースのオフライン音声認識ツールです。本プロジェクトは、ユーザーに効率的で便利なオフライン音声認識体験を提供することを目的としています。

<img src="https://github.com/user-attachments/assets/84f46047-d144-4cc3-976b-24670f66e463" alt="サンプル画像" width="250"/>

## 特徴

- **オフライン音声認識:** データプライバシーを保護
- **多言語対応:** 中国語（標準語）、広東語、英語、日本語、韓国語
- **内蔵200MBモデル:** より大きなモデルに置き換えることで認識精度向上が可能
- **簡単操作:** ダウンロード、解凍、起動の3ステップ
- **技術基盤:** sherpa-onnxとFunASR SenseVoice技術を採用
- **互換性:** Windows 10対応（他のバージョンは未検証）

## 使用方法

1. **圧縮ファイルのダウンロード**
   本プロジェクトの[Releases](https://github.com/dapanggougou/sensevox/releases)ページから最新バージョンの圧縮ファイルをダウンロード。

2. **解凍**
   ダウンロードした圧縮ファイルを任意のディレクトリに解凍 **（注意: パスはすべて半角英字である必要あり）**。

3. **アプリケーション起動**
   **神色语音sensevox.exe** を実行。

4. **ホットキー設定**
   - Set Hotkeyをマウスクリック後、任意のキーを押下。
   - デフォルトホットキーは **スペースキー (Space)**
   - **Caps Lockキー** への変更を推奨
   - ホットキーは組み合わせキーも可能。

5. **タスクトレイ/バックグラウンド化**
   - [Traymond](https://github.com/fcFn/traymond) や [RBTray](https://sourceforge.net/projects/rbtray/) などのサードパーティ製ソフトでタスクトレイに最小化可能。
   - **Win+Tab** または **タッチパッドのマルチフィンジャー上スワイプ** で別デスクトップに移動可能。

## 録音デバイスの切り替え

Windows設定で以下の手順によりデバイス選択可能:
1. **Windows設定** を開く。
2. **システム** > **サウンド** を選択。
   またはスピーカーアイコンを右クリックし、**サウンド設定を開く** で素早くアクセス可能。

未検証のため効果不明。

## モデルファイル説明

- 圧縮ファイルにはデフォルトで200MBモデルを含む。
- 認識精度向上のため900MBモデルが必要な場合、以下からダウンロード:
  - **[モデルリンク](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)**
- ダウンロード後、`model.onnx`ファイルを置換:
  - 配置パス: `_internal/assets/sensevoicesmallonnx/model.onnx`

---

### その他
複数モデルの対応予定は？ ありません。

whisperについてはwhisper.cppのvulkan版をテスト済みですが、smallq5モデルでも遅延が0.6秒と高く、瞬時入力は不可能でした。GPU加速版は汎用性に欠け、パッケージサイズも肥大化するため非採用。

paraformer-zhは中国語でやや優れていますが、アクセントとマイクの収音が良好であれば、効果は同程度でどちらも高精度。よって追加予定なし。

sensevoice.cppの使用も検討しましたが、Pythonでの実装方法が不明でした。

---

改めて、FunASR SenseVoice、sherpa-onnx、CapsWriter-Offline、flet、Gemini 2.5 Proに感謝申し上げます。