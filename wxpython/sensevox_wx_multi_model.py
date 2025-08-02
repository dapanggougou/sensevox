import wx
import ctypes
import threading
import time
import os
import sys
import datetime
import re
import traceback
import wave
from collections import deque

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except (ImportError, PermissionError):
    KEYBOARD_AVAILABLE = False

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False

import numpy as np

# 设置高DPI支持
try:
    ctypes.windll.shcore.SetProcessDpiAwarenessContext(-4)
except (AttributeError, OSError):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

def get_asset_path(relative_path):
    """获取资源文件路径"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        try: 
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError: 
            base_path = os.path.abspath(".")
    return os.path.join(base_path, "assets", relative_path)

# 模型配置
SENSEVOICE_MODEL_DIR_BASE = "sensevoicesmallonnx"
PARAFORMER_MODEL_DIR_BASE = "paraformermodel"
DOLPHIN_MODEL_DIR_BASE = "dolphinmodel"
MODEL_DIR = get_asset_path(SENSEVOICE_MODEL_DIR_BASE)
PARAFORMER_MODEL_DIR = get_asset_path(PARAFORMER_MODEL_DIR_BASE)
DOLPHIN_MODEL_DIR = get_asset_path(DOLPHIN_MODEL_DIR_BASE)

MODEL_FILENAME = "model.onnx"
TOKENS_FILENAME = "tokens.txt"

MODEL_FILE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TOKENS_FILE_PATH = os.path.join(MODEL_DIR, TOKENS_FILENAME)
PARAFORMER_MODEL_FILE_PATH = os.path.join(PARAFORMER_MODEL_DIR, MODEL_FILENAME)
PARAFORMER_TOKENS_FILE_PATH = os.path.join(PARAFORMER_MODEL_DIR, TOKENS_FILENAME)
DOLPHIN_MODEL_FILE_PATH = os.path.join(DOLPHIN_MODEL_DIR, MODEL_FILENAME)
DOLPHIN_TOKENS_FILE_PATH = os.path.join(DOLPHIN_MODEL_DIR, TOKENS_FILENAME)

# 音频配置
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MIN_RECORD_SECONDS = 0.3

# 热键配置
DEFAULT_HOTKEY = "space"
HOTKEY_FILE = "hotkey.txt"
HOTKEY_FILE_PATH = get_asset_path(HOTKEY_FILE)

# 文件名配置
MAX_FILENAME_TEXT_LEN = 15
all_punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』“”‘’·～《》〈〉﹏——……〜・〝〟‹›'''

# 全局变量
p = None
audio_stream = None
model = None
is_listening = False
listener_thread = None
is_capturing_hotkey = False
capture_thread = None
recognizer_stream = None

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="神色语音sensevox", size=(500, 450))
      
        # 设置图标
        icon_path = get_asset_path("app_icon.ico")
        if os.path.exists(icon_path):
            icon = wx.Icon(icon_path, wx.BITMAP_TYPE_ICO)
            self.SetIcon(icon)
        else:
            print(f"Warning: Icon file not found at {icon_path}")
          
        self.current_hotkey = self.load_hotkey()
        self.current_model = "sensevoice"
        self.current_language = "auto"
        self.model_initialized = False

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
      
        # 添加模型选择ComboBox
        model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_label = wx.StaticText(panel, label="模型 Model：")
        self.model_combo = wx.ComboBox(
            panel,
            value=self.current_model,
            choices=["sensevoice", "paraformer", "dolphin"],
            style=wx.CB_READONLY
        )
        model_sizer.Add(self.model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        model_sizer.Add(self.model_combo, 1, wx.EXPAND)
        main_sizer.Add(model_sizer, 0, wx.ALL | wx.EXPAND, 15)
      
        # 添加分割线
        main_sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 10)
      
        grid_sizer = wx.FlexGridSizer(rows=1, cols=3, hgap=8, vgap=8)
        grid_sizer.AddGrowableCol(0, 1)
        grid_sizer.AddGrowableCol(1, 1)
        grid_sizer.AddGrowableCol(2, 1)

        self.btn_start = wx.Button(panel, label="Start 启动")
        self.btn_stop = wx.Button(panel, label="Stop 暂停")
        self.btn_hotkey = wx.Button(panel, label=f"Hotkey: {self.current_hotkey}")
      
        for btn in [self.btn_start, self.btn_stop, self.btn_hotkey]:
            btn.SetMinSize((-1, 34))
          
        grid_sizer.Add(self.btn_start, 0, wx.EXPAND | wx.ALL, 2)
        grid_sizer.Add(self.btn_stop, 0, wx.EXPAND | wx.ALL, 2)
        grid_sizer.Add(self.btn_hotkey, 0, wx.EXPAND | wx.ALL, 2)

        main_sizer.Add(grid_sizer, 0, wx.ALL | wx.EXPAND, 10)
      
        # 语言选择框
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lang_label = wx.StaticText(panel, label="语言 Language：")
        self.combo_box = wx.ComboBox(
            panel,
            value="auto",
            choices=["auto", "zh", "en", "ja", "ko", "yue"],
            style=wx.CB_READONLY
        )
        lang_sizer.Add(self.lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        lang_sizer.Add(self.combo_box, 1, wx.EXPAND)

        main_sizer.Add(lang_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 15)
      
        # 根据模型更新语言选择UI状态
        self.update_language_ui_state()

        self.log_text = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL
        )
        self.log_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
      
        main_sizer.Add(self.log_text, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 15)
      
        bottom_grid = wx.FlexGridSizer(rows=1, cols=2, hgap=8, vgap=8)
        bottom_grid.AddGrowableCol(0, 1)
        bottom_grid.AddGrowableCol(1, 1)

        self.btn_clear_log = wx.Button(panel, label="Clear 清除日志")
        self.btn_copy_log = wx.Button(panel, label="Copy 复制日志")
      
        for btn in [self.btn_clear_log, self.btn_copy_log]:
            btn.SetMinSize((-1, 34))
          
        bottom_grid.Add(self.btn_clear_log, 0, wx.EXPAND | wx.ALL, 2)
        bottom_grid.Add(self.btn_copy_log, 0, wx.EXPAND | wx.ALL, 2)

        main_sizer.Add(bottom_grid, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        panel.SetSizer(main_sizer)
      
        # 绑定事件
        self.BindEvents()
        self.SetMinSize((400, 400))
        self.Centre()
      
        # 检查和更新UI状态
        self.run_initial_checks()
        self.update_ui_state()

    def BindEvents(self):
        """绑定所有事件处理函数"""
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start_listening)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop_listening)
        self.btn_hotkey.Bind(wx.EVT_BUTTON, self.on_set_hotkey)
        self.btn_clear_log.Bind(wx.EVT_BUTTON, self.on_clear_log)
        self.btn_copy_log.Bind(wx.EVT_BUTTON, self.on_copy_log)
        self.combo_box.Bind(wx.EVT_COMBOBOX, self.on_combo_select)
        self.model_combo.Bind(wx.EVT_COMBOBOX, self.on_model_select)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def log(self, message, level="INFO"):
        """添加日志信息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp} {level}] {message}\n"
        self.log_text.AppendText(log_entry)

    def update_ui_state(self):
        """更新UI控件状态"""
        global is_listening, is_capturing_hotkey
        self.btn_start.Enable(not is_listening and not is_capturing_hotkey)
        self.btn_stop.Enable(is_listening and not is_capturing_hotkey)
        self.btn_hotkey.Enable(not is_listening and not is_capturing_hotkey)
        self.model_combo.Enable(not is_listening and not is_capturing_hotkey)
        self.combo_box.Enable(not is_listening and not is_capturing_hotkey and self.current_model == "sensevoice")
      
        if is_capturing_hotkey:
            self.btn_hotkey.SetLabel("Capturing...")
        else:
            self.btn_hotkey.SetLabel(f"Hotkey: {self.current_hotkey}")

    def update_language_ui_state(self):
        """根据模型启用/禁用语言选择 UI"""
        if self.current_model == "sensevoice":
            self.combo_box.Enable(True)
            self.lang_label.Enable(True)
        else:  # paraformer or dolphin
            self.combo_box.Enable(False)
            self.lang_label.Enable(False)

    def on_model_select(self, event):
        """处理模型选择变化"""
        self.current_model = self.model_combo.GetValue()
        self.log(f"Model changed to: {self.current_model}", "INFO")
        # 当模型更改时，标记为未初始化，下次启动时会重新加载
        self.model_initialized = False
        self.update_language_ui_state()

    def on_start_listening(self, event):
        global is_listening, listener_thread, model, p, audio_stream
      
        if not self.run_initial_checks():
            wx.MessageBox("Initial checks failed. Please see the log for details.", "Error", wx.OK | wx.ICON_ERROR)
            return

        if is_listening:
            self.log("Listener already running.", "WARNING")
            return

        if p is None:
            self.log("Initializing PyAudio...", "INFO")
            try:
                p = pyaudio.PyAudio()
            except Exception as e:
                self.log(f"Failed to initialize PyAudio: {e}", "ERROR")
                p = None
                self.update_ui_state()
                return

        try:
            self.log("Opening PyAudio stream...", "INFO")
            audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=CHUNK, start=False)
            audio_stream.start_stream()
        except Exception as e:
            self.log(f"Failed to open PyAudio stream: {e}", "ERROR")
            audio_stream = None
            self.update_ui_state()
            return

        # 检查是否需要重新初始化模型
        if not self.model_initialized or model is None:
            # 卸载之前的模型
            if model is not None:
                model = None
                self.log("Previous model unloaded.", "INFO")
          
            self.log("Loading Sherpa-ONNX recognizer...", "INFO")
            try:
                if self.current_model == "sensevoice":
                    model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                        model=MODEL_FILE_PATH, tokens=TOKENS_FILE_PATH, 
                        language=self.current_language if self.current_language != "auto" else "",
                        use_itn=True, num_threads=1, provider="cpu"
                    )
                elif self.current_model == "paraformer":
                    model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                        paraformer=PARAFORMER_MODEL_FILE_PATH,
                        tokens=PARAFORMER_TOKENS_FILE_PATH,
                        num_threads=2,
                        decoding_method="greedy_search",
                        sample_rate=16000,
                        feature_dim=80,
                        provider="cpu",
                    )
                elif self.current_model == "dolphin":
                    model = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
                        model=DOLPHIN_MODEL_FILE_PATH,
                        tokens=DOLPHIN_TOKENS_FILE_PATH,
                        num_threads=2,
                        decoding_method="greedy_search",
                        debug=False,
                        provider="cpu"
                    )
                self.log("Recognizer loaded successfully.", "SUCCESS")
                self.model_initialized = True
            except Exception as e:
                self.log(f"Recognizer loading failed: {e}", "ERROR")
                model = None
                if audio_stream:
                    audio_stream.stop_stream()
                    audio_stream.close()
                    audio_stream = None
                self.update_ui_state()
                return
      
        is_listening = True
        self.log(f"Starting listener thread (Hotkey: '{self.current_hotkey}')...", "INFO")
        listener_thread = threading.Thread(target=self.listen_loop, args=(audio_stream, p, model), daemon=True)
        listener_thread.start()
        self.update_ui_state()

    def on_stop_listening(self, event):
        global is_listening, listener_thread, audio_stream
        if not is_listening:
            return
      
        self.log("Stopping listener...", "INFO")
        is_listening = False

        if listener_thread and listener_thread.is_alive():
            listener_thread.join(timeout=1.0)
      
        if audio_stream:
            try:
                if audio_stream.is_active():
                    audio_stream.stop_stream()
                audio_stream.close()
            except Exception as e:
                self.log(f"Error closing audio stream: {e}", "WARNING")
            finally:
                audio_stream = None
      
        self.log("Listener stopped.", "INFO")
        self.update_ui_state()

    def on_set_hotkey(self, event):
        global is_capturing_hotkey, capture_thread
        if is_capturing_hotkey:
            return
      
        is_capturing_hotkey = True
        self.update_ui_state()
        self.log("Hotkey capture active. Press desired key/combination...", "INFO")
        capture_thread = threading.Thread(target=self.capture_hotkey_thread_func, daemon=True)
        capture_thread.start()

    def on_combo_select(self, event):
        self.current_language = self.combo_box.GetValue()
        self.log(f"Language preference set to: '{self.current_language}'", "INFO")
        self.log("Language will be used on next start.", "DEBUG")

    def on_clear_log(self, event):
        self.log_text.Clear()
        self.log("Logs cleared.", "INFO")

    def on_copy_log(self, event):
        log_content = self.log_text.GetValue()
        if log_content:
            clipboard_data = wx.TextDataObject(log_content)
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(clipboard_data)
                wx.TheClipboard.Close()
                self.log("Logs copied to clipboard.", "SUCCESS")
            else:
                wx.MessageBox("Cannot access clipboard.", "Error", wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox("Log is empty.", "Info", wx.OK | wx.ICON_INFORMATION)

    def on_close(self, event):
        global is_listening, p, audio_stream
        self.log("Window closing, cleaning up...", "INFO")
      
        is_listening = False
        if listener_thread and listener_thread.is_alive():
            listener_thread.join(timeout=1.0)
          
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
                self.log("Keyboard unhooked.", "DEBUG")
            except Exception:
                pass

        if audio_stream:
            if audio_stream.is_active(): 
                audio_stream.stop_stream()
            audio_stream.close()
        if p:
            p.terminate()

        self.Destroy()

    def listen_loop(self, stream, pyaudio_instance, current_model):
        global is_listening
        wx.CallAfter(self.log, f"Listener thread started. Monitoring hotkey: '{self.current_hotkey}'.")
        while is_listening:
            try:
                self.perform_record_and_transcribe(stream, pyaudio_instance, current_model)
                time.sleep(0.02)
            except Exception as e:
                if is_listening:
                    wx.CallAfter(self.log, f"Error in listener loop: {e}", "ERROR")
                    traceback.print_exc()
                break
        wx.CallAfter(self.log, "Listener thread finished.")

    def perform_record_and_transcribe(self, stream, pyaudio_instance, current_model):
        global is_listening
        audio_data = None
        blocked_keys = []
        hotkey_parts = [part.strip() for part in self.current_hotkey.split('+')]
        is_caps_lock_hotkey = (self.current_hotkey.lower() == 'caps lock')

        try:
            audio_data, duration = self.record_audio(stream, self.current_hotkey, hotkey_parts, blocked_keys, is_caps_lock_hotkey)
          
            if audio_data and is_listening:
                transcription_result = self.transcribe_local(audio_data, current_model, self.current_language)
                self.save_audio_with_transcription(audio_data, transcription_result, pyaudio_instance)
              
                if transcription_result and "Transcription failed:" not in transcription_result:
                    processed_text = self.process_text(transcription_result)
                    self.type_text(processed_text)
                    if is_caps_lock_hotkey:
                        try:
                            keyboard.press_and_release('caps lock')
                        except Exception as e:
                            wx.CallAfter(self.log, f"Failed to toggle Caps Lock state: {e}", "WARNING")

        except Exception as e:
            if is_listening:
                wx.CallAfter(self.log, f"Error during record/process cycle: {e}", "ERROR")
                traceback.print_exc()
        finally:
            if blocked_keys:
                wx.CallAfter(self.log, f"Unblocking keys: {blocked_keys}", "DEBUG")
                for key_part in blocked_keys:
                    try:
                        keyboard.unblock_key(key_part)
                    except Exception as e:
                        wx.CallAfter(self.log, f"Could not unblock '{key_part}': {e}", "WARNING")

    def record_audio(self, stream, hotkey, hotkey_parts, blocked_keys_list, is_caps_lock):
        global is_listening
        frames = deque()
        recording_started = False
        start_time = 0

        while is_listening:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if keyboard.is_pressed(hotkey):
                    if not recording_started:
                        wx.CallAfter(self.log, f"Recording started (hotkey '{hotkey}' detected)...", "DEBUG")
                        start_time = time.time()
                        recording_started = True

                        if not is_caps_lock:
                            for key_part in hotkey_parts:
                                try:
                                    keyboard.block_key(key_part)
                                    blocked_keys_list.append(key_part)
                                except Exception as e:
                                    wx.CallAfter(self.log, f"Could not block '{key_part}': {e}", "WARNING")
                    frames.append(data)
                else:
                    if recording_started:
                        break
            except IOError:
                wx.CallAfter(self.log, "Audio input overflowed!", "WARNING")
                break
            except Exception as e:
                wx.CallAfter(self.log, f"Error reading audio stream: {e}", "ERROR")
                break
      
        if not recording_started:
            return None, 0.0
          
        duration = time.time() - start_time
        if duration < MIN_RECORD_SECONDS and is_listening:
            wx.CallAfter(self.log, f"Recording too short ({duration:.2f}s), ignored.", "WARNING")
            return None, duration
      
        return b''.join(frames), duration

    def transcribe_local(self, audio_input, recognizer, language):
        global recognizer_stream
        if not recognizer or not SHERPA_AVAILABLE:
            return "Recognizer unavailable."
      
        recognizer_stream = recognizer.create_stream()
        audio_array = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
      
        start_time = time.perf_counter()
        try:
            recognizer_stream.accept_waveform(sample_rate=RATE, waveform=audio_array)
            tail_padding = np.zeros(int(0.5 * RATE), dtype=np.float32)
            recognizer_stream.accept_waveform(sample_rate=RATE, waveform=tail_padding)
            recognizer.decode_stream(recognizer_stream)
            transcribed_text = recognizer_stream.result.text
        except Exception as e:
            wx.CallAfter(self.log, f"Sherpa-ONNX transcription error: {e}", "ERROR")
            return f"Transcription failed: {e}"
      
        duration = time.perf_counter() - start_time
        wx.CallAfter(self.log, f"识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")
        return transcribed_text if transcribed_text else None

    def type_text(self, text_to_type):
        if not text_to_type:
            return
        wx.CallAfter(self.log, "Typing text...", "INFO")
        try:
            keyboard.write(text_to_type)
            wx.CallAfter(self.log, f"Typed: '{text_to_type}'", "SUCCESS")
        except Exception as e:
            wx.CallAfter(self.log, f"Keyboard input failed: {e}", "ERROR")

    def capture_hotkey_thread_func(self):
        global is_capturing_hotkey
        try:
            new_hotkey = keyboard.read_hotkey(suppress=False)
            if len(new_hotkey) > 30:
                wx.CallAfter(self.log, "Captured hotkey is too long, ignoring.", "WARNING")
                return

            self.current_hotkey = new_hotkey
            wx.CallAfter(self.log, f"Captured: '{new_hotkey}'", "SUCCESS")
            self.save_hotkey(self.current_hotkey)
        except Exception as e:
            wx.CallAfter(self.log, f"Hotkey capture failed: {e}", "ERROR")
        finally:
            is_capturing_hotkey = False
            wx.CallAfter(self.update_ui_state)

    def run_initial_checks(self):
        self.log("Running initial checks...", "INFO")
        all_ok = True
        if not SHERPA_AVAILABLE:
            self.log("Sherpa-ONNX library not found.", "ERROR"); all_ok = False
        if not PYAUDIO_AVAILABLE:
            self.log("PyAudio library not found.", "ERROR"); all_ok = False
        if not KEYBOARD_AVAILABLE:
            self.log("Keyboard library not found or permission error.", "ERROR"); all_ok = False
        if not os.path.isfile(MODEL_FILE_PATH):
            self.log(f"SenseVoice Model file not found: {MODEL_FILE_PATH}", "ERROR"); all_ok = False
        if not os.path.isfile(TOKENS_FILE_PATH):
            self.log(f"SenseVoice Tokens file not found: {TOKENS_FILE_PATH}", "ERROR"); all_ok = False
        if not os.path.isfile(PARAFORMER_MODEL_FILE_PATH):
            self.log(f"Paraformer Model file not found: {PARAFORMER_MODEL_FILE_PATH}", "ERROR"); all_ok = False
        if not os.path.isfile(PARAFORMER_TOKENS_FILE_PATH):
            self.log(f"Paraformer Tokens file not found: {PARAFORMER_TOKENS_FILE_PATH}", "ERROR"); all_ok = False
        if not os.path.isfile(DOLPHIN_MODEL_FILE_PATH):
            self.log(f"Dolphin Model file not found: {DOLPHIN_MODEL_FILE_PATH}", "ERROR"); all_ok = False
        if not os.path.isfile(DOLPHIN_TOKENS_FILE_PATH):
            self.log(f"Dolphin Tokens file not found: {DOLPHIN_TOKENS_FILE_PATH}", "ERROR"); all_ok = False
      
        if all_ok:
             self.log("All checks passed. Ready to start.", "SUCCESS")
        else:
             self.log("One or more initial checks failed. Please resolve the errors.", "ERROR")
        return all_ok

    def load_hotkey(self):
        try:
            if os.path.exists(HOTKEY_FILE_PATH):
                with open(HOTKEY_FILE_PATH, 'r', encoding='utf-8') as f:
                    return f.read().strip() or DEFAULT_HOTKEY
        except Exception:
            pass
        return DEFAULT_HOTKEY

    def save_hotkey(self, key_to_save):
        try:
            os.makedirs(os.path.dirname(HOTKEY_FILE_PATH), exist_ok=True)
            with open(HOTKEY_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(key_to_save)
            wx.CallAfter(self.log, f"Hotkey '{key_to_save}' saved.", "INFO")
        except Exception as e:
            wx.CallAfter(self.log, f"Failed to save hotkey: {e}", "ERROR")

    def process_text(self, text):
        if not isinstance(text, str): return text
        punctuation_count = sum(1 for char in text if char in all_punctuation)
        if punctuation_count <= 1:
            return text.translate(str.maketrans('', '', all_punctuation))
        return text

    def sanitize_filename_part(self, text, max_len=MAX_FILENAME_TEXT_LEN):
        if not text or not isinstance(text, str): return ""
        sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]+', '', text)
        sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
        return sanitized[:max_len].strip('_') if len(sanitized) > max_len else sanitized

    def save_audio_with_transcription(self, audio_data, transcription, pyaudio_instance):
        if not audio_data: return
        try:
            recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "录音")
            os.makedirs(recordings_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_text = self.sanitize_filename_part(transcription)
            filename = f"rec_{timestamp}_{sanitized_text}.wav" if sanitized_text else f"rec_{timestamp}.wav"
            filepath = os.path.join(recordings_dir, filename)

            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)
            wx.CallAfter(self.log, f"Recording saved: {filename}", "INFO")
        except Exception as e:
            wx.CallAfter(self.log, f"Error saving audio: {e}", "ERROR")

if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()