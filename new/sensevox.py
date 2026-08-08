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
import pyperclip
from collections import deque

# --- 依赖库安全导入 ---
try:
    import miniaudio
    MINIAUDIO_AVAILABLE = True
except ImportError:
    MINIAUDIO_AVAILABLE = False

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

try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

import numpy as np

# --- DPI 适配 ---
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
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_path = os.path.abspath(".")
    return os.path.join(base_path, "assets", relative_path)

# --- 核心音频参数配置 ---
CHUNK = 1024 * 2
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH_BYTES = 2
MIN_RECORD_SECONDS = 0.3
MAX_RECORD_SECONDS = 30.0
IDLE_POLL_SLEEP = 0.02
DEFAULT_HOTKEY = "space"
MAX_FILENAME_TEXT_LEN = 15
ALL_PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』""''·～《》〈〉﹏——……〜・〝〟‹›"""

# ── 终极预计算常量，压榨 CPU 性能极限 ──────────────────────────────────
_PUNCTUATION_SET = frozenset(ALL_PUNCTUATION)                     # O(1) 极速成员测试
_TRANSLATE_TABLE = str.maketrans("", "", ALL_PUNCTUATION)         # 缓存 C 级字符转换表
_TAIL_PADDING = np.zeros(int(0.5 * RATE), dtype=np.float32)       # 预计算半秒静音尾音
_RE_INVALID_CHARS = re.compile(r'[\\/*?:"<>|\n\r\t]+')            # 预编译正则：文件名非法字符
_RE_WHITESPACE = re.compile(r"\s+")                               # 预编译正则：空白字符
_INT16_SCALE = np.float32(1.0 / 32768.0)                          # 乘法代替除法，加速 Int16 -> Float32
_FLOAT32_SCALE = np.float32(32767.0)                              # 乘法代替除法，加速 Float32 -> Int16
_MAX_LOG_LINES = 100                                              # 最大日志行数

# --- 路径与文件 ---
MODEL_DIR = get_asset_path("sensevoicesmallonnx")
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENS_FILE_PATH = os.path.join(MODEL_DIR, "tokens.txt")
GTCRN_MODEL_PATH = get_asset_path("gtcrn_simple.onnx")

HOTKEY_FILE_PATH = get_asset_path("hotkey.txt")
GTCRN_CONFIG_PATH = get_asset_path("gtcrn_config.txt")
SAVE_RECORDING_CONFIG_PATH = get_asset_path("save_recording_config.txt")
OPENCC_CONFIG_PATH = get_asset_path("opencc_config.txt")
OPENCC_ENABLED_PATH = get_asset_path("opencc_enabled.txt")

OPENCC_OPTIONS = [
    "s2t.json 簡 → 繁", "t2s.json 繁 → 簡", "s2tw.json 簡 → 臺灣繁",
    "tw2s.json 臺灣繁 → 簡", "s2hk.json 簡 → 香港繁", "hk2s.json 香港繁 → 簡",
    "s2twp.json 簡 → 臺灣繁 + 臺灣用語", "tw2sp.json 臺灣繁 → 簡 + 大陸用語",
    "t2tw.json 繁 → 臺灣繁", "hk2t.json 香港繁 → 繁", "t2hk.json 繁 → 香港繁",
    "t2jp.json 繁 → 日文新字體", "jp2t.json 日文新字體 → 繁", "tw2t.json 臺灣繁 → 繁"
]

class AudioDeviceError(Exception):
    pass

class PyMiniaudioRecorder:
    def __init__(self, device_index=-1, frame_length=CHUNK, stall_timeout=1.0, max_queue_chunks=128):
        self.device_index = device_index
        self.frame_length = frame_length
        self.sample_rate = RATE
        self.channels = CHANNELS
        self.sample_width = SAMPLE_WIDTH_BYTES
        self.dev = None
        self.buffer = bytearray()
        self.queue = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = False
        self._gen = None
        self.received_data = False
        self.last_data_time = 0.0
        self.stall_timeout = stall_timeout
        self.max_queue_chunks = max_queue_chunks
        self._chunk_bytes = self.frame_length * self.channels * self.sample_width

    def _capture_generator(self):
        _ = yield
        chunk_bytes = self._chunk_bytes
        buf = self.buffer
        q = self.queue
        max_q = self.max_queue_chunks
        cond = self.cond
        monotonic = time.monotonic
        while True:
            data = yield
            if not data:
                continue
            with cond:
                buf.extend(data)
                while len(buf) >= chunk_bytes:
                    chunk = bytes(buf[:chunk_bytes])
                    del buf[:chunk_bytes]
                    q.append(chunk)
                    if len(q) > max_q:
                        q.popleft()
                self.received_data = True
                self.last_data_time = monotonic()
                cond.notify_all()

    def start(self):
        if self.running:
            return
        self.buffer = bytearray()
        self.queue = deque()
        self.running = True
        self.received_data = False
        self.last_data_time = time.monotonic()
        bs_msec = max(1, int(self.frame_length * 1000 / self.sample_rate))
        self.dev = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
            buffersize_msec=bs_msec,
        )
        self._gen = self._capture_generator()
        next(self._gen)
        self.dev.start(self._gen)

    def stop(self):
        if self.dev:
            try:
                self.dev.stop()
            except Exception:
                pass
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None
        with self.lock:
            self.running = False
            self.cond.notify_all()
        self._gen = None

    def wait_ready(self, timeout=1.0):
        end = time.monotonic() + timeout
        with self.lock:
            while self.running and not self.received_data:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self.cond.wait(timeout=min(0.1, remaining))
            return self.received_data

    def read(self):
        with self.lock:
            waited = 0.0
            while self.running and len(self.queue) == 0:
                self.cond.wait(timeout=0.1)
                waited += 0.1
                if self.received_data and (time.monotonic() - self.last_data_time) > self.stall_timeout:
                    raise AudioDeviceError("Audio device stalled")
                if not self.received_data and waited >= self.stall_timeout:
                    raise AudioDeviceError("No audio received")
            if len(self.queue) == 0:
                return bytes(self._chunk_bytes) # 极速生成全 0 bytes
            return self.queue.popleft()


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="神色语音sensevox", size=(500, 520))
        self.recorder = None
        self.model = None
        self.gtcrn_denoiser = None
        self.opencc_converter = None
        self.listening_event = threading.Event()
        self.capturing_hotkey_event = threading.Event()
        self.listener_thread = None
        self.was_listening_before_suspend = False
        self._log_line_count = 0 # 纯 Python 内存级行数计数器

        icon_path = get_asset_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_ICO))

        self.current_hotkey = self.load_setting(HOTKEY_FILE_PATH, DEFAULT_HOTKEY)
        initial_gtcrn_state = self.load_setting(GTCRN_CONFIG_PATH, "true").lower() == "true"
        initial_save_state = self.load_setting(SAVE_RECORDING_CONFIG_PATH, "true").lower() == "true"
        initial_opencc_state = self.load_setting(OPENCC_ENABLED_PATH, "false").lower() == "true"
        initial_opencc_config = self.load_setting(OPENCC_CONFIG_PATH, OPENCC_OPTIONS[0])

        self._init_ui(initial_gtcrn_state, initial_save_state, initial_opencc_state, initial_opencc_config)
        self.BindEvents()
        self.SetMinSize((450, 520))
        self.Centre()
        self.run_initial_checks()
        self.update_ui_state()

    def _init_ui(self, initial_gtcrn_state, initial_save_state, initial_opencc_state, initial_opencc_config):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

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

        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lang_label = wx.StaticText(panel, label="语言 Language：")
        self.combo_box = wx.ComboBox(panel, value="auto", choices=["auto", "zh", "en", "ja", "ko", "yue"], style=wx.CB_READONLY)
        self.current_language = "auto"
        lang_sizer.Add(self.lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        lang_sizer.Add(self.combo_box, 1, wx.EXPAND)
        main_sizer.Add(lang_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        self.chk_gtcrn_enhance = wx.CheckBox(panel, label="GTCRN 增强 (需要 assets/gtcrn_simple.onnx)")
        self.chk_gtcrn_enhance.SetValue(initial_gtcrn_state)
        main_sizer.Add(self.chk_gtcrn_enhance, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        self.chk_save_recording = wx.CheckBox(panel, label="保存录音 (保存在 '录音' 文件夹)")
        self.chk_save_recording.SetValue(initial_save_state)
        main_sizer.Add(self.chk_save_recording, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        opencc_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_opencc = wx.CheckBox(panel, label="OpenCC")
        self.chk_opencc.SetValue(initial_opencc_state)
        self.combo_opencc = wx.ComboBox(panel, value=initial_opencc_config, choices=OPENCC_OPTIONS, style=wx.CB_READONLY)
        opencc_sizer.Add(self.chk_opencc, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        opencc_sizer.Add(self.combo_opencc, 1, wx.EXPAND)
        main_sizer.Add(opencc_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        self.log_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.log_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main_sizer.Add(self.log_text, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

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

    def BindEvents(self):
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start_listening)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop_listening)
        self.btn_hotkey.Bind(wx.EVT_BUTTON, self.on_set_hotkey)
        self.btn_clear_log.Bind(wx.EVT_BUTTON, self.on_clear_log)
        self.btn_copy_log.Bind(wx.EVT_BUTTON, self.on_copy_log)
        self.combo_box.Bind(wx.EVT_COMBOBOX, self.on_combo_select)
        self.chk_opencc.Bind(wx.EVT_CHECKBOX, self.on_opencc_toggle)
        self.combo_opencc.Bind(wx.EVT_COMBOBOX, self.on_opencc_select)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        if hasattr(wx, "EVT_POWER_SUSPENDING"):
            self.Bind(wx.EVT_POWER_SUSPENDING, self.on_power_suspending)
        if hasattr(wx, "EVT_POWER_RESUME"):
            self.Bind(wx.EVT_POWER_RESUME, self.on_power_resume)

    def log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp} {level}] {message}\n"

        def append_and_prune():
            self.log_text.AppendText(log_entry)
            self._log_line_count += 1
            if self._log_line_count > _MAX_LOG_LINES:
                extra = self._log_line_count - _MAX_LOG_LINES
                end_pos = self.log_text.XYToPosition(0, extra)
                if end_pos != wx.NOT_FOUND and end_pos > 0:
                    self.log_text.Freeze() # 冻结 UI 绘制，防止频繁删减导致的闪烁
                    self.log_text.Remove(0, end_pos)
                    self.log_text.Thaw()   # 恢复 UI 绘制
                self._log_line_count = _MAX_LOG_LINES

        wx.CallAfter(append_and_prune)

    def update_ui_state(self):
        is_listening = self.listening_event.is_set()
        is_capturing_hotkey = self.capturing_hotkey_event.is_set()
        is_running_or_capturing = is_listening or is_capturing_hotkey
        
        self.btn_start.Enable(not is_running_or_capturing)
        self.btn_stop.Enable(is_listening and not is_capturing_hotkey)
        self.btn_hotkey.Enable(not is_running_or_capturing)
        self.combo_box.Enable(not is_running_or_capturing)
        self.chk_gtcrn_enhance.Enable(not is_running_or_capturing and os.path.exists(GTCRN_MODEL_PATH))
        self.chk_save_recording.Enable(not is_running_or_capturing)
        self.chk_opencc.Enable(not is_running_or_capturing and OPENCC_AVAILABLE)
        self.combo_opencc.Enable(not is_running_or_capturing and self.chk_opencc.IsChecked() and OPENCC_AVAILABLE)
        
        if is_capturing_hotkey:
            self.btn_hotkey.SetLabel("Capturing...")
        else:
            self.btn_hotkey.SetLabel(f"Hotkey: {self.current_hotkey}")

    def _initialize_audio_system(self):
        self.log("Initializing PyMiniaudio...", "INFO")
        try:
            self.recorder = PyMiniaudioRecorder(device_index=-1, frame_length=CHUNK)
            self.recorder.start()
            if not self.recorder.wait_ready(timeout=1.0):
                raise AudioDeviceError("No audio input detected")
            return True
        except Exception as e:
            self.log(f"Failed to initialize audio system: {e}", "ERROR")
            self._cleanup_audio_resources()
            return False

    def _cleanup_audio_resources(self):
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                self.log(f"Error stopping recorder: {e}", "WARNING")
            finally:
                self.recorder = None

    def on_start_listening(self, event):
        if not self.run_initial_checks():
            wx.MessageBox("Initial checks failed. Please see the log for details.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.listening_event.is_set():
            return
        if not self._initialize_audio_system():
            self.update_ui_state()
            return
            
        if self.model is None:
            self.log("Loading Sherpa-ONNX recognizer...", "INFO")
            try:
                self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=MODEL_FILE_PATH,
                    tokens=TOKENS_FILE_PATH,
                    language=self.current_language if self.current_language != "auto" else "",
                    use_itn=True, num_threads=2, provider="cpu",
                )
                self.log("Recognizer loaded successfully.", "SUCCESS")
            except Exception as e:
                self.log(f"Recognizer loading failed: {e}", "ERROR")
                self.model = None
                self._cleanup_audio_resources()
                self.update_ui_state()
                return
                
        if self.chk_gtcrn_enhance.IsChecked() and os.path.exists(GTCRN_MODEL_PATH):
            if self.gtcrn_denoiser is None:
                self.log("Loading GTCRN denoiser model...", "INFO")
                try:
                    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
                        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=GTCRN_MODEL_PATH),
                            provider="cpu", num_threads=2,
                        )
                    )
                    self.gtcrn_denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
                    self.log("GTCRN denoiser loaded successfully.", "SUCCESS")
                except Exception as e:
                    self.log(f"Failed to load GTCRN denoiser: {e}", "ERROR")
                    self.gtcrn_denoiser = None
        else:
            self.gtcrn_denoiser = None

        if self.chk_opencc.IsChecked() and OPENCC_AVAILABLE:
            try:
                selected_config = self.combo_opencc.GetValue().split()[0]
                self.opencc_converter = opencc.OpenCC(selected_config)
            except Exception as e:
                self.log(f"Failed to initialize OpenCC: {e}", "ERROR")
                self.opencc_converter = None

        self.listening_event.set()
        self.log(f"Starting listener thread (Hotkey: '{self.current_hotkey}')...", "INFO")
        self.listener_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listener_thread.start()
        self.update_ui_state()

    def on_stop_listening(self, event):
        if not self.listening_event.is_set():
            return
        self.log("Stopping listener...", "INFO")
        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
        self._cleanup_audio_resources()
        self.update_ui_state()

    def listen_loop(self):
        MAX_RETRIES = 5
        RETRY_DELAY = 2.0
        while self.listening_event.is_set():
            try:
                self.perform_record_and_transcribe()
            except AudioDeviceError as e:
                if not self.listening_event.is_set():
                    break
                self.log(f"Audio device error: {e}. Reconnecting...", "ERROR")
                self._cleanup_audio_resources()
                reconnected = False
                for i in range(MAX_RETRIES):
                    if not self.listening_event.is_set(): break
                    time.sleep(RETRY_DELAY)
                    if self._initialize_audio_system():
                        self.log("Reconnection successful.", "SUCCESS")
                        reconnected = True
                        break
                if not reconnected and self.listening_event.is_set():
                    self.log("Failed to reconnect. Stopping.", "ERROR")
                    wx.CallAfter(self.on_stop_listening, None)
                    break
            except Exception as e:
                if self.listening_event.is_set():
                    self.log(f"Unhandled error in listener loop: {e}", "ERROR")
                    traceback.print_exc()
                break
        self.log("Listener thread finished.", "DEBUG")

    def perform_record_and_transcribe(self):
        blocked_keys = []
        hotkey_parts = [part.strip() for part in self.current_hotkey.split("+")]
        is_caps_lock_hotkey = self.current_hotkey.lower() == "caps lock"
        
        try:
            raw_audio_bytes, duration = self.record_audio(hotkey_parts, blocked_keys)
            
            if raw_audio_bytes and self.listening_event.is_set():
                ok, transcription_result, bytes_to_save = self.transcribe_local(raw_audio_bytes)
                self._unblock_keys(blocked_keys)
                
                if ok and transcription_result:
                    processed_text = self.process_text(transcription_result)
                    
                    if self.chk_opencc.IsChecked() and OPENCC_AVAILABLE and self.opencc_converter:
                        try:
                            processed_text = self.opencc_converter.convert(processed_text)
                        except Exception as e:
                            self.log(f"OpenCC conversion failed: {e}", "ERROR")
                            
                    self.type_text(processed_text)
                    
                    if is_caps_lock_hotkey:
                        try: keyboard.press_and_release("caps lock")
                        except: pass
                        
                if self.chk_save_recording.IsChecked() and bytes_to_save:
                    # 异步非阻塞保存音频，绝不影响下一次录音
                    threading.Thread(
                        target=self.save_audio_worker, 
                        args=(bytes_to_save, transcription_result), 
                        daemon=True
                    ).start()
                    
        except AudioDeviceError:
            raise
        except Exception as e:
            if self.listening_event.is_set():
                self.log(f"Error during cycle: {e}", "ERROR")
        finally:
            self._unblock_keys(blocked_keys)

    def record_audio(self, hotkey_parts, blocked_keys_list):
        frames = []
        recording_started = False
        start_time = 0.0
        hotkey = self.current_hotkey
        is_caps_lock_hotkey = hotkey.lower() == "caps lock"
        
        # 极速环境配置：将常用方法引用提取到本地变量
        listening_event_is_set = self.listening_event.is_set
        monotonic = time.monotonic
        recorder_read = None 

        while listening_event_is_set():
            if not self.recorder or not self.recorder.running:
                raise AudioDeviceError("Recorder not running")
            if recorder_read is None:
                recorder_read = self.recorder.read

            try:
                samples_bytes = recorder_read()
            except AudioDeviceError: raise
            except Exception as e: raise AudioDeviceError(f"Stream error: {e}")

            try:
                pressed = KEYBOARD_AVAILABLE and keyboard.is_pressed(hotkey)
            except:
                pressed = False

            if pressed:
                if not recording_started:
                    self.log(f"Recording started (hotkey '{hotkey}')...", "DEBUG")
                    start_time = monotonic()
                    recording_started = True
                    if not is_caps_lock_hotkey:
                        for key_part in hotkey_parts:
                            try:
                                keyboard.block_key(key_part)
                                blocked_keys_list.append(key_part)
                            except: pass
                frames.append(samples_bytes)

                if (monotonic() - start_time) > MAX_RECORD_SECONDS:
                    self.log(f"Exceeded {MAX_RECORD_SECONDS}s, auto-stopped.", "WARNING")
                    break
            elif recording_started:
                frames.append(samples_bytes)
                try: frames.append(recorder_read()) # 获取尾部额外一层缓冲
                except: pass
                break
            else:
                time.sleep(IDLE_POLL_SLEEP) # 睡眠释放 CPU

        if not recording_started:
            return None, 0.0

        duration = monotonic() - start_time
        if duration < MIN_RECORD_SECONDS and listening_event_is_set():
            return None, duration

        return b"".join(frames), duration

    def transcribe_local(self, raw_audio_bytes):
        """返回: (是否成功, 识别文本, 用于保存的最终 Bytes)"""
        if not self.model or not SHERPA_AVAILABLE:
            return False, None, raw_audio_bytes

        # 原声 bytes 转 float32 数组，乘法替代除法极速化
        audio_np_float32 = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32, copy=False) * _INT16_SCALE
        audio_for_transcription = audio_np_float32
        bytes_to_save = raw_audio_bytes  # 默认保存原声：零 CPU 转换成本！

        use_gtcrn = self.chk_gtcrn_enhance.IsChecked() and self.gtcrn_denoiser is not None
        if use_gtcrn:
            try:
                enhanced_audio = self.gtcrn_denoiser(audio_np_float32, RATE)
                audio_for_transcription = np.asarray(enhanced_audio.samples, dtype=np.float32)
                
                # 只有在使用 GTCRN 且要求保存时，才进行代价高昂的 float32 -> int16 -> bytes 转换
                if self.chk_save_recording.IsChecked():
                    bytes_to_save = (audio_for_transcription * _FLOAT32_SCALE).clip(-32768, 32767).astype(np.int16, copy=False).tobytes()
            except Exception as e:
                self.log(f"GTCRN failed: {e}", "ERROR")

        stream = self.model.create_stream()
        start_time = time.perf_counter()
        try:
            stream.accept_waveform(sample_rate=RATE, waveform=audio_for_transcription)
            stream.accept_waveform(sample_rate=RATE, waveform=_TAIL_PADDING) # 尾部填0，防截断
            self.model.decode_stream(stream)
            transcribed_text = stream.result.text
        except Exception as e:
            self.log(f"Transcription error: {e}", "ERROR")
            return False, None, bytes_to_save
        finally:
            del stream  # 核心：必须显式销毁 C++ 流，彻底防止内存泄漏

        duration = time.perf_counter() - start_time
        self.log(f"识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")
        
        return True, transcribed_text, bytes_to_save

    def save_audio_worker(self, audio_bytes, transcription):
        """完全独立的后台线程运行：不占用主录音线程"""
        if not audio_bytes: return
        try:
            recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "录音")
            os.makedirs(recordings_dir, exist_ok=True)
            
            # 加入微秒 %f 防止极快连按导致重名文件相互覆盖
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] 
            sanitized_text = self.sanitize_filename_part(transcription)
            enh_tag = "_enhanced" if self.chk_gtcrn_enhance.IsChecked() else ""
            
            filename = f"rec_{timestamp}{enh_tag}_{sanitized_text}.wav" if sanitized_text else f"rec_{timestamp}{enh_tag}.wav"
            filepath = os.path.join(recordings_dir, filename)
            
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)
            self.log(f"Recording saved: {os.path.basename(filepath)}", "INFO")
        except Exception as e:
            self.log(f"Error saving audio: {e}", "ERROR")

    def run_initial_checks(self):
        self.log("Running initial checks...", "INFO")
        all_ok = True
        if not SHERPA_AVAILABLE:
            self.log("Sherpa-ONNX library not found.", "ERROR")
            all_ok = False
        if not MINIAUDIO_AVAILABLE:
            self.log("Miniaudio library not found.", "ERROR")
            all_ok = False
        if not KEYBOARD_AVAILABLE:
            self.log("Keyboard library not found or permission error.", "ERROR")
            all_ok = False
        if not os.path.isfile(MODEL_FILE_PATH):
            self.log(f"Model file not found: {MODEL_FILE_PATH}", "ERROR")
            all_ok = False
        if not os.path.isfile(TOKENS_FILE_PATH):
            self.log(f"Tokens file not found: {TOKENS_FILE_PATH}", "ERROR")
            all_ok = False
        if not os.path.exists(GTCRN_MODEL_PATH):
            self.chk_gtcrn_enhance.SetValue(False)
            self.chk_gtcrn_enhance.Disable()
        else:
            self.chk_gtcrn_enhance.Enable()
        if not OPENCC_AVAILABLE:
            self.chk_opencc.SetValue(False)
            self.chk_opencc.Disable()
            self.combo_opencc.Disable()
        else:
            self.chk_opencc.Enable()
            self.combo_opencc.Enable(self.chk_opencc.IsChecked())
        if all_ok: self.log("All critical checks passed. Ready to start.", "SUCCESS")
        return all_ok

    def on_close(self, event):
        self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        self.save_setting(GTCRN_CONFIG_PATH, self.chk_gtcrn_enhance.IsChecked())
        self.save_setting(SAVE_RECORDING_CONFIG_PATH, self.chk_save_recording.IsChecked())
        self.save_setting(OPENCC_ENABLED_PATH, self.chk_opencc.IsChecked())
        self.save_setting(OPENCC_CONFIG_PATH, self.combo_opencc.GetValue())
        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        if KEYBOARD_AVAILABLE:
            try: keyboard.unhook_all()
            except: pass
        self._cleanup_audio_resources()
        self.Destroy()

    def on_set_hotkey(self, event):
        if self.capturing_hotkey_event.is_set(): return
        self.capturing_hotkey_event.set()
        self.update_ui_state()
        self.log("Hotkey capture active. Press desired key/combination...", "INFO")
        threading.Thread(target=self.capture_hotkey_thread_func, daemon=True).start()

    def on_combo_select(self, event):
        self.current_language = self.combo_box.GetValue()

    def on_opencc_toggle(self, event):
        self.combo_opencc.Enable(self.chk_opencc.IsChecked())
        self.update_ui_state()

    def on_opencc_select(self, event):
        self.save_setting(OPENCC_CONFIG_PATH, self.combo_opencc.GetValue())

    def on_clear_log(self, event):
        self.log_text.Clear()
        self._log_line_count = 0

    def on_copy_log(self, event):
        log_content = self.log_text.GetValue()
        if log_content and wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(log_content))
            wx.TheClipboard.Close()

    def type_text(self, text_to_type):
        if not text_to_type: return
        try:
            if sys.platform == "win32":
                keyboard.write(text_to_type)
            elif sys.platform.startswith("linux"):
                oc = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.05)
                keyboard.press_and_release("shift+insert")
                time.sleep(0.05)
                pyperclip.copy(oc)
            elif sys.platform == "darwin":
                oc = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.05)
                keyboard.press_and_release("command+v")
                time.sleep(0.05)
                pyperclip.copy(oc)
            else:
                keyboard.write(text_to_type)
            self.log(f"Typed: '{text_to_type}'", "SUCCESS")
        except Exception as e:
            self.log(f"Keyboard input failed: {e}", "ERROR")

    def _unblock_keys(self, keys):
        if not KEYBOARD_AVAILABLE: return
        seen = set()
        for key_part in keys:
            if key_part in seen: continue
            seen.add(key_part)
            try: keyboard.unblock_key(key_part)
            except: pass

    def capture_hotkey_thread_func(self):
        try:
            new_hotkey = keyboard.read_hotkey(suppress=False)
            if len(new_hotkey) > 30: return
            self.current_hotkey = new_hotkey
            self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
            self.log(f"Captured: '{new_hotkey}'", "SUCCESS")
        except: pass
        finally:
            self.capturing_hotkey_event.clear()
            wx.CallAfter(self.update_ui_state)

    def load_setting(self, file_path, default_value):
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    return content if content else default_value
        except: pass
        return default_value

    def save_setting(self, file_path, value_to_save):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(value_to_save))
        except: pass

    def process_text(self, text):
        if not text or not isinstance(text, str):
            return text
        punct_count = 0
        # O(1) 极速短路算法：只要标点数 > 1 立刻结束遍历并返回
        for char in text:
            if char in _PUNCTUATION_SET:
                punct_count += 1
                if punct_count > 1:
                    return text
        if punct_count <= 1:
            return text.translate(_TRANSLATE_TABLE)
        return text

    def sanitize_filename_part(self, text, max_len=MAX_FILENAME_TEXT_LEN):
        if not text or not isinstance(text, str):
            return ""
        sanitized = _RE_INVALID_CHARS.sub("", text)
        sanitized = _RE_WHITESPACE.sub("_", sanitized).strip("_")
        return sanitized[:max_len].strip("_") if len(sanitized) > max_len else sanitized

    def on_power_suspending(self, event):
        self.was_listening_before_suspend = self.listening_event.is_set()
        if self.was_listening_before_suspend:
            self.listening_event.clear()
            self._cleanup_audio_resources()
        event.Skip()

    def on_power_resume(self, event):
        if self.was_listening_before_suspend:
            wx.CallAfter(self.on_start_listening, None)
        event.Skip()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
