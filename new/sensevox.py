# -*- coding: utf-8 -*-
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
    """获取 assets 目录下文件的完整路径（兼容打包环境）"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_path = os.path.abspath(".")
    return os.path.join(base_path, "assets", relative_path)


# --- 核心常量配置 ---
CHUNK = 1024 * 2
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH_BYTES = 2
MIN_RECORD_SECONDS = 0.3
MAX_RECORD_SECONDS = 30.0          # 最长录音时间
IDLE_POLL_SLEEP = 0.02            # 空闲轮询间隔（防止 CPU 飙高）
DEFAULT_HOTKEY = "space"
MAX_FILENAME_TEXT_LEN = 15

# 标点符号集合（用于去除少量标点）
ALL_PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』“”‘’·～《》〈〉﹏——……〜・〝〟‹›"""

# 预计算常量，提升运行时性能
_PUNCTUATION_SET = frozenset(ALL_PUNCTUATION)
_TRANSLATE_TABLE = str.maketrans("", "", ALL_PUNCTUATION)
_TAIL_PADDING = np.zeros(int(0.5 * RATE), dtype=np.float32)
_RE_INVALID_CHARS = re.compile(r'[\\/*?:"<>|\n\r\t]+')
_RE_WHITESPACE = re.compile(r"\s+")
_INT16_SCALE = np.float32(1.0 / 32768.0)
_FLOAT32_SCALE = np.float32(32767.0)
_MAX_LOG_LINES = 100

# --- 路径与文件 ---
MODEL_DIR = get_asset_path("sensevoicesmallonnx")
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENS_FILE_PATH = os.path.join(MODEL_DIR, "tokens.txt")
GTCRN_MODEL_PATH = get_asset_path("gtcrn_simple.onnx")
ICON_PATH = get_asset_path("app_icon.ico")

QWEN3_ASR_DIR = get_asset_path("qwen3-asr")
QWEN3_ASR_FILES = {
    "conv_frontend": os.path.join(QWEN3_ASR_DIR, "conv_frontend.onnx"),
    "encoder": os.path.join(QWEN3_ASR_DIR, "encoder.int8.onnx"),
    "decoder": os.path.join(QWEN3_ASR_DIR, "decoder.int8.onnx"),
    "tokenizer": os.path.join(QWEN3_ASR_DIR, "tokenizer"),
}

HOTKEY_FILE_PATH = get_asset_path("hotkey.txt")
GTCRN_CONFIG_PATH = get_asset_path("gtcrn_config.txt")
SAVE_RECORDING_CONFIG_PATH = get_asset_path("save_recording_config.txt")
OPENCC_CONFIG_PATH = get_asset_path("opencc_config.txt")
OPENCC_ENABLED_PATH = get_asset_path("opencc_enabled.txt")
ENGINE_CONFIG_PATH = get_asset_path("engine_config.txt")

# --- 语音识别引擎 ---
ENGINE_SENSEVOICE = "sensevoice"
ENGINE_QWEN3_ASR = "qwen3-asr"
ENGINE_OPTIONS = [ENGINE_SENSEVOICE, ENGINE_QWEN3_ASR]

ENGINE_LANGUAGES = {
    ENGINE_SENSEVOICE: ["auto", "zh", "en", "ja", "ko", "yue"],
    ENGINE_QWEN3_ASR: [
        "auto", "Chinese", "English", "Japanese", "Korean", "Cantonese",
        "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian",
        "Italian", "Russian", "Thai", "Vietnamese", "Turkish", "Hindi",
        "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
        "Filipino", "Persian", "Greek", "Hungarian", "Macedonian", "Romanian",
    ],
}

DEFAULT_ENGINE_LANGUAGE = "auto"

OPENCC_OPTIONS = [
    "s2t.json 簡 → 繁", "t2s.json 繁 → 簡", "s2tw.json 簡 → 臺灣繁",
    "tw2s.json 臺灣繁 → 簡", "s2hk.json 簡 → 香港繁", "hk2s.json 香港繁 → 簡",
    "s2twp.json 簡 → 臺灣繁 + 臺灣用語", "tw2sp.json 臺灣繁 → 簡 + 大陸用語",
    "t2tw.json 繁 → 臺灣繁", "hk2t.json 香港繁 → 繁", "t2hk.json 繁 → 香港繁",
    "t2jp.json 繁 → 日文新字體", "jp2t.json 日文新字體 → 繁", "tw2t.json 臺灣繁 → 繁"
]


class AudioDeviceError(Exception):
    """音频设备异常"""


class PyMiniaudioRecorder:
    """基于 miniaudio 的录音类（由 v2 优化，保留 v1 的稳定性）"""

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
        """miniaudio 数据回调生成器（v2 优化版：局部变量减少属性访问）"""
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
        """启动录音设备"""
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
        """停止录音设备"""
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
        """等待设备就绪"""
        end = time.monotonic() + timeout
        with self.lock:
            while self.running and not self.received_data:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self.cond.wait(timeout=min(0.1, remaining))
            return self.received_data

    def read(self):
        """读取一块音频数据（若阻塞超时则抛出 AudioDeviceError）"""
        with self.lock:
            waited = 0.0
            while self.running and len(self.queue) == 0:
                self.cond.wait(timeout=0.1)
                waited += 0.1
                if self.received_data and (time.monotonic() - self.last_data_time) > self.stall_timeout:
                    raise AudioDeviceError("音频设备无响应，可能已断开")
                if not self.received_data and waited >= self.stall_timeout:
                    raise AudioDeviceError("未收到任何音频数据")
            if len(self.queue) == 0:
                # 极速生成全 0 bytes，避免创建 np 数组再转字节的开销
                return bytes(self._chunk_bytes)
            return self.queue.popleft()


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="神色语音sensevox", size=(500, 560))
        self.recorder = None
        self.model = None
        self.gtcrn_denoiser = None
        self.opencc_converter = None
        self.listening_event = threading.Event()
        self.capturing_hotkey_event = threading.Event()
        self.listener_thread = None
        self.capture_thread = None
        self.was_listening_before_suspend = False
        self._log_line_count = 0
        self._save_threads = []          # 后台保存音频的线程列表，用于退出时等待

        # 启动监听时锁定的一组配置状态，供后台线程安全访问
        self._save_recording_enabled = False
        self._enhance_enabled = False
        self._opencc_enabled = False
        self._engine_name = ENGINE_SENSEVOICE
        self._engine_language = DEFAULT_ENGINE_LANGUAGE

        if os.path.exists(ICON_PATH):
            self.SetIcon(wx.Icon(ICON_PATH, wx.BITMAP_TYPE_ICO))

        # 加载配置
        self.current_hotkey = self.load_setting(HOTKEY_FILE_PATH, DEFAULT_HOTKEY)
        initial_gtcrn_state = self.load_setting(GTCRN_CONFIG_PATH, "true").lower() == "true"
        initial_save_state = self.load_setting(SAVE_RECORDING_CONFIG_PATH, "true").lower() == "true"
        initial_opencc_state = self.load_setting(OPENCC_ENABLED_PATH, "false").lower() == "true"
        initial_opencc_config = self.load_setting(OPENCC_CONFIG_PATH, OPENCC_OPTIONS[0])

        saved_engine = self.load_setting(ENGINE_CONFIG_PATH, ENGINE_SENSEVOICE)
        self.current_engine = saved_engine if saved_engine in ENGINE_OPTIONS else ENGINE_SENSEVOICE
        self.engine_languages = {
            engine: DEFAULT_ENGINE_LANGUAGE for engine in ENGINE_OPTIONS
        }
        self.current_language = DEFAULT_ENGINE_LANGUAGE

        # 构建 UI
        self._init_ui(initial_gtcrn_state, initial_save_state,
                      initial_opencc_state, initial_opencc_config)
        self.BindEvents()
        self.SetMinSize((450, 560))
        self.Centre()
        self.run_initial_checks()
        self.update_ui_state()

    def _init_ui(self, initial_gtcrn_state, initial_save_state,
                 initial_opencc_state, initial_opencc_config):
        """创建界面（v2 结构，保留 v1 的全部控件）"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # 第一行按钮：启动 / 暂停 / 设置热键
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

        # 语音识别引擎选择
        engine_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.engine_label = wx.StaticText(panel, label="语音识别引擎 ASR Engine：")
        self.combo_engine = wx.ComboBox(
            panel, value=self.current_engine,
            choices=ENGINE_OPTIONS, style=wx.CB_READONLY
        )
        engine_sizer.Add(self.engine_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        engine_sizer.Add(self.combo_engine, 1, wx.EXPAND)
        main_sizer.Add(engine_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 语言选择
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lang_label = wx.StaticText(panel, label="语言 Language：")
        self.combo_box = wx.ComboBox(
            panel, value=self.current_language,
            choices=ENGINE_LANGUAGES[self.current_engine],
            style=wx.CB_READONLY
        )
        lang_sizer.Add(self.lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        lang_sizer.Add(self.combo_box, 1, wx.EXPAND)
        main_sizer.Add(lang_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # GTCRN 增强复选框
        self.chk_gtcrn_enhance = wx.CheckBox(panel, label="GTCRN 增强 (需要 assets/gtcrn_simple.onnx)")
        self.chk_gtcrn_enhance.SetValue(initial_gtcrn_state)
        main_sizer.Add(self.chk_gtcrn_enhance, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 保存录音复选框
        self.chk_save_recording = wx.CheckBox(panel, label="保存录音 (保存在 '录音' 文件夹)")
        self.chk_save_recording.SetValue(initial_save_state)
        main_sizer.Add(self.chk_save_recording, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # OpenCC 设置行
        opencc_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_opencc = wx.CheckBox(panel, label="OpenCC")
        self.chk_opencc.SetValue(initial_opencc_state)
        self.combo_opencc = wx.ComboBox(panel, value=initial_opencc_config,
                                        choices=OPENCC_OPTIONS, style=wx.CB_READONLY)
        opencc_sizer.Add(self.chk_opencc, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        opencc_sizer.Add(self.combo_opencc, 1, wx.EXPAND)
        main_sizer.Add(opencc_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 日志文本框
        self.log_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.log_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main_sizer.Add(self.log_text, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 日志操作按钮
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
        """绑定事件（v1 全部事件 + v2 的电源管理）"""
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start_listening)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop_listening)
        self.btn_hotkey.Bind(wx.EVT_BUTTON, self.on_set_hotkey)
        self.btn_clear_log.Bind(wx.EVT_BUTTON, self.on_clear_log)
        self.btn_copy_log.Bind(wx.EVT_BUTTON, self.on_copy_log)
        self.combo_box.Bind(wx.EVT_COMBOBOX, self.on_combo_select)
        self.combo_engine.Bind(wx.EVT_COMBOBOX, self.on_engine_select)
        self.chk_opencc.Bind(wx.EVT_CHECKBOX, self.on_opencc_toggle)
        self.combo_opencc.Bind(wx.EVT_COMBOBOX, self.on_opencc_select)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        if hasattr(wx, "EVT_POWER_SUSPENDING"):
            self.Bind(wx.EVT_POWER_SUSPENDING, self.on_power_suspending)
        if hasattr(wx, "EVT_POWER_RESUME"):
            self.Bind(wx.EVT_POWER_RESUME, self.on_power_resume)

    def log(self, message, level="INFO"):
        """线程安全日志（v2 高效计数器 + v1 完整格式）"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp} {level}] {message}\n"

        def append_and_prune():
            self.log_text.AppendText(log_entry)
            self._log_line_count += 1
            if self._log_line_count > _MAX_LOG_LINES:
                extra = self._log_line_count - _MAX_LOG_LINES
                end_pos = self.log_text.XYToPosition(0, extra)
                if end_pos != wx.NOT_FOUND and end_pos > 0:
                    self.log_text.Freeze()   # 防止闪烁
                    self.log_text.Remove(0, end_pos)
                    self.log_text.Thaw()
                self._log_line_count = _MAX_LOG_LINES

        wx.CallAfter(append_and_prune)

    def update_ui_state(self):
        """根据运行状态更新控件可用性"""
        is_listening = self.listening_event.is_set()
        is_capturing_hotkey = self.capturing_hotkey_event.is_set()
        is_running_or_capturing = is_listening or is_capturing_hotkey

        self.btn_start.Enable(not is_running_or_capturing)
        self.btn_stop.Enable(is_listening and not is_capturing_hotkey)
        self.btn_hotkey.Enable(not is_running_or_capturing)
        self.combo_box.Enable(not is_running_or_capturing)
        self.combo_engine.Enable(not is_running_or_capturing)
        self.chk_gtcrn_enhance.Enable(not is_running_or_capturing and os.path.exists(GTCRN_MODEL_PATH))
        self.chk_save_recording.Enable(not is_running_or_capturing)
        self.chk_opencc.Enable(not is_running_or_capturing and OPENCC_AVAILABLE)
        self.combo_opencc.Enable(not is_running_or_capturing and self.chk_opencc.IsChecked() and OPENCC_AVAILABLE)

        if is_capturing_hotkey:
            self.btn_hotkey.SetLabel("Capturing...")
        else:
            self.btn_hotkey.SetLabel(f"Hotkey: {self.current_hotkey}")

    # ------------------------------------------------------------
    # 音频与识别核心逻辑
    # ------------------------------------------------------------
    def _initialize_audio_system(self):
        """初始化录音设备（v1 详细日志 + v2 简洁启动）"""
        self.log("Initializing PyMiniaudio recording device... 初始化 PyMiniaudio 录音设备...", "INFO")
        try:
            self.recorder = PyMiniaudioRecorder(device_index=-1, frame_length=CHUNK)
            self.recorder.start()
            if not self.recorder.wait_ready(timeout=1.0):
                raise AudioDeviceError("未检测到音频输入")
            return True
        except Exception as e:
            self.log(f"Failed to initialize audio device 录音设备初始化失败: {e}", "ERROR")
            self._cleanup_audio_resources()
            return False

    def _cleanup_audio_resources(self):
        """释放录音资源（v1 完整日志）"""
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                self.log(f"Error stopping audio device 停止录音设备时出错: {e}", "WARNING")
            finally:
                self.recorder = None
        self.log("Audio resources released. 音频资源已释放。", "DEBUG")

    def on_start_listening(self, event):
        """启动监听（融合 v1 的健壮检查与 v2 的模型复用）"""
        if not self.run_initial_checks():
            wx.MessageBox("初始检查未通过，请查看日志。\nInitial checks failed, please see the log.", "错误 Error", wx.OK | wx.ICON_ERROR)
            return
        if self.listening_event.is_set():
            self.log("Listening is already running. 监听已在运行中。", "WARNING")
            return

        if not self._initialize_audio_system():
            self.update_ui_state()
            return

        # 锁定 UI 状态供后台线程使用
        self._save_recording_enabled = self.chk_save_recording.IsChecked()
        self._enhance_enabled = self.chk_gtcrn_enhance.IsChecked()
        self._opencc_enabled = self.chk_opencc.IsChecked()
        self._engine_name = self.current_engine
        self._engine_language = self.current_language

        # 加载 / 复用语音识别模型
        if self.model is None:
            engine_display = "Qwen3-ASR" if self._engine_name == ENGINE_QWEN3_ASR else "SenseVoice"
            self.log(f"Loading Sherpa-ONNX speech recognition model ({engine_display})... 正在加载 {engine_display} 语音识别模型...", "INFO")
            try:
                self.model = self._create_recognizer()
                self.log("Speech recognition model loaded successfully. 语音识别模型加载成功。", "SUCCESS")
            except Exception as e:
                self.log(f"Failed to load model 模型加载失败: {e}", "ERROR")
                self.model = None
                self._cleanup_audio_resources()
                self.update_ui_state()
                return
        else:
            self.log("Reusing already loaded speech recognition model. 复用已加载的语音识别模型。", "DEBUG")

        # 加载 GTCRN 降噪模型（若启用且已安装模型）
        if self._enhance_enabled and os.path.exists(GTCRN_MODEL_PATH):
            if self.gtcrn_denoiser is None:
                self.log("Loading GTCRN noise reduction model... 正在加载 GTCRN 降噪模型...", "INFO")
                try:
                    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
                        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                                model=GTCRN_MODEL_PATH
                            ),
                            provider="cpu",
                            num_threads=2,
                        )
                    )
                    self.gtcrn_denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
                    self.log("GTCRN noise reduction model loaded successfully. GTCRN 降噪模型加载成功。", "SUCCESS")
                except Exception as e:
                    self.log(f"Failed to load GTCRN model GTCRN 模型加载失败: {e}", "ERROR")
                    self.gtcrn_denoiser = None
        else:
            self.gtcrn_denoiser = None

        # 初始化 OpenCC 转换器
        if self._opencc_enabled and OPENCC_AVAILABLE:
            try:
                selected_config = self.combo_opencc.GetValue().split()[0]
                self.opencc_converter = opencc.OpenCC(selected_config)
                self.log(f"OpenCC converter initialized (config: {selected_config}). OpenCC 转换器已初始化 (配置: {selected_config})", "INFO")
            except Exception as e:
                self.log(f"Failed to initialize OpenCC OpenCC 初始化失败: {e}", "ERROR")
                self.opencc_converter = None
        else:
            self.opencc_converter = None

        self.listening_event.set()
        self.log(f"Starting listening thread (Hotkey: '{self.current_hotkey}')... 启动监听线程（热键: '{self.current_hotkey}'）...", "INFO")
        self.listener_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listener_thread.start()
        self.update_ui_state()

    def on_stop_listening(self, event):
        """停止监听（v1 释放 GTCRN/OpenCC 以节省内存）"""
        if not self.listening_event.is_set():
            return
        self.log("Stopping listening... 正在停止监听...", "INFO")
        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
        self._cleanup_audio_resources()

        # 释放辅助模型，主模型保留以便快速重启
        self.gtcrn_denoiser = None
        self.opencc_converter = None

        self.log("Listening stopped, resources released. 监听已停止，资源已释放。", "INFO")
        self.update_ui_state()

    def _create_recognizer(self):
        """根据当前引擎创建对应的 Sherpa-ONNX 离线识别器"""
        if self._engine_name == ENGINE_QWEN3_ASR:
            return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=QWEN3_ASR_FILES["conv_frontend"],
                encoder=QWEN3_ASR_FILES["encoder"],
                decoder=QWEN3_ASR_FILES["decoder"],
                tokenizer=QWEN3_ASR_FILES["tokenizer"],
                max_new_tokens=256,
                num_threads=2,
                provider="cpu",
            )
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=MODEL_FILE_PATH,
            tokens=TOKENS_FILE_PATH,
            language=self._engine_language if self._engine_language != "auto" else "",
            use_itn=True,
            num_threads=2,
            provider="cpu",
        )

    def _missing_engine_files(self, engine):
        """返回指定引擎缺失的模型文件列表；全部存在时返回空列表"""
        if engine == ENGINE_SENSEVOICE:
            required = [MODEL_FILE_PATH, TOKENS_FILE_PATH]
        elif engine == ENGINE_QWEN3_ASR:
            required = [QWEN3_ASR_DIR] + [
                os.path.join(QWEN3_ASR_DIR, "conv_frontend.onnx"),
                os.path.join(QWEN3_ASR_DIR, "encoder.int8.onnx"),
                os.path.join(QWEN3_ASR_DIR, "decoder.int8.onnx"),
                os.path.join(QWEN3_ASR_DIR, "tokenizer"),
            ]
        else:
            required = []
        return [f for f in required if not os.path.exists(f)]

    def listen_loop(self):
        """主监听循环（融合 v1 的详细错误处理与 v2 的简洁逻辑）"""
        MAX_RETRIES = 5
        RETRY_DELAY = 2.0

        self.log("Listening thread started. 监听线程已启动。", "DEBUG")
        while self.listening_event.is_set():
            try:
                self.perform_record_and_transcribe()
            except AudioDeviceError as e:
                if not self.listening_event.is_set():
                    break
                self.log(f"Audio device error: {e}, starting auto-reconnect... 音频设备错误: {e}，开始自动重连...", "ERROR")
                self._cleanup_audio_resources()

                reconnected = False
                for i in range(MAX_RETRIES):
                    if not self.listening_event.is_set():
                        break
                    self.log(f"Reconnection attempt {i+1}/{MAX_RETRIES}, waiting {RETRY_DELAY}s... 第 {i+1}/{MAX_RETRIES} 次重连尝试，等待 {RETRY_DELAY} 秒...")
                    time.sleep(RETRY_DELAY)
                    if self._initialize_audio_system():
                        self.log("Reconnection successful. 重连成功。", "SUCCESS")
                        reconnected = True
                        break

                if not reconnected and self.listening_event.is_set():
                    self.log(f"Failed after {MAX_RETRIES} reconnection attempts, auto-stopping listening. 重连 {MAX_RETRIES} 次后仍失败，自动停止监听。", "ERROR")
                    wx.CallAfter(self.on_stop_listening, None)
                    break

            except Exception as e:
                if self.listening_event.is_set():
                    self.log(f"Unhandled exception in listening loop 监听循环发生未处理异常: {e}", "ERROR")
                    traceback.print_exc()
                break

        self.log("Listening thread finished. 监听线程已结束。", "DEBUG")

    def perform_record_and_transcribe(self):
        """执行一次完整的“录音→识别→输出→可选保存”流程"""
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

                    # OpenCC 繁简转换
                    if self._opencc_enabled and OPENCC_AVAILABLE and self.opencc_converter:
                        try:
                            processed_text = self.opencc_converter.convert(processed_text)
                            self.log("OpenCC conversion completed. OpenCC 转换完成。", "DEBUG")
                        except Exception as e:
                            self.log(f"OpenCC conversion failed OpenCC 转换失败: {e}", "ERROR")

                    # 输出到光标位置
                    self.type_text(processed_text)

                    # 处理 Caps Lock 热键（模拟开关）
                    if is_caps_lock_hotkey:
                        try:
                            keyboard.press_and_release("caps lock")
                        except Exception as e:
                            self.log(f"Failed to simulate Caps Lock 模拟 Caps Lock 失败: {e}", "WARNING")

                # 异步保存录音（不阻塞下一次录音）
                if self._save_recording_enabled and bytes_to_save:
                    self._spawn_save_worker(bytes_to_save, transcription_result)

        except AudioDeviceError:
            raise
        except Exception as e:
            if self.listening_event.is_set():
                self.log(f"Error in record/transcribe cycle 录音/识别周期出错: {e}", "ERROR")
                traceback.print_exc()
        finally:
            self._unblock_keys(blocked_keys)

    def record_audio(self, hotkey_parts, blocked_keys_list):
        """录音（核心高性能版本，来自 v2）"""
        frames = []
        recording_started = False
        start_time = 0.0
        hotkey = self.current_hotkey
        is_caps_lock_hotkey = hotkey.lower() == "caps lock"

        listening_event_is_set = self.listening_event.is_set   # 局部引用加速
        monotonic = time.monotonic
        recorder_read = None

        while listening_event_is_set():
            if not self.recorder or not self.recorder.running:
                raise AudioDeviceError("录音设备已停止")

            if recorder_read is None:
                recorder_read = self.recorder.read

            try:
                samples_bytes = recorder_read()
            except AudioDeviceError:
                raise
            except Exception as e:
                raise AudioDeviceError(f"读取音频流异常: {e}")

            # 检测热键状态
            try:
                pressed = KEYBOARD_AVAILABLE and keyboard.is_pressed(hotkey)
            except Exception:
                pressed = False

            if pressed:
                if not recording_started:
                    self.log(f"Detected hotkey '{hotkey}', starting recording... 检测到热键 '{hotkey}'，开始录音...", "DEBUG")
                    start_time = monotonic()
                    recording_started = True

                    # 屏蔽热键，防止被输入的文本触发
                    if not is_caps_lock_hotkey:
                        for key_part in hotkey_parts:
                            try:
                                keyboard.block_key(key_part)
                                blocked_keys_list.append(key_part)
                            except Exception as e:
                                self.log(f"Failed to block key '{key_part}': {e} 无法屏蔽按键 '{key_part}': {e}", "WARNING")

                frames.append(samples_bytes)

                if (monotonic() - start_time) > MAX_RECORD_SECONDS:
                    self.log(f"Recording exceeded {MAX_RECORD_SECONDS:.0f}s, auto-stopping. 录音超过 {MAX_RECORD_SECONDS:.0f} 秒，自动停止。", "WARNING")
                    break

            elif recording_started:
                # 松开热键后，补录一点尾部语音，避免截断
                frames.append(samples_bytes)
                try:
                    frames.append(recorder_read())
                except Exception:
                    pass
                break
            else:
                time.sleep(IDLE_POLL_SLEEP)

        if not recording_started:
            return None, 0.0

        duration = monotonic() - start_time
        if duration < MIN_RECORD_SECONDS and listening_event_is_set():
            self.log(f"Recording too short ({duration:.2f}s), ignored. 录音太短 ({duration:.2f}s)，忽略。", "WARNING")
            return None, duration

        return b"".join(frames), duration

    def transcribe_local(self, raw_audio_bytes):
        """本地语音识别（融合 v1 的稳健返回和 v2 的高性能转换）"""
        # 原始 int16 bytes 直接用于返回（如果不需要增强，则保存原始音频）
        if not self.model or not SHERPA_AVAILABLE:
            return False, None, raw_audio_bytes

        # int16 bytes -> float32 numpy，使用乘法极速转换
        audio_np_float32 = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) * _INT16_SCALE
        audio_for_transcription = audio_np_float32
        bytes_to_save = raw_audio_bytes   # 默认保存原声，零转换成本

        use_gtcrn = self._enhance_enabled and self.gtcrn_denoiser is not None
        if use_gtcrn:
            try:
                self.log("Applying GTCRN enhancement... 应用 GTCRN 增强...", "DEBUG")
                enhanced_audio = self.gtcrn_denoiser(audio_np_float32, RATE)
                audio_for_transcription = np.asarray(enhanced_audio.samples, dtype=np.float32)

                # 只有需要保存录音时，才将 float32 转换回 int16 bytes
                if self._save_recording_enabled:
                    bytes_to_save = (audio_for_transcription * _FLOAT32_SCALE).clip(
                        -32768, 32767
                    ).astype(np.int16).tobytes()
            except Exception as e:
                self.log(f"GTCRN enhancement failed, using original audio GTCRN 增强失败，使用原始音频: {e}", "ERROR")

        # 使用离线流进行识别
        stream = self.model.create_stream()
        if self._engine_name == ENGINE_QWEN3_ASR:
            qwen3_language = self._engine_language if self._engine_language != "auto" else ""
            if qwen3_language:
                stream.set_option("language", qwen3_language)
        start_time = time.perf_counter()
        try:
            stream.accept_waveform(sample_rate=RATE, waveform=audio_for_transcription)
            stream.accept_waveform(sample_rate=RATE, waveform=_TAIL_PADDING)  # 尾部静音
            self.model.decode_stream(stream)
            transcribed_text = stream.result.text
        except Exception as e:
            self.log(f"Speech recognition exception 语音识别异常: {e}", "ERROR")
            return False, None, bytes_to_save
        finally:
            del stream   # 显式释放 C++ 流，防止内存泄漏

        duration = time.perf_counter() - start_time
        self.log(f"Transcription completed 识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")

        return True, transcribed_text, bytes_to_save

    def _spawn_save_worker(self, audio_bytes, transcription):
        """启动后台保存线程（v2 异步优点 + v1 线程跟踪安全退出）"""
        # 在子线程创建工作前先计算增强标签，避免访问 wx 控件
        use_enhance = self._enhance_enabled
        worker = threading.Thread(
            target=self._save_audio_worker,
            args=(audio_bytes, transcription, use_enhance),
            daemon=True
        )
        self._save_threads.append(worker)
        worker.start()

    def _save_audio_worker(self, audio_bytes, transcription, use_enhance):
        """后台保存音频（完全独立，不阻塞录音）"""
        if not audio_bytes:
            return
        try:
            recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "录音")
            os.makedirs(recordings_dir, exist_ok=True)

            # 带微秒的时间戳，避免快速连按时文件重名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            sanitized_text = self.sanitize_filename_part(transcription)
            enh_tag = "_enhanced" if use_enhance else ""
            filename = f"rec_{timestamp}{enh_tag}_{sanitized_text}.wav" if sanitized_text else f"rec_{timestamp}{enh_tag}.wav"
            filepath = os.path.join(recordings_dir, filename)

            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)
            self.log(f"Recording saved 录音已保存: {os.path.basename(filepath)}", "INFO")
        except Exception as e:
            self.log(f"Failed to save recording 保存录音失败: {e}", "ERROR")
            traceback.print_exc()

    # ------------------------------------------------------------
    # 初始检查与配置
    # ------------------------------------------------------------
    def run_initial_checks(self):
        """初始依赖检查（v1 完整日志）"""
        self.log("Running initial checks... 正在执行初始检查...", "INFO")
        all_ok = True

        if not SHERPA_AVAILABLE:
            self.log("sherpa-onnx library not installed. 未安装 sherpa-onnx 库。", "ERROR")
            all_ok = False
        if not MINIAUDIO_AVAILABLE:
            self.log("miniaudio library not installed. 未安装 miniaudio 库。", "ERROR")
            all_ok = False
        if not KEYBOARD_AVAILABLE:
            self.log("keyboard library not installed or permission denied. 未安装 keyboard 库或无权限。", "ERROR")
            all_ok = False
        # 当前 ASR 引擎的模型文件检查
        missing_engine_files = self._missing_engine_files(self.current_engine)
        if missing_engine_files:
            for f in missing_engine_files:
                self.log(f"Model file not found 模型文件不存在: {f}", "ERROR")
            all_ok = False
        else:
            self.log("ASR engine model files complete. ASR 引擎模型文件完整。", "INFO")

        # GTCRN 可选
        if not os.path.exists(GTCRN_MODEL_PATH):
            self.log("GTCRN model not found, enhancement disabled. GTCRN 模型未找到，增强功能已禁用。", "WARNING")
            self.chk_gtcrn_enhance.SetValue(False)
            self.chk_gtcrn_enhance.Disable()
        else:
            self.log("GTCRN model found, enhancement available. GTCRN 模型已找到，增强功能可用。", "INFO")
            self.chk_gtcrn_enhance.Enable()

        # OpenCC 可选
        if not OPENCC_AVAILABLE:
            self.log("opencc library not installed, conversion disabled. 未安装 opencc 库，简繁转换功能已禁用。", "WARNING")
            self.chk_opencc.SetValue(False)
            self.chk_opencc.Disable()
            self.combo_opencc.Disable()
        else:
            self.log("opencc library found, conversion available. OpenCC 库已找到，简繁转换功能可用。", "INFO")
            self.chk_opencc.Enable()
            self.combo_opencc.Enable(self.chk_opencc.IsChecked())

        if all_ok:
            self.log("All critical dependencies checked successfully, ready to use. 所有关键依赖检查通过，可以开始使用。", "SUCCESS")
        else:
            self.log("Critical dependencies missing, please fix and retry. 存在关键依赖缺失，请修复后重试。", "ERROR")

        return all_ok

    # ------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------
    def on_close(self, event):
        """窗口关闭（v2 的异步线程等待 + v1 的完整清理）"""
        self.log("Closing window, saving settings and cleaning up resources... 正在关闭窗口，保存设置并清理资源...", "INFO")

        # 保存所有配置
        self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        self.save_setting(GTCRN_CONFIG_PATH, self.chk_gtcrn_enhance.IsChecked())
        self.save_setting(SAVE_RECORDING_CONFIG_PATH, self.chk_save_recording.IsChecked())
        self.save_setting(OPENCC_ENABLED_PATH, self.chk_opencc.IsChecked())
        self.save_setting(OPENCC_CONFIG_PATH, self.combo_opencc.GetValue())
        self.save_setting(ENGINE_CONFIG_PATH, self.current_engine)

        # 停止监听
        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)

        # 等待后台保存线程完成（最多等 2 秒）
        for worker in self._save_threads:
            if worker.is_alive():
                worker.join(timeout=2.0)

        # 释放键盘钩子
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except Exception:
                pass

        self._cleanup_audio_resources()
        self.Destroy()

    def on_set_hotkey(self, event):
        """开始捕获热键（保留 v1 的独立线程属性，便于追踪）"""
        if self.capturing_hotkey_event.is_set():
            return
        self.capturing_hotkey_event.set()
        self.update_ui_state()
        self.log("Hotkey capture active, please press new hotkey... 热键捕获已开启，请按下新的热键...", "INFO")
        self.capture_thread = threading.Thread(target=self.capture_hotkey_thread_func, daemon=True)
        self.capture_thread.start()

    def capture_hotkey_thread_func(self):
        """热键捕获线程（v1 完整日志）"""
        try:
            new_hotkey = keyboard.read_hotkey(suppress=False)
            if len(new_hotkey) > 30:
                self.log("Captured hotkey too long, ignored. 捕获的热键过长，已忽略。", "WARNING")
                return
            self.current_hotkey = new_hotkey
            self.log(f"Captured new hotkey 已捕获新热键: '{new_hotkey}'", "SUCCESS")
            self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        except Exception as e:
            self.log(f"Hotkey capture failed 热键捕获失败: {e}", "ERROR")
        finally:
            self.capturing_hotkey_event.clear()
            wx.CallAfter(self.update_ui_state)

    def on_combo_select(self, event):
        self.engine_languages[self.current_engine] = self.combo_box.GetValue()
        self.current_language = self.combo_box.GetValue()
        self.log(f"Language preference set to 语言偏好已设为: '{self.current_language}'", "INFO")

    def on_engine_select(self, event):
        """切换语音识别引擎（切换时清空已加载模型，下次启动时重新加载）"""
        new_engine = self.combo_engine.GetValue()
        if new_engine == self.current_engine:
            return
        missing = self._missing_engine_files(new_engine)
        if missing:
            self.log(f"Engine '{new_engine}' model files missing: {', '.join(missing)} 引擎 '{new_engine}' 模型文件缺失: {', '.join(missing)}", "ERROR")
            wx.MessageBox(
                "语音识别引擎 '" + new_engine + "' 模型文件缺失：\n\n"
                + "\n".join(missing)
                + "\n\n请将模型放入 assets/" + new_engine + " 目录。",
                "模型缺失", wx.OK | wx.ICON_ERROR,
            )
            self.combo_engine.SetValue(self.current_engine)
            return
        self.current_engine = new_engine
        self.engine_languages[new_engine] = self.combo_box.GetValue()
        self.current_language = self.engine_languages[new_engine]
        self.combo_box.SetItems(ENGINE_LANGUAGES[new_engine])
        self.combo_box.SetValue(self.current_language)
        self.save_setting(ENGINE_CONFIG_PATH, self.current_engine)
        self.model = None
        engine_display = "Qwen3-ASR" if new_engine == ENGINE_QWEN3_ASR else "SenseVoice"
        self.log(f"ASR engine switched to ASR 引擎已切换为: '{engine_display}', language: {self.current_language}", "INFO")

    def on_opencc_toggle(self, event):
        self.combo_opencc.Enable(self.chk_opencc.IsChecked())
        self.update_ui_state()
        self.log(f"OpenCC {'enabled 已启用' if self.chk_opencc.IsChecked() else 'disabled 已禁用'}.", "INFO")

    def on_opencc_select(self, event):
        self.save_setting(OPENCC_CONFIG_PATH, self.combo_opencc.GetValue())
        self.log(f"OpenCC config set to OpenCC 配置已设为: '{self.combo_opencc.GetValue()}'", "INFO")

    def on_clear_log(self, event):
        self.log_text.Clear()
        self._log_line_count = 0

    def on_copy_log(self, event):
        log_content = self.log_text.GetValue()
        if log_content and wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(log_content))
            wx.TheClipboard.Close()
            self.log("Log copied to clipboard. 日志已复制到剪贴板。", "SUCCESS")

    # ------------------------------------------------------------
    # 文本输出与处理
    # ------------------------------------------------------------
    def type_text(self, text_to_type):
        """模拟键盘输入（v1 完整日志 + 各平台适配）"""
        if not text_to_type:
            return
        try:
            if sys.platform == "win32":
                keyboard.write(text_to_type)
                self.log(f"Keyboard typed 键盘输出: '{text_to_type}'", "SUCCESS")

            elif sys.platform.startswith("linux"):
                original_clipboard = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.05)
                keyboard.press_and_release("shift+insert")
                time.sleep(0.05)
                pyperclip.copy(original_clipboard)
                self.log(f"Linux keyboard typed Linux 键盘输出: '{text_to_type}'", "SUCCESS")

            elif sys.platform == "darwin":
                original_clipboard = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.05)
                keyboard.press_and_release("command+v")
                time.sleep(0.05)
                pyperclip.copy(original_clipboard)
                self.log(f"macOS keyboard typed macOS 键盘输出: '{text_to_type}'", "SUCCESS")

            else:
                keyboard.write(text_to_type)
                self.log(f"Unknown OS keyboard typed 未知系统键盘输出: '{text_to_type}'", "SUCCESS")

        except Exception as e:
            self.log(f"Keyboard input failed 键盘输入失败: {e}", "ERROR")

    def process_text(self, text):
        """文本清理：若标点数量 ≤1 则去除所有标点（v2 高效短路版）"""
        if not text or not isinstance(text, str):
            return text

        punct_count = 0
        for char in text:
            if char in _PUNCTUATION_SET:
                punct_count += 1
                if punct_count > 1:
                    return text   # 大于 1 个标点则保留原文本

        # 只有 0 或 1 个标点时才移除
        return text.translate(_TRANSLATE_TABLE)

    def sanitize_filename_part(self, text, max_len=MAX_FILENAME_TEXT_LEN):
        """清理文件名中的非法字符（v2 预编译正则）"""
        if not text or not isinstance(text, str):
            return ""
        sanitized = _RE_INVALID_CHARS.sub("", text)
        sanitized = _RE_WHITESPACE.sub("_", sanitized).strip("_")
        return sanitized[:max_len].strip("_") if len(sanitized) > max_len else sanitized

    # ------------------------------------------------------------
    # 休眠 / 唤醒处理
    # ------------------------------------------------------------
    def on_power_suspending(self, event):
        self.log("System suspend detected, releasing audio resources... 检测到系统休眠，释放音频资源...", "WARNING")
        self.was_listening_before_suspend = self.listening_event.is_set()
        if self.was_listening_before_suspend:
            self.listening_event.clear()
            self._cleanup_audio_resources()
        event.Skip()

    def on_power_resume(self, event):
        self.log("System resumed. 系统已唤醒。", "INFO")
        if self.was_listening_before_suspend:
            self.log("Resuming listening... 正在恢复监听...", "INFO")
            wx.CallAfter(self.on_start_listening, None)
        event.Skip()

    # ------------------------------------------------------------
    # 配置文件读写
    # ------------------------------------------------------------
    def load_setting(self, file_path, default_value):
        """读取配置文件（v1 完整异常日志）"""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    return content if content else default_value
        except Exception as e:
            self.log(f"Failed to load config file 读取配置文件 {os.path.basename(file_path)} 失败: {e}", "WARNING")
        return default_value

    def save_setting(self, file_path, value_to_save):
        """写入配置文件（v1 完整异常日志）"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(value_to_save))
        except Exception as e:
            self.log(f"Failed to save config file 保存配置文件 {os.path.basename(file_path)} 失败: {e}", "ERROR")

    # ------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------
    def _unblock_keys(self, keys):
        """解除按键屏蔽（v1 去重逻辑）"""
        if not KEYBOARD_AVAILABLE:
            return
        seen = set()
        for key_part in keys:
            if key_part in seen:
                continue
            seen.add(key_part)
            try:
                keyboard.unblock_key(key_part)
            except Exception:
                pass


if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
