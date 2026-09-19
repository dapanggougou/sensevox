# -*- coding: utf-8 -*-
import ctypes
import datetime
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from ctypes import wintypes

import wx

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
def _enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwarenessContext(-4)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_enable_dpi_awareness()


# ------------------------------------------------------------------
# 路径工具：不写死任何绝对路径，全部相对脚本 / 配置文件
# ------------------------------------------------------------------
def get_app_dir():
    """程序所在目录（兼容 PyInstaller 打包）。作为默认保底模型目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_asset_path(relative_path):
    """assets 目录下文件的完整路径（与 v0.6 一致，兼容打包环境）"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = get_app_dir()
    return os.path.join(base_path, "assets", relative_path)


def load_app_icon():
    """加载窗口图标，优先使用 assets/app_icon.ico，否则回退到 exe 自身内嵌图标。

    PyInstaller 的 -i/--icon 只写入 exe 的图标资源，不会附带 .ico 文件。
    这里用 wx.IconLocation 指定 exe 内的图标索引：wxWidgets 在 Windows 下
    对 ICO 的处理会调用 Win32 ExtractIconEx/ExtractIcon，而它们同样能读取
    .exe 的图标资源，因此无需再单独分发 app_icon.ico。
    """
    if os.path.exists(ICON_PATH):
        try:
            icon = wx.Icon(ICON_PATH, wx.BITMAP_TYPE_ICO)
            if icon.IsOk():
                return icon
        except Exception:
            pass

    if getattr(sys, "frozen", False):
        try:
            executable = sys.executable
            # 先尝试大图标(索引 0)，失败再取小图标(索引 1)
            for index in (0, 1):
                icon = wx.Icon(wx.IconLocation(executable, index))
                if icon.IsOk():
                    return icon
        except Exception:
            pass

    return None


# --- 核心常量配置 ---
CHUNK = 1024 * 2
CHANNELS = 1
RATE = 16000
MIN_RECORD_SECONDS = 0.3
MAX_RECORD_SECONDS = 30.0          # 最长录音时间
IDLE_POLL_SLEEP = 0.02            # 空闲轮询间隔（防止 CPU 飙高）
DEFAULT_HOTKEY = "caps lock"

# 标点符号集合（用于去除少量标点）
ALL_PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』“”‘’·～《》〈〉﹏——……〜・〝〟‹›"""

# SenseVoice 富文本模型会输出 "词。" + \ue000 + "N" 形式（N 为停顿等级）。
# 该控制字符不能直接注入键盘，需在输出前剥离。
_PAUSE_MARKER = "\ue000"
_RE_PAUSE_MARKER = re.compile("\ue000[0-9]*")

# 预计算常量，提升运行时性能
_PUNCTUATION_SET = frozenset(ALL_PUNCTUATION)
_TRANSLATE_TABLE = str.maketrans("", "", ALL_PUNCTUATION)
_TAIL_PADDING = np.zeros(int(0.5 * RATE), dtype=np.float32)
_INT16_SCALE = np.float32(1.0 / 32768.0)
_MAX_LOG_LINES = 100
_MAX_MONITOR_CHUNKS = 16          # 监听播放队列上限，超出则丢最旧块以限制延迟累积

# --- 路径与文件（模型目录运行时决定，其余配置与 v0.6 一致放 assets） ---
MODEL_DIR_CONFIG_PATH = get_asset_path("model_dir.txt")
HOTKEY_FILE_PATH = get_asset_path("hotkey.txt")
MONITOR_CONFIG_PATH = get_asset_path("monitor_config.txt")
OPENCC_CONFIG_PATH = get_asset_path("opencc_config.txt")
OPENCC_ENABLED_PATH = get_asset_path("opencc_enabled.txt")
ICON_PATH = get_asset_path("app_icon.ico")

# 模型目录中必须存在的文件
MODEL_ONNX_NAME = "model.onnx"
TOKENS_NAME = "tokens.txt"

DEFAULT_LANGUAGE = "auto"
LANGUAGES = ["auto", "zh", "en", "ja", "ko", "yue"]

OPENCC_OPTIONS = [
    "s2t.json 簡 → 繁", "t2s.json 繁 → 簡", "s2tw.json 簡 → 臺灣繁",
    "tw2s.json 臺灣繁 → 簡", "s2hk.json 簡 → 香港繁", "hk2s.json 香港繁 → 簡",
    "s2twp.json 簡 → 臺灣繁 + 臺灣用語", "tw2sp.json 臺灣繁 → 簡 + 大陸用語",
    "t2tw.json 繁 → 臺灣繁", "hk2t.json 香港繁 → 繁", "t2hk.json 繁 → 香港繁",
    "t2jp.json 繁 → 日文新字體", "jp2t.json 日文新字體 → 繁", "tw2t.json 臺灣繁 → 繁"
]


# ==================================================================
# SendInput 文本注入（Windows 原生，性能最佳）
# ==================================================================
user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    # 必须包含全部三个成员：SendInput 会校验 cbSize，而 union 的大小由最大的
    # 成员 MOUSEINPUT 决定（x64 下使 INPUT 为 56 字节）。只声明 ki 会得到 40，
    # 调用即返回 ERROR_INVALID_PARAMETER (87)。
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

_INPUT_SIZE = ctypes.sizeof(INPUT)


def _send_inputs(events):
    """一次性把事件数组送入系统输入流（原子操作，不会被用户输入插入）"""
    count = len(events)
    if count == 0:
        return
    arr = (INPUT * count)(*events)
    sent = user32.SendInput(count, arr, _INPUT_SIZE)
    if sent != count:
        raise ctypes.WinError(ctypes.get_last_error())


def _unicode_events(text):
    """把字符串编成 KEYEVENTF_UNICODE 按下/抬起事件（UTF-16，含 surrogate pair）"""
    units = text.encode("utf-16-le")
    events = []
    for i in range(0, len(units), 2):
        code = units[i] | (units[i + 1] << 8)
        events.append(INPUT(INPUT_KEYBOARD, _INPUTUNION(
            ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, None))))
        events.append(INPUT(INPUT_KEYBOARD, _INPUTUNION(
            ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None))))
    return events


def send_unicode_text(text):
    """把整个字符串一次性送进系统输入流（含 Unicode / emoji）"""
    _send_inputs(_unicode_events(text))


# ==================================================================
# 音频设备
# ==================================================================
class AudioDeviceError(Exception):
    """音频设备异常"""


class PyMiniaudioRecorder:
    """基于 miniaudio 的录音类；可选把每个数据块同步给监听播放器"""

    def __init__(self, frame_length=CHUNK, stall_timeout=1.0, max_queue_chunks=128):
        self.frame_length = frame_length
        self.sample_rate = RATE
        self.channels = CHANNELS
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
        self._chunk_bytes = self.frame_length * self.channels * 2
        self._monitor_sink = None      # 监听播放器 push 回调（在音频线程中调用）

    def set_monitor_sink(self, sink):
        """设置监听播放回调（None 表示关闭）。回调必须极快且不抛异常。"""
        self._monitor_sink = sink

    def _capture_generator(self):
        """miniaudio 数据回调生成器（局部变量减少属性访问）"""
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
            chunk = bytes(data)
            # 同步送给监听播放器（送入识别前的原始音频）
            sink = self._monitor_sink
            if sink is not None:
                try:
                    sink(chunk)
                except Exception:
                    pass
            with cond:
                buf.extend(chunk)
                while len(buf) >= chunk_bytes:
                    piece = bytes(buf[:chunk_bytes])
                    del buf[:chunk_bytes]
                    q.append(piece)
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
        self._monitor_sink = None

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
                # 补静音，避免下游出现空块
                return bytes(self._chunk_bytes)
            return self.queue.popleft()


class MonitorSpeaker:
    """监听播放器：把录音数据块实时输出到默认扬声器。

    采样率对齐由 miniaudio 内部重采样处理（16000 -> 设备采样率）。
    """

    def __init__(self, frame_length=CHUNK):
        self.frame_length = frame_length
        self.queue = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = False
        self.dev = None
        self._gen = None
        self.underruns = 0
        self.overruns = 0

    def _playback_generator(self):
        """miniaudio 播放回调生成器（必须已在 start 前 next() 一次）

        回调要求恰好 required_frames 帧（frames * channels * 2 字节）。
        少给会被静音补齐，多给会抛 MiniaudioError，故此处严格对齐字节数。
        """
        required_frames = yield b""
        frame_length = self.frame_length
        chunk_bytes = frame_length * CHANNELS * 2
        q = self.queue
        cond = self.cond
        while True:
            need_bytes = (required_frames or frame_length) * CHANNELS * 2
            out = bytearray()
            with cond:
                # 以整块取出，取出后按需截断以满足精确字节数
                while self.running and len(out) < need_bytes:
                    if q:
                        out.extend(q.popleft())
                    else:
                        self.underruns += 1
                        break
            if len(out) < need_bytes:
                out.extend(bytes(need_bytes - len(out)))
            elif len(out) > need_bytes:
                extra = out[need_bytes:]
                out = out[:need_bytes]
                with cond:
                    self.queue.appendleft(bytes(extra))   # 余量放回队首
            required_frames = yield bytes(out)

    def push(self, chunk):
        """由录音线程调用：压入待播放数据（极快，不阻塞）"""
        with self.cond:
            q = self.queue
            q.append(chunk)
            if len(q) > _MAX_MONITOR_CHUNKS:
                q.popleft()
                self.overruns += 1

    def start(self):
        if self.running:
            return
        self.queue = deque()
        self.running = True
        self.dev = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=CHANNELS,
            sample_rate=RATE,
            buffersize_msec=60,
        )
        self._gen = self._playback_generator()
        next(self._gen)
        self.dev.start(self._gen)

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
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
        self._gen = None
        self.queue = deque()


# ==================================================================
# 主窗口
# ==================================================================
class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="神色语音sensevox", size=(480, 540))
        self.recorder = None
        self.speaker = None
        self.model = None
        self.opencc_converter = None
        self.listening_event = threading.Event()
        self.capturing_hotkey_event = threading.Event()
        self.listener_thread = None
        self.capture_thread = None
        self.was_listening_before_suspend = False
        self._log_line_count = 0

        # 启动监听时锁定的一组配置状态，供后台线程安全访问
        self._monitor_enabled = False
        self._opencc_enabled = False
        self._language = DEFAULT_LANGUAGE

        icon = load_app_icon()
        if icon is not None:
            self.SetIcon(icon)

        # 加载配置
        self.current_hotkey = self.load_setting(HOTKEY_FILE_PATH, DEFAULT_HOTKEY)
        self.current_model_dir = self.load_setting(MODEL_DIR_CONFIG_PATH, "").strip() or get_app_dir()
        initial_monitor_state = self.load_setting(MONITOR_CONFIG_PATH, "false").lower() == "true"
        initial_opencc_state = self.load_setting(OPENCC_ENABLED_PATH, "false").lower() == "true"
        initial_opencc_config = self.load_setting(OPENCC_CONFIG_PATH, OPENCC_OPTIONS[0])
        # 构建 UI
        self._init_ui(initial_monitor_state, initial_opencc_state, initial_opencc_config)
        self.BindEvents()
        self.SetMinSize((440, 500))
        self.Centre()
        self.run_initial_checks()
        self.update_ui_state()

    # ------------------------------------------------------------
    # 模型路径
    # ------------------------------------------------------------
    def model_onnx_path(self):
        return os.path.join(self.current_model_dir, MODEL_ONNX_NAME)

    def tokens_path(self):
        return os.path.join(self.current_model_dir, TOKENS_NAME)

    def missing_model_files(self):
        return [p for p in (self.model_onnx_path(), self.tokens_path()) if not os.path.exists(p)]

    def _model_dir_tooltip(self):
        """模型目录按钮/标签的悬浮提示（含当前持久化目录）"""
        return (
            "模型目录：该目录内需包含 model.onnx 与 tokens.txt。\n"
            "本程序使用模型：sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17\n\n"
            "当前持久化目录：\n" + self.current_model_dir
        )

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _init_ui(self, initial_monitor_state, initial_opencc_state, initial_opencc_config):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # 第一行按钮：启动 / 暂停 / 设置热键
        grid_sizer = wx.FlexGridSizer(rows=1, cols=3, hgap=8, vgap=8)
        for col in range(3):
            grid_sizer.AddGrowableCol(col, 1)

        self.btn_start = wx.Button(panel, label="Start 启动")
        self.btn_stop = wx.Button(panel, label="Stop 暂停")
        self.btn_hotkey = wx.Button(panel, label=f"Hotkey: {self.current_hotkey}")
        for btn in (self.btn_start, self.btn_stop, self.btn_hotkey):
            btn.SetMinSize((-1, 34))
        grid_sizer.Add(self.btn_start, 0, wx.EXPAND | wx.ALL, 2)
        grid_sizer.Add(self.btn_stop, 0, wx.EXPAND | wx.ALL, 2)
        grid_sizer.Add(self.btn_hotkey, 0, wx.EXPAND | wx.ALL, 2)
        main_sizer.Add(grid_sizer, 0, wx.ALL | wx.EXPAND, 10)

        # 模型目录选择：标签 + 按钮同一行，详细说明放气泡提示
        self.lbl_model_dir = wx.StaticText(panel, label="模型目录 Model Dir：")
        self.lbl_model_dir.SetToolTip(self._model_dir_tooltip())
        self.btn_model_dir = wx.Button(panel, label="选择模型目录...")
        self.btn_model_dir.SetMinSize((140, 30))
        self.btn_model_dir.SetToolTip(self._model_dir_tooltip())
        model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_sizer.Add(self.lbl_model_dir, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        model_sizer.Add(self.btn_model_dir, 1, wx.EXPAND)
        main_sizer.Add(model_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 语言选择
        lang_tip = "这个选项几乎没用，无论选择什么都会和 auto 自动的效果一致，请随意选择吧"
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lang_label = wx.StaticText(panel, label="语言 Language：")
        self.lang_label.SetToolTip(lang_tip)
        self.combo_box = wx.ComboBox(panel, value=DEFAULT_LANGUAGE,
                                     choices=LANGUAGES, style=wx.CB_READONLY)
        self.combo_box.SetToolTip(lang_tip)
        lang_sizer.Add(self.lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        lang_sizer.Add(self.combo_box, 1, wx.EXPAND)
        main_sizer.Add(lang_sizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        # 监听复选框（替代原保存录音）
        self.chk_monitor = wx.CheckBox(panel, label="监听 Monitor")
        self.chk_monitor.SetValue(initial_monitor_state)
        self.chk_monitor.SetToolTip(
            "实时输出到扬声器，内容为送入识别前的原始音频。\n"
            "注意：外放时会形成回环自激（啸叫），请佩戴耳机。")
        main_sizer.Add(self.chk_monitor, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

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
        for btn in (self.btn_clear_log, self.btn_copy_log):
            btn.SetMinSize((-1, 34))
        bottom_grid.Add(self.btn_clear_log, 0, wx.EXPAND | wx.ALL, 2)
        bottom_grid.Add(self.btn_copy_log, 0, wx.EXPAND | wx.ALL, 2)
        main_sizer.Add(bottom_grid, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        panel.SetSizer(main_sizer)

    def BindEvents(self):
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start_listening)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop_listening)
        self.btn_hotkey.Bind(wx.EVT_BUTTON, self.on_set_hotkey)
        self.btn_model_dir.Bind(wx.EVT_BUTTON, self.on_choose_model_dir)
        self.btn_clear_log.Bind(wx.EVT_BUTTON, self.on_clear_log)
        self.btn_copy_log.Bind(wx.EVT_BUTTON, self.on_copy_log)
        self.combo_box.Bind(wx.EVT_COMBOBOX, self.on_language_select)
        self.chk_opencc.Bind(wx.EVT_CHECKBOX, self.on_opencc_toggle)
        self.combo_opencc.Bind(wx.EVT_COMBOBOX, self.on_opencc_select)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        if hasattr(wx, "EVT_POWER_SUSPENDING"):
            self.Bind(wx.EVT_POWER_SUSPENDING, self.on_power_suspending)
        if hasattr(wx, "EVT_POWER_RESUME"):
            self.Bind(wx.EVT_POWER_RESUME, self.on_power_resume)

    def log(self, message, level="INFO"):
        """线程安全日志"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp} {level}] {message}\n"

        def append_and_prune():
            self.log_text.AppendText(log_entry)
            self._log_line_count += 1
            if self._log_line_count > _MAX_LOG_LINES:
                extra = self._log_line_count - _MAX_LOG_LINES
                end_pos = self.log_text.XYToPosition(0, extra)
                if end_pos != wx.NOT_FOUND and end_pos > 0:
                    self.log_text.Freeze()
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
        self.btn_model_dir.Enable(not is_running_or_capturing)
        self.combo_box.Enable(not is_running_or_capturing)
        self.chk_monitor.Enable(not is_running_or_capturing)
        self.chk_opencc.Enable(not is_running_or_capturing and OPENCC_AVAILABLE)
        self.combo_opencc.Enable(not is_running_or_capturing and self.chk_opencc.IsChecked() and OPENCC_AVAILABLE)

        if is_capturing_hotkey:
            self.btn_hotkey.SetLabel("Capturing...")
        else:
            self.btn_hotkey.SetLabel(f"Hotkey: {self.current_hotkey}")

    # ------------------------------------------------------------
    # 音频资源管理
    # ------------------------------------------------------------
    def _initialize_audio_system(self):
        """初始化录音设备，并按需启动监听播放"""
        self.log("Initializing PyMiniaudio recording device... 初始化 PyMiniaudio 录音设备...", "INFO")
        try:
            self.recorder = PyMiniaudioRecorder(frame_length=CHUNK)
            self.recorder.start()
            if not self.recorder.wait_ready(timeout=1.0):
                raise AudioDeviceError("未检测到音频输入")
        except Exception as e:
            self.log(f"Failed to initialize audio device 录音设备初始化失败: {e}", "ERROR")
            self._cleanup_audio_resources()
            return False

        if self._monitor_enabled:
            try:
                self.speaker = MonitorSpeaker(frame_length=CHUNK)
                self.speaker.start()
                self.recorder.set_monitor_sink(self.speaker.push)
                self.log("Monitor output started (raw audio -> speaker). 监听已开启（原始音频 → 扬声器）。", "SUCCESS")
                self.log("Tip: use headphones to avoid feedback howling. 提示：请佩戴耳机，外放会产生回环啸叫。", "WARNING")
            except Exception as e:
                self.speaker = None
                self.recorder.set_monitor_sink(None)
                self.log(f"Monitor output unavailable 监听输出不可用: {e}", "WARNING")

        return True

    def _cleanup_audio_resources(self):
        """释放录音与监听资源"""
        if self.recorder:
            self.recorder.set_monitor_sink(None)
        if self.speaker:
            try:
                self.speaker.stop()
            except Exception as e:
                self.log(f"Error stopping monitor 停止监听时出错: {e}", "WARNING")
            finally:
                self.speaker = None
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                self.log(f"Error stopping audio device 停止录音设备时出错: {e}", "WARNING")
            finally:
                self.recorder = None
        self.log("Audio resources released. 音频资源已释放。", "DEBUG")

    # ------------------------------------------------------------
    # 启动 / 停止监听
    # ------------------------------------------------------------
    def on_start_listening(self, event):
        if not self.run_initial_checks():
            wx.MessageBox("初始检查未通过，请查看日志。\nInitial checks failed, please see the log.",
                          "错误 Error", wx.OK | wx.ICON_ERROR)
            return
        if self.listening_event.is_set():
            self.log("Listening is already running. 监听已在运行中。", "WARNING")
            return

        # 先锁定 UI 状态供后台线程使用（音频初始化需要知道是否开启监听）
        self._monitor_enabled = self.chk_monitor.IsChecked()
        self._opencc_enabled = self.chk_opencc.IsChecked()
        self._language = self.combo_box.GetValue()

        if not self._initialize_audio_system():
            self.update_ui_state()
            return

        # 加载 / 复用语音识别模型
        if self.model is None:
            self.log("Loading Sherpa-ONNX SenseVoice model... 正在加载 SenseVoice 语音识别模型...", "INFO")
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

        # 初始化 OpenCC 转换器
        if self._opencc_enabled and OPENCC_AVAILABLE:
            try:
                selected_config = self.combo_opencc.GetValue().split()[0]
                self.opencc_converter = opencc.OpenCC(selected_config)
                self.log(f"OpenCC converter initialized (config: {selected_config}). OpenCC 转换器已初始化。", "INFO")
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
        if not self.listening_event.is_set():
            return
        self.log("Stopping listening... 正在停止监听...", "INFO")
        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
        self._cleanup_audio_resources()
        self.opencc_converter = None
        self.log("Listening stopped, resources released. 监听已停止，资源已释放。", "INFO")
        self.update_ui_state()

    def _create_recognizer(self):
        """创建 SenseVoice 离线识别器"""
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self.model_onnx_path(),
            tokens=self.tokens_path(),
            language=self._language if self._language != "auto" else "",
            use_itn=True,
            num_threads=2,
            provider="cpu",
        )

    def listen_loop(self):
        """主监听循环（含音频设备自动重连）"""
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
        """执行一次完整的「录音 → 识别 → 输出」流程"""
        blocked_keys = []
        hotkey_parts = [part.strip() for part in self.current_hotkey.split("+")]
        is_caps_lock_hotkey = self.current_hotkey.lower() == "caps lock"

        try:
            raw_audio_bytes, duration = self.record_audio(hotkey_parts, blocked_keys)

            if raw_audio_bytes and self.listening_event.is_set():
                ok, transcription_result = self.transcribe_local(raw_audio_bytes)
                self._unblock_keys(blocked_keys)

                if ok and transcription_result:
                    processed_text = self.process_text(transcription_result)

                    if self._opencc_enabled and OPENCC_AVAILABLE and self.opencc_converter:
                        try:
                            processed_text = self.opencc_converter.convert(processed_text)
                            self.log("OpenCC conversion completed. OpenCC 转换完成。", "DEBUG")
                        except Exception as e:
                            self.log(f"OpenCC conversion failed OpenCC 转换失败: {e}", "ERROR")

                    self.type_text(processed_text)

                    if is_caps_lock_hotkey:
                        try:
                            keyboard.press_and_release("caps lock")
                        except Exception as e:
                            self.log(f"Failed to simulate Caps Lock 模拟 Caps Lock 失败: {e}", "WARNING")

        except AudioDeviceError:
            raise
        except Exception as e:
            if self.listening_event.is_set():
                self.log(f"Error in record/transcribe cycle 录音/识别周期出错: {e}", "ERROR")
                traceback.print_exc()
        finally:
            self._unblock_keys(blocked_keys)

    def record_audio(self, hotkey_parts, blocked_keys_list):
        """录音：热键按下开始，松开结束"""
        frames = []
        recording_started = False
        start_time = 0.0
        hotkey = self.current_hotkey
        is_caps_lock_hotkey = hotkey.lower() == "caps lock"

        listening_event_is_set = self.listening_event.is_set
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
        """本地语音识别（送入识别的音频即监听输出的音频）"""
        if not self.model or not SHERPA_AVAILABLE:
            return False, None

        audio_for_transcription = (
            np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) * _INT16_SCALE
        )

        stream = self.model.create_stream()
        start_time = time.perf_counter()
        try:
            stream.accept_waveform(sample_rate=RATE, waveform=audio_for_transcription)
            stream.accept_waveform(sample_rate=RATE, waveform=_TAIL_PADDING)  # 尾部静音
            self.model.decode_stream(stream)
            transcribed_text = stream.result.text
        except Exception as e:
            self.log(f"Speech recognition exception 语音识别异常: {e}", "ERROR")
            return False, None
        finally:
            del stream   # 显式释放 C++ 流，防止内存泄漏

        duration = time.perf_counter() - start_time
        self.log(f"Transcription completed 识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")
        return True, transcribed_text

    # ------------------------------------------------------------
    # 初始检查与配置
    # ------------------------------------------------------------
    def run_initial_checks(self):
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

        missing = self.missing_model_files()
        if missing:
            for f in missing:
                self.log(f"Model file not found 模型文件不存在: {f}", "ERROR")
            self.log(f"Current model dir 当前模型目录: {self.current_model_dir}", "ERROR")
            all_ok = False
        else:
            self.log(f"Model files complete 模型文件完整: {self.current_model_dir}", "INFO")

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
    def on_choose_model_dir(self, event):
        """选择模型文件并持久化其所在目录。

        用 FileDialog 而非 DirDialog：目录选择器看不到文件内容，无法确认
        目录里是不是真的放了模型；让用户直接点选 model.onnx 更直观。
        """
        current_dir = self.current_model_dir if os.path.isdir(self.current_model_dir) else get_app_dir()
        with wx.FileDialog(
            self,
            "选择模型文件 model.onnx（同目录需有 tokens.txt）",
            defaultDir=current_dir,
            defaultFile=MODEL_ONNX_NAME,
            wildcard="ONNX 模型 (*.onnx)|*.onnx|所有文件 (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            chosen_file = dlg.GetPath()

        chosen = os.path.dirname(chosen_file)
        if os.path.basename(chosen_file).lower() != MODEL_ONNX_NAME:
            wx.MessageBox(
                "请选择名为 " + MODEL_ONNX_NAME + " 的文件。",
                "文件不符", wx.OK | wx.ICON_ERROR,
            )
            self.log(f"Model file rejected (unexpected name): {chosen_file}", "ERROR")
            return

        missing = [n for n in (MODEL_ONNX_NAME, TOKENS_NAME)
                   if not os.path.exists(os.path.join(chosen, n))]
        if missing:
            wx.MessageBox(
                "所选目录缺少模型文件：\n\n" + "\n".join(missing)
                + "\n\n请选择包含 model.onnx 与 tokens.txt 的目录。",
                "模型缺失", wx.OK | wx.ICON_ERROR,
            )
            self.log(f"Model dir rejected (missing {', '.join(missing)}): {chosen}", "ERROR")
            return

        self.current_model_dir = chosen
        tip = self._model_dir_tooltip()
        self.lbl_model_dir.SetToolTip(tip)
        self.btn_model_dir.SetToolTip(tip)
        self.save_setting(MODEL_DIR_CONFIG_PATH, chosen)
        # 目录变更后需重新加载模型
        self.model = None
        self.log(f"Model dir set to 模型目录已设为: {chosen}", "SUCCESS")

    def on_close(self, event):
        """窗口关闭：保存设置并清理资源"""
        self.log("Closing window, saving settings and cleaning up resources... 正在关闭窗口，保存设置并清理资源...", "INFO")

        self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        self.save_setting(MODEL_DIR_CONFIG_PATH, self.current_model_dir)
        self.save_setting(MONITOR_CONFIG_PATH, self.chk_monitor.IsChecked())
        self.save_setting(OPENCC_ENABLED_PATH, self.chk_opencc.IsChecked())
        self.save_setting(OPENCC_CONFIG_PATH, self.combo_opencc.GetValue())

        self.listening_event.clear()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)

        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except Exception:
                pass

        self._cleanup_audio_resources()
        self.Destroy()

    def on_set_hotkey(self, event):
        """开始捕获热键"""
        if self.capturing_hotkey_event.is_set():
            return
        self.capturing_hotkey_event.set()
        self.update_ui_state()
        self.log("Hotkey capture active, please press new hotkey... 热键捕获已开启，请按下新的热键...", "INFO")
        self.capture_thread = threading.Thread(target=self.capture_hotkey_thread_func, daemon=True)
        self.capture_thread.start()

    def capture_hotkey_thread_func(self):
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

    def on_language_select(self, event):
        self.log(f"Language preference set to 语言偏好已设为: '{self.combo_box.GetValue()}'", "INFO")

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
    # 文本处理与输出
    # ------------------------------------------------------------
    def type_text(self, text_to_type):
        """用 SendInput 一次性把整段文本注入光标位置（含 Unicode / emoji）"""
        if not text_to_type:
            return
        try:
            send_unicode_text(text_to_type)
            self.log(f"Keyboard typed 键盘输出: '{text_to_type}'", "SUCCESS")
        except OSError as e:
            # 目标窗口权限更高（以管理员运行）时 UIPI 会拦截 SendInput
            self.log(f"Keyboard input failed 键盘输入失败: {e}", "ERROR")

    @staticmethod
    def _apply_punct_rule(text):
        """若标点数量 ≤1 则去除所有标点"""
        if not text:
            return text
        count = 0
        for char in text:
            if char in _PUNCTUATION_SET:
                count += 1
                if count > 1:
                    return text
        return text.translate(_TRANSLATE_TABLE)

    def process_text(self, text):
        """文本清理：剥离 SenseVoice 停顿控制标记，再按标点规则清理。"""
        if not text or not isinstance(text, str):
            return text
        if _PAUSE_MARKER in text:
            text = _RE_PAUSE_MARKER.sub("", text)
        return self._apply_punct_rule(text)

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
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if file_path == MODEL_DIR_CONFIG_PATH:
                        return content.strip()
                    content = content.strip()
                    return content if content else default_value
        except Exception as e:
            self.log(f"Failed to load config file 读取配置文件 {os.path.basename(file_path)} 失败: {e}", "WARNING")
        return default_value

    def save_setting(self, file_path, value_to_save):
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


def main():
    if sys.platform != "win32":
        print("sensevox only supports Windows.", file=sys.stderr)
        return 1
    app = wx.App(False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
