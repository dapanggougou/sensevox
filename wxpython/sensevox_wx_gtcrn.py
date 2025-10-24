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
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_path = os.path.abspath(".")
    return os.path.join(base_path, "assets", relative_path)


MODEL_DIR_BASE = "sensevoicesmallonnx"
MODEL_DIR = get_asset_path(MODEL_DIR_BASE)
MODEL_FILENAME = "model.onnx"
TOKENS_FILENAME = "tokens.txt"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TOKENS_FILE_PATH = os.path.join(MODEL_DIR, TOKENS_FILENAME)
GTCRN_MODEL_PATH = get_asset_path("gtcrn_simple.onnx")

CHUNK = 1024 * 2
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH_BYTES = 2
MIN_RECORD_SECONDS = 0.3
DEFAULT_HOTKEY = "space"
MAX_FILENAME_TEXT_LEN = 15
all_punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』“”‘’·～《》〈〉﹏——……〜・〝〟‹›'''

HOTKEY_FILE = "hotkey.txt"
GTCRN_CONFIG_FILE = "gtcrn_config.txt"
SAVE_RECORDING_CONFIG_FILE = "save_recording_config.txt"
OPENCC_CONFIG_FILE = "opencc_config.txt"
OPENCC_ENABLED_FILE = "opencc_enabled.txt"

HOTKEY_FILE_PATH = get_asset_path(HOTKEY_FILE)
GTCRN_CONFIG_PATH = get_asset_path(GTCRN_CONFIG_FILE)
SAVE_RECORDING_CONFIG_PATH = get_asset_path(SAVE_RECORDING_CONFIG_FILE)
OPENCC_CONFIG_PATH = get_asset_path(OPENCC_CONFIG_FILE)
OPENCC_ENABLED_PATH = get_asset_path(OPENCC_ENABLED_FILE)

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

    def _capture_generator(self):
        _ = (yield)
        chunk_bytes = self.frame_length * self.channels * self.sample_width
        while True:
            data = (yield)
            if not data:
                continue
            with self.lock:
                self.buffer.extend(data)
                while len(self.buffer) >= chunk_bytes:
                    chunk = bytes(self.buffer[:chunk_bytes])
                    del self.buffer[:chunk_bytes]
                    self.queue.append(chunk)
                    if len(self.queue) > self.max_queue_chunks:
                        self.queue.popleft()
                self.received_data = True
                self.last_data_time = time.monotonic()
                self.cond.notify_all()

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
            buffersize_msec=bs_msec
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
                return np.zeros(self.frame_length, dtype=np.int16).tobytes()
            return self.queue.popleft()

    def delete(self):
        pass


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="神色语音sensevox", size=(500, 520))
        self.recorder = None
        self.model = None
        self.gtcrn_denoiser = None
        self.recognizer_stream = None
        self.opencc_converter = None
        self.listening_event = threading.Event()
        self.capturing_hotkey_event = threading.Event()
        self.listener_thread = None
        self.capture_thread = None

        icon_path = get_asset_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_ICO))

        self.current_hotkey = self.load_setting(HOTKEY_FILE_PATH, DEFAULT_HOTKEY)
        initial_gtcrn_state = self.load_setting(GTCRN_CONFIG_PATH, "true").lower() == "true"
        initial_save_state = self.load_setting(SAVE_RECORDING_CONFIG_PATH, "true").lower() == "true"
        initial_opencc_state = self.load_setting(OPENCC_ENABLED_PATH, "false").lower() == "true"
        initial_opencc_config = self.load_setting(OPENCC_CONFIG_PATH, OPENCC_OPTIONS[0])

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

        self.BindEvents()
        self.SetMinSize((450, 520))
        self.Centre()
        self.run_initial_checks()
        self.update_ui_state()

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

    def log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp} {level}] {message}\n"

        def append_and_prune():
            self.log_text.AppendText(log_entry)
            max_lines = 100
            lines = self.log_text.GetNumberOfLines()
            if lines > max_lines:
                extra = lines - max_lines
                end_pos = self.log_text.XYToPosition(0, extra)
                if end_pos != wx.NOT_FOUND and end_pos > 0:
                    self.log_text.Remove(0, end_pos)

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
        self.log("Audio resources cleaned up.", "DEBUG")

    def on_start_listening(self, event):
        if not self.run_initial_checks():
            wx.MessageBox("Initial checks failed. Please see the log for details.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.listening_event.is_set():
            self.log("Listener already running.", "WARNING")
            return
        if not self._initialize_audio_system():
            self.update_ui_state()
            return
        if self.model is None:
            self.log("Loading Sherpa-ONNX recognizer...", "INFO")
            try:
                self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=MODEL_FILE_PATH, tokens=TOKENS_FILE_PATH,
                    language=self.current_language if self.current_language != "auto" else "",
                    use_itn=True, num_threads=1, provider="cpu"
                )
                self.log("Recognizer loaded successfully.", "SUCCESS")
            except Exception as e:
                self.log(f"Recognizer loading failed: {e}", "ERROR")
                self.model = None
                self._cleanup_audio_resources()
                self.update_ui_state()
                return
        if self.chk_gtcrn_enhance.IsChecked() and os.path.exists(GTCRN_MODEL_PATH):
            self.log("Loading GTCRN denoiser model...", "INFO")
            try:
                config = sherpa_onnx.OfflineSpeechDenoiserConfig(
                    model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                        gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=GTCRN_MODEL_PATH),
                        provider="cpu", num_threads=1,
                    ))
                self.gtcrn_denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
                self.log("GTCRN denoiser loaded successfully.", "SUCCESS")
            except Exception as e:
                self.log(f"Failed to load GTCRN denoiser: {e}", "ERROR")
                self.gtcrn_denoiser = None
        if self.chk_opencc.IsChecked() and OPENCC_AVAILABLE:
            try:
                selected_config = self.combo_opencc.GetValue().split()[0]
                self.opencc_converter = opencc.OpenCC(selected_config)
                self.log(f"OpenCC converter initialized with config: {selected_config}", "INFO")
            except Exception as e:
                self.log(f"Failed to initialize OpenCC converter: {e}", "ERROR")
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
        self.gtcrn_denoiser = None
        self.opencc_converter = None
        self.log("Listener stopped and resources released.", "INFO")
        self.update_ui_state()

    def listen_loop(self):
        MAX_RETRIES = 5
        RETRY_DELAY_SECONDS = 2
        self.log(f"Listener thread started. Monitoring hotkey: '{self.current_hotkey}'.")
        while self.listening_event.is_set():
            try:
                self.perform_record_and_transcribe()
                time.sleep(0.02)
            except AudioDeviceError as e:
                if not self.listening_event.is_set():
                    break
                self.log(f"Audio device error: {e}. Starting reconnection procedure.", "ERROR")
                self._cleanup_audio_resources()
                reconnected = False
                for i in range(MAX_RETRIES):
                    if not self.listening_event.is_set():
                        break
                    self.log(f"Reconnect attempt {i + 1}/{MAX_RETRIES} in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    if self._initialize_audio_system():
                        self.log("Reconnection successful.", "SUCCESS")
                        reconnected = True
                        break
                    else:
                        self.log(f"Reconnect attempt {i + 1} failed.", "WARNING")
                if not reconnected and self.listening_event.is_set():
                    self.log(f"Failed to reconnect after {MAX_RETRIES} attempts. Stopping listener.", "ERROR")
                    wx.CallAfter(self.on_stop_listening, None)
                    break
            except Exception as e:
                if self.listening_event.is_set():
                    self.log(f"Unhandled error in listener loop: {e}", "ERROR")
                    traceback.print_exc()
                break
        self.log("Listener thread finished.")

    def perform_record_and_transcribe(self):
        blocked_keys = []
        hotkey_parts = [part.strip() for part in self.current_hotkey.split('+')]
        is_caps_lock_hotkey = (self.current_hotkey.lower() == 'caps lock')
        try:
            raw_audio_data, duration = self.record_audio(hotkey_parts, blocked_keys)
            if raw_audio_data and self.listening_event.is_set():
                ok, transcription_result, audio_for_saving = self.transcribe_local(raw_audio_data)
                self._unblock_keys(blocked_keys)
                if ok and transcription_result:
                    processed_text = self.process_text(transcription_result)
                    if self.chk_opencc.IsChecked() and OPENCC_AVAILABLE and self.opencc_converter:
                        try:
                            processed_text = self.opencc_converter.convert(processed_text)
                            self.log(f"Text converted with OpenCC", "INFO")
                        except Exception as e:
                            self.log(f"OpenCC conversion failed: {e}", "ERROR")
                    self.type_text(processed_text)
                    if is_caps_lock_hotkey:
                        try:
                            keyboard.press_and_release('caps lock')
                        except Exception as e:
                            self.log(f"Failed to toggle Caps Lock state: {e}", "WARNING")
                if self.chk_save_recording.IsChecked():
                    self.save_audio_with_transcription(audio_for_saving, transcription_result)
        except AudioDeviceError:
            raise
        except Exception as e:
            if self.listening_event.is_set():
                self.log(f"Error during record/process cycle: {e}", "ERROR")
                traceback.print_exc()
        finally:
            self._unblock_keys(blocked_keys)

    def record_audio(self, hotkey_parts, blocked_keys_list):
        frames = deque()
        recording_started = False
        start_time = 0
        hotkey = self.current_hotkey
        is_caps_lock_hotkey = (hotkey.lower() == 'caps lock')
        tail_chunks = 1

        while self.listening_event.is_set():
            try:
                samples_bytes = self.recorder.read()
                pressed = KEYBOARD_AVAILABLE and keyboard.is_pressed(hotkey)

                if pressed:
                    if not recording_started:
                        self.log(f"Recording started (hotkey '{hotkey}' detected)...", "DEBUG")
                        start_time = time.time()
                        recording_started = True
                        if not is_caps_lock_hotkey:
                            for key_part in hotkey_parts:
                                try:
                                    keyboard.block_key(key_part)
                                    blocked_keys_list.append(key_part)
                                except Exception as e:
                                    self.log(f"Could not block '{key_part}': {e}", "WARNING")
                    frames.append(samples_bytes)
                elif recording_started:
                    frames.append(samples_bytes)
                    for _ in range(tail_chunks):
                        try:
                            extra = self.recorder.read()
                            frames.append(extra)
                        except Exception:
                            break
                    break

            except AudioDeviceError:
                raise
            except Exception as e:
                self.log(f"Unexpected error reading audio stream: {e}", "ERROR")
                raise AudioDeviceError(f"Unexpected stream error: {e}")

        if not recording_started:
            return None, 0.0

        duration = time.time() - start_time
        if duration < MIN_RECORD_SECONDS and self.listening_event.is_set():
            self.log(f"Recording too short ({duration:.2f}s), ignored.", "WARNING")
            return None, duration

        return b''.join(frames), duration

    def transcribe_local(self, audio_input_bytes):
        if not self.model or not SHERPA_AVAILABLE:
            return False, None, np.frombuffer(audio_input_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        original_audio_np = np.frombuffer(audio_input_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_for_transcription = original_audio_np
        use_gtcrn = self.chk_gtcrn_enhance.IsChecked() and self.gtcrn_denoiser is not None
        if use_gtcrn:
            self.log("Applying GTCRN enhancement...", "DEBUG")
            try:
                enhanced_audio = self.gtcrn_denoiser(original_audio_np, RATE)
                audio_for_transcription = np.array(enhanced_audio.samples, dtype=np.float32)
                log_msg_part = "denoised" if self.chk_save_recording.IsChecked() else "original"
                self.log(f"GTCRN enhancement applied. Saving {log_msg_part} audio if enabled.", "INFO")
            except Exception as e:
                self.log(f"GTCRN enhancement failed: {e}. Using original audio.", "ERROR")
        else:
            if self.chk_save_recording.IsChecked():
                self.log("GTCRN not used. Saving original audio.", "INFO")
        self.recognizer_stream = self.model.create_stream()
        start_time = time.perf_counter()
        try:
            self.recognizer_stream.accept_waveform(sample_rate=RATE, waveform=audio_for_transcription)
            tail_padding = np.zeros(int(0.5 * RATE), dtype=np.float32)
            self.recognizer_stream.accept_waveform(sample_rate=RATE, waveform=tail_padding)
            self.model.decode_stream(self.recognizer_stream)
            transcribed_text = self.recognizer_stream.result.text
        except Exception as e:
            self.log(f"Sherpa-ONNX transcription error: {e}", "ERROR")
            return False, None, original_audio_np
        duration = time.perf_counter() - start_time
        self.log(f"识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")
        if transcribed_text:
            return True, transcribed_text, audio_for_transcription
        return False, None, audio_for_transcription

    def save_audio_with_transcription(self, audio_np_data, transcription):
        if audio_np_data is None or (hasattr(audio_np_data, "size") and audio_np_data.size == 0):
            self.log("Audio data is empty, skipping save.", "WARNING")
            return
        try:
            audio_np_data = (audio_np_data * 32767).clip(-32768, 32767).astype(np.int16)
            audio_bytes = audio_np_data.tobytes()
            recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "录音")
            os.makedirs(recordings_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_text = self.sanitize_filename_part(transcription)
            enh_tag = "_enhanced" if self.chk_gtcrn_enhance.IsChecked() else ""
            filename = f"rec_{timestamp}{enh_tag}_{sanitized_text}.wav" if sanitized_text else f"rec_{timestamp}{enh_tag}.wav"
            filepath = os.path.join(recordings_dir, filename)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)
            self.log(f"Recording saved: {os.path.basename(filepath)}", "INFO")
        except Exception as e:
            self.log(f"Error saving audio: {e}", "ERROR")
            traceback.print_exc()

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
            self.log("GTCRN model (gtcrn_simple.onnx) not found in assets. Enhancement disabled.", "WARNING")
            self.chk_gtcrn_enhance.SetValue(False)
            self.chk_gtcrn_enhance.Disable()
        else:
            self.log("GTCRN model found. Enhancement option is available.", "INFO")
            self.chk_gtcrn_enhance.Enable()
        if not OPENCC_AVAILABLE:
            self.log("OpenCC library not found. Conversion feature disabled.", "WARNING")
            self.chk_opencc.SetValue(False)
            self.chk_opencc.Disable()
            self.combo_opencc.Disable()
        else:
            self.log("OpenCC library found. Conversion feature is available.", "INFO")
            self.chk_opencc.Enable()
            self.combo_opencc.Enable(self.chk_opencc.IsChecked())
        if all_ok:
            self.log("All critical checks passed. Ready to start.", "SUCCESS")
        else:
            self.log("One or more critical checks failed. Please resolve the errors.", "ERROR")
        return all_ok

    def on_close(self, event):
        self.log("Window closing, saving settings and cleaning up...", "INFO")
        self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        self.save_setting(GTCRN_CONFIG_PATH, self.chk_gtcrn_enhance.IsChecked())
        self.save_setting(SAVE_RECORDING_CONFIG_PATH, self.chk_save_recording.IsChecked())
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
        if self.capturing_hotkey_event.is_set():
            return
        self.capturing_hotkey_event.set()
        self.update_ui_state()
        self.log("Hotkey capture active. Press desired key/combination...", "INFO")
        self.capture_thread = threading.Thread(target=self.capture_hotkey_thread_func, daemon=True)
        self.capture_thread.start()

    def on_combo_select(self, event):
        self.current_language = self.combo_box.GetValue()
        self.log(f"Language preference set to: '{self.current_language}'", "INFO")
        self.log("Language will be used on next start.", "DEBUG")

    def on_opencc_toggle(self, event):
        self.combo_opencc.Enable(self.chk_opencc.IsChecked())
        self.update_ui_state()
        self.log(f"OpenCC {'enabled' if self.chk_opencc.IsChecked() else 'disabled'}.", "INFO")

    def on_opencc_select(self, event):
        selected_config = self.combo_opencc.GetValue()
        self.log(f"OpenCC config set to: '{selected_config}'", "INFO")
        self.save_setting(OPENCC_CONFIG_PATH, selected_config)

    def on_clear_log(self, event):
        self.log_text.Clear()

    def on_copy_log(self, event):
        log_content = self.log_text.GetValue()
        if log_content and wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(log_content))
            wx.TheClipboard.Close()
            self.log("Logs copied to clipboard.", "SUCCESS")

    def type_text(self, text_to_type):
        if not text_to_type:
            return
        
        try:
            if sys.platform == "win32":
                keyboard.write(text_to_type)
                self.log(f"Typed: '{text_to_type}'", "SUCCESS")
                
            elif sys.platform.startswith("linux"):
                original_clipboard = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.1)
                keyboard.press_and_release('shift+insert')
                time.sleep(0.1)
                pyperclip.copy(original_clipboard)
                self.log(f"Typed (Linux): '{text_to_type}'", "SUCCESS")
            elif sys.platform == "darwin":
                original_clipboard = pyperclip.paste()
                pyperclip.copy(text_to_type)
                time.sleep(0.1)
                keyboard.press_and_release('command+v')
                time.sleep(0.1)
                pyperclip.copy(original_clipboard)
                self.log(f"Typed (Mac): '{text_to_type}'", "SUCCESS")
                
            else:
                keyboard.write(text_to_type)
                self.log(f"Typed (Unknown OS): '{text_to_type}'", "SUCCESS")
                
        except Exception as e:
            self.log(f"Keyboard input failed: {e}", "ERROR")

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

    def capture_hotkey_thread_func(self):
        try:
            new_hotkey = keyboard.read_hotkey(suppress=False)
            if len(new_hotkey) > 30:
                self.log("Captured hotkey is too long, ignoring.", "WARNING")
                return
            self.current_hotkey = new_hotkey
            self.log(f"Captured: '{new_hotkey}'", "SUCCESS")
            self.save_setting(HOTKEY_FILE_PATH, self.current_hotkey)
        except Exception as e:
            self.log(f"Hotkey capture failed: {e}", "ERROR")
        finally:
            self.capturing_hotkey_event.clear()
            wx.CallAfter(self.update_ui_state)

    def load_setting(self, file_path, default_value):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    return content if content else default_value
        except Exception as e:
            self.log(f"Failed to load setting from {os.path.basename(file_path)}: {e}", "WARNING")
        return default_value

    def save_setting(self, file_path, value_to_save):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(value_to_save))
            self.log(f"Setting saved to {os.path.basename(file_path)}.", "DEBUG")
        except Exception as e:
            self.log(f"Failed to save setting to {os.path.basename(file_path)}: {e}", "ERROR")

    def process_text(self, text):
        if not isinstance(text, str):
            return text
        punctuation_count = sum(1 for char in text if char in all_punctuation)
        if punctuation_count <= 1:
            return text.translate(str.maketrans('', '', all_punctuation))
        return text

    def sanitize_filename_part(self, text, max_len=MAX_FILENAME_TEXT_LEN):
        if not text or not isinstance(text, str):
            return ""
        sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]+', '', text)
        sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
        return sanitized[:max_len].strip('_') if len(sanitized) > max_len else sanitized


if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
