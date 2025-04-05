# --- START OF MODIFIED sensevox.py ---

# --- START OF REFACTORED sensevox.py (Back to PyAudio, Always-On Stream) ---

import datetime
import os
import re
from collections import deque
import sys
import threading
import time
import traceback
import wave
import flet as ft
# --- MODIFIED: Use PyAudio ---
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("[INFO] Successfully imported PyAudio.")
except ImportError as e:
    print(f"[ERROR] Failed to import PyAudio: {e}")
    print("[ERROR] Please install it: 'pip install pyaudio'")
    print("[ERROR] On Linux/macOS, you might need system packages first (e.g., portaudio19-dev, portaudio).")
    PYAUDIO_AVAILABLE = False
except Exception as e: # Catch other potential loading errors
    print(f"[ERROR] Error loading PyAudio or its dependencies: {e}")
    PYAUDIO_AVAILABLE = False
# ---------------------------
import keyboard
import numpy as np
# import soundfile as sf # Keep if needed elsewhere

# --- Local Model Import (Sherpa-ONNX) ---
_SHERPA_IMPORT_ERROR = None
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
    print("[INFO] Successfully imported sherpa_onnx.")
except ImportError as e:
    print(f"[ERROR] Failed to import sherpa_onnx: {e}")
    _SHERPA_IMPORT_ERROR = e
    SHERPA_AVAILABLE = False
# --------------------------

# --- Asset Path Function ---
def get_asset_path(relative_path):
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        try: base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError: base_path = os.path.abspath(".")
    return os.path.join(base_path, "assets", relative_path)

# ------------------- Configuration -------------------
MODEL_DIR_BASE = "sensevoicesmallonnx"
MODEL_DIR = get_asset_path(MODEL_DIR_BASE)
MODEL_FILENAME = "model.onnx"
TOKENS_FILENAME = "tokens.txt"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TOKENS_FILE_PATH = os.path.join(MODEL_DIR, TOKENS_FILENAME)
# --- PyAudio Configuration ---
CHUNK = 1024 * 2 # Increased chunk size slightly for potentially less frequent reads
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Sherpa SenseVoice model expects 16kHz
# ---------------------------
MIN_RECORD_SECONDS = 0.5
LANG_OPTIONS = ["auto", "zh", "en", "yue", "ja", "ko"]
DEFAULT_HOTKEY = "space"
HOTKEY_FILE = "hotkey.txt"
HOTKEY_FILE_PATH = get_asset_path(HOTKEY_FILE)
LOG_MAX_LINES = 200
MAX_FILENAME_TEXT_LEN = 15
# ---------------------------------------------

# --- Helper Functions for Hotkey Persistence ---
# (load_hotkey, save_hotkey - unchanged)
def load_hotkey():
    try:
        if os.path.exists(HOTKEY_FILE_PATH):
            with open(HOTKEY_FILE_PATH, 'r', encoding='utf-8') as f:
                loaded_key = f.read().strip()
                if loaded_key:
                    print(f"[INFO] Loaded hotkey '{loaded_key}' from {HOTKEY_FILE}")
                    return loaded_key
                else:
                    print(f"[WARNING] Hotkey file '{HOTKEY_FILE}' was empty, using default.")
        else:
            print(f"[INFO] Hotkey file '{HOTKEY_FILE}' not found, using default.")
    except Exception as e:
        print(f"[ERROR] Unexpected error loading hotkey: {e}. Using default.")
    return DEFAULT_HOTKEY

def save_hotkey(key_to_save):
    try:
        os.makedirs(os.path.dirname(HOTKEY_FILE_PATH), exist_ok=True)
        with open(HOTKEY_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(key_to_save)
        print(f"[INFO] Saved hotkey '{key_to_save}' to {HOTKEY_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Unexpected error saving hotkey: {e}")
    return False

# --- Global Variables / State ---
# --- MODIFIED: Use PyAudio instance and stream ---
p = None            # PyAudio instance
audio_stream = None # PyAudio stream object
# -------------------------------------------------
model = None
is_listening = False
listener_thread = None
current_hotkey = load_hotkey()
is_capturing_hotkey = False
capture_thread = None
current_selected_language_value = "auto"
all_punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、！？：；（）【】「」『』“”‘’·～《》〈〉﹏——……〜・〝〟‹›'''

# --- Flet Refs ---
# (Unchanged: language_chip_row, log_view, page_ref, start_button, stop_button, set_hotkey_button, status_text)
language_chip_row = ft.Ref[ft.Row]()
log_view = ft.Ref[ft.ListView]()
page_ref = ft.Ref[ft.Page]()
start_button = ft.Ref[ft.ElevatedButton]()
stop_button = ft.Ref[ft.ElevatedButton]()
set_hotkey_button = ft.Ref[ft.ElevatedButton]()
status_text = ft.Ref[ft.Text]()
# --------------------------------

# ***** Log function (Optimized Page Update Check) *****
# (Unchanged)
def log(page: ft.Page, list_view_instance: ft.ListView, msg, level="INFO"):
    if not list_view_instance:
        print(f"[{level}] {msg} (GUI log skipped: ListView not ready)")
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    color_map = { "ERROR": ft.Colors.RED_500, "WARNING": ft.Colors.ORANGE_500, "SUCCESS": ft.Colors.GREEN_500, "INFO": ft.Colors.BLUE_500, "DEBUG": ft.Colors.PURPLE_300, }
    color = color_map.get(level, ft.Colors.GREY_500)
    log_entry = ft.Text(f"[{timestamp} {level}] {msg}", color=color, size=12, selectable=True)
    try:
        list_view_instance.controls.append(log_entry)
        if len(list_view_instance.controls) > LOG_MAX_LINES: list_view_instance.controls.pop(0)
        list_view_instance.scroll_to(offset=-1, duration=100)
        current_page = page_ref.current
        if current_page and current_page.session: current_page.update()
    except Exception as e: print(f"[{timestamp} {level}] {msg} (GUI log failed: {e})")

# (process_text - unchanged)
def process_text(text):
    global all_punctuation
    if not isinstance(text, str): return text
    punctuation_count = sum(1 for char in text if char in all_punctuation)
    if punctuation_count <= 1: return text.translate(str.maketrans('', '', all_punctuation))
    else: return text

# (transcribe_local - unchanged, still uses bytes input)
recognizer_stream = None
def transcribe_local(page: ft.Page, lv: ft.ListView, audio_input, recognizer: sherpa_onnx.OfflineRecognizer, language):
    global recognizer_stream
    if not page or not lv or not recognizer or not SHERPA_AVAILABLE: return "Recognizer unavailable."
    recognizer_stream = recognizer.create_stream()
    audio_array = None; sample_rate = RATE
    if isinstance(audio_input, bytes):
        audio_int16 = np.frombuffer(audio_input, dtype=np.int16)
        audio_array = audio_int16.astype(np.float32) / 32768.0
    elif isinstance(audio_input, np.ndarray): # Should not happen with current record_audio returning bytes
        audio_array = audio_input.astype(np.float32) / 32768.0 if audio_input.dtype != np.float32 else audio_input
    else: return "Unsupported audio input type."
    if audio_array is None: return "Audio data preparation failed."
    start_time = time.perf_counter()
    try:
        recognizer_stream.accept_waveform(sample_rate=sample_rate, waveform=audio_array)
        tail_padding = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
        recognizer_stream.accept_waveform(sample_rate=sample_rate, waveform=tail_padding)
        recognizer.decode_stream(recognizer_stream)
        transcribed_text = recognizer_stream.result.text
    except Exception as e:
         log(page, lv, f"Sherpa-ONNX transcription error: {e}", "ERROR"); traceback.print_exc()
         return f"Transcription failed: {e}"
    duration = time.perf_counter() - start_time
    log(page, lv, f"识别完成 ({duration:.2f}s): '{transcribed_text}'", "INFO")
    return transcribed_text if transcribed_text else None

# (type_text - unchanged)
def type_text(page: ft.Page, lv: ft.ListView, text_to_type):
    if not page or not lv: return
    if not text_to_type: log(page, lv, "No text to type.", "DEBUG"); return
    log(page, lv, "Typing text (ensure target window is focused)...", "INFO")
    try: keyboard.write(text_to_type); log(page, lv, f"Typed: '{text_to_type}'", "SUCCESS")
    except ImportError: log(page, lv, "Keyboard library not available.", "ERROR")
    except Exception as e: log(page, lv, f"Keyboard input failed: {e}", "ERROR"); log(page, lv, "Check permissions.", "WARNING")

# --- MODIFIED: Record Audio - Now includes blocking logic ---
def record_audio(page: ft.Page, lv: ft.ListView, stream: pyaudio.Stream, hotkey: str, hotkey_parts: list, blocked_keys_list: list, is_caps_lock: bool):
    """
    Record audio from the stream while hotkey is held.
    Blocks hotkey parts (except Caps Lock) on first detection.
    Appends successfully blocked keys to blocked_keys_list.
    """
    global is_listening
    if not page or not lv or not stream or not stream.is_active():
        log(page, lv, "Recording skipped: Invalid page, log view, or inactive stream.", "WARNING")
        return None, 0.0

    frames = deque()
    start_time = time.time()
    duration = 0.0
    recording_started = False # Flag to indicate if we actually captured any hotkey-pressed data

    while is_listening:
        try:
            data = stream.read(CHUNK, exception_on_overflow = False)

            if keyboard.is_pressed(hotkey):
                if not recording_started:
                    log(page, lv, f"Recording started (hotkey '{hotkey}' detected)...", "DEBUG")
                    start_time = time.time() # Reset start time
                    recording_started = True

                    # --- BLOCK HOTKEY PARTS ---
                    if not is_caps_lock:
                        log(page, lv, f"Attempting to block hotkey parts: {hotkey_parts}", "DEBUG")
                        for key_part in hotkey_parts:
                            try:
                                keyboard.block_key(key_part)
                                blocked_keys_list.append(key_part) # Add to list managed by caller
                                log(page, lv, f"Blocked '{key_part}'.", "DEBUG")
                            except Exception as e_block:
                                log(page, lv, f"Could not block '{key_part}': {e_block}.", "WARNING")
                    else:
                        log(page, lv, "Caps Lock hotkey: Skipping blocking.", "DEBUG")
                    # ---------------------------

                frames.append(data)
            else:
                # Hotkey not pressed
                if recording_started:
                    log(page, lv, f"Hotkey '{hotkey}' released.", "DEBUG")
                    break # Exit loop
                else:
                    pass # Discard data if key wasn't pressed yet

            # time.sleep(0.001) # Optional small sleep

        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed: log(page, lv, "Audio input overflowed!", "WARNING")
            else: log(page, lv, f"PyAudio IOError reading stream: {e}", "ERROR"); traceback.print_exc()
            break
        except Exception as e:
             log(page, lv, f"Unexpected error reading audio stream: {e}", "ERROR"); traceback.print_exc()
             break

    # --- Loop ended ---
    if not is_listening and recording_started:
        log(page, lv, "Recording loop terminated by stop signal while recording.", "DEBUG")

    if not recording_started:
        # log(page, lv, "Recording stopped. Hotkey likely released early or not pressed.", "DEBUG")
        return None, 0.0 # No valid recording started

    duration = time.time() - start_time
    log(page, lv, f"Recording finished processing. Duration: {duration:.2f}s", "DEBUG")

    # Check minimum duration AFTER loop ends
    if duration < MIN_RECORD_SECONDS and is_listening: # Warn only if stopped by key release
        log(page, lv, f"Recording too short ({duration:.2f}s < {MIN_RECORD_SECONDS}s), ignored.", "WARNING")
        return None, duration
    elif duration < MIN_RECORD_SECONDS and not is_listening:
         log(page, lv, f"Recording stopped early ({duration:.2f}s) and is short.", "DEBUG")

    audio_data_bytes = b''.join(frames)

    # --- REMOVED UNBLOCKING FROM HERE ---
    # Unblocking is now handled by the caller in the finally block

    return audio_data_bytes, duration


# (sanitize_filename_part - unchanged)
def sanitize_filename_part(text, max_len=MAX_FILENAME_TEXT_LEN):
    if not text or not isinstance(text, str): return ""
    sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]+', '', text); sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
    if len(sanitized) > max_len: sanitized = sanitized[:max_len].strip('_')
    if not sanitized or sanitized == "_": return ""
    return sanitized

# --- MODIFIED: Save Audio (requires PyAudio instance `p` again) ---
def save_audio_with_transcription(page: ft.Page, lv: ft.ListView, audio_data: bytes, transcription: str, pyaudio_instance: pyaudio.PyAudio):
    """Saves the provided audio data as a WAV file with transcription in the name."""
    if not page or not lv or not pyaudio_instance: return False
    if not audio_data: log(page, lv, "No audio data to save.", "WARNING"); return False

    filename = "unknown_recording.wav"
    try:
        try: script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_dir = os.path.abspath(".")
        recordings_dir = os.path.join(script_dir, "录音")
        os.makedirs(recordings_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_text = sanitize_filename_part(transcription)
        filename_base = f"rec_{timestamp}"
        filename = f"{filename_base}_{sanitized_text}.wav" if sanitized_text else f"{filename_base}.wav"
        filepath = os.path.join(recordings_dir, filename)

        log(page, lv, f"Attempting to save audio to: {filename}", "DEBUG")
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        log(page, lv, f"Recording successfully saved: {filename}", "INFO")
        return True
    except IOError as e_io: log(page, lv, f"Failed to save WAV '{filename}': {e_io}", "ERROR")
    except Exception as e: log(page, lv, f"Error saving audio '{filename}': {e}", "ERROR"); traceback.print_exc()
    return False

# --- MODIFIED: Record & Transcribe Cycle (Handles Blocking/Unblocking/Caps Lock) ---
def perform_record_and_transcribe(page: ft.Page, lv: ft.ListView, stream: pyaudio.Stream, pyaudio_instance: pyaudio.PyAudio, current_model):
    """Handles one cycle: Waits for hotkey, records, blocks, transcribes, types, unblocks."""
    global is_listening, current_hotkey, current_selected_language_value
    if not page or not lv: return

    audio_data = None
    transcription_result = None
    duration = 0.0
    blocked_keys = [] # Keys successfully blocked by record_audio
    # Determine hotkey parts and if it's Caps Lock
    hotkey_parts = [part.strip() for part in current_hotkey.split('+')]
    is_caps_lock_hotkey = (current_hotkey.lower() == 'caps lock')

    try:
        log(page, lv, f"Ready. Press and hold '{current_hotkey}' to record.", "INFO")

        # --- MODIFIED: Call record_audio with blocking info ---
        # record_audio will now populate `blocked_keys` if it blocks successfully (and not caps lock)
        audio_data, duration = record_audio(page, lv, stream, current_hotkey, hotkey_parts, blocked_keys, is_caps_lock_hotkey)

        # Process if data recorded and still listening
        if audio_data and is_listening:
            lang = current_selected_language_value
            transcription_result = transcribe_local(page, lv, audio_data, current_model, lang)
            transcription_failed = transcription_result is None or "Transcription failed:" in str(transcription_result)

            save_audio_with_transcription(page, lv, audio_data,
                                           transcription_result if not transcription_failed else "transcription_failed",
                                           pyaudio_instance)

            if not transcription_failed and is_listening:
                processed_result = process_text(transcription_result)
                if processed_result != transcription_result: log(page, lv, f"Processed: '{processed_result}'", "DEBUG")
                type_text(page, lv, processed_result)

                # --- SPECIAL CAPS LOCK HANDLING ---
                # Toggle Caps Lock *after* typing to restore state
                if is_caps_lock_hotkey:
                    try:
                        log(page, lv, "Caps Lock hotkey: Toggling state back.", "DEBUG")
                        keyboard.press_and_release('caps lock')
                    except Exception as e_caps:
                        log(page, lv, f"Failed to toggle Caps Lock state: {e_caps}", "WARNING")
                # ---------------------------------

        elif not audio_data and duration >= MIN_RECORD_SECONDS and is_listening:
            log(page, lv, "Recording finished but no audio data bytes captured.", "WARNING")

    except KeyError as e:
        if is_listening: log(page, lv, f"KeyError during hotkey handling ('{current_hotkey}'): {e}", "ERROR")
    except Exception as e:
        if is_listening:
            log(page, lv, f"Error during record/process cycle ('{current_hotkey}'): {e}", "ERROR")
            log(page, lv, f"Exception type: {type(e)}", "DEBUG")
            traceback.print_exc()
    finally:
        # --- UNBLOCK HOTKEY PARTS ---
        # This runs regardless of errors or successful completion,
        # ensuring keys don't stay blocked.
        if blocked_keys: # Only unblock if keys were actually blocked
            log(page, lv, f"Unblocking hotkey parts: {blocked_keys}", "DEBUG")
            # Make a copy for safe iteration if needed, though pop should be fine here
            keys_to_unblock = list(blocked_keys)
            blocked_keys.clear() # Clear original list immediately
            for key_part in keys_to_unblock:
                try:
                    keyboard.unblock_key(key_part)
                    log(page, lv, f"Unblocked '{key_part}'.", "DEBUG")
                except Exception as e_unblock:
                    # Log error but continue trying to unblock others
                    log(page, lv, f"Could not unblock '{key_part}': {e_unblock}.", "WARNING")
            log(page, lv, "Finished unblocking attempts.", "DEBUG")
        # --- End Unblock ---


# --- MODIFIED: Listener Loop (Passes PyAudio stream and instance) ---
def listen_loop(page: ft.Page, lv: ft.ListView, stream: pyaudio.Stream, pyaudio_instance: pyaudio.PyAudio, current_model):
    """Listening loop thread function, uses the provided PyAudio stream."""
    global is_listening
    if not page or not lv: return

    log(page, lv, f"Listener thread started. Monitoring hotkey: '{current_hotkey}'. Stream Active: {stream.is_active()}", "INFO")
    while is_listening:
        try:
            perform_record_and_transcribe(page, lv, stream, pyaudio_instance, current_model)
            # Small sleep prevents tight loop when hotkey not pressed
            time.sleep(0.02) # Reduce CPU usage
        except Exception as e:
            if is_listening:
                log(page, lv, f"Error in listener loop: {e}", "ERROR")
                traceback.print_exc()
            break # Exit loop on error
    log(page, lv, "Listener thread finished.", "INFO")


# --- Hotkey Capture Thread ---
# (Unchanged - does not involve audio)
def capture_hotkey_thread_func(page: ft.Page, lv: ft.ListView):
    global current_hotkey, is_capturing_hotkey
    if not page or not lv: return
    log(page, lv, "Hotkey capture active. Press key/combination...", "INFO")
    new_hotkey = None; capture_successful = False
    try:
        new_hotkey = keyboard.read_hotkey(suppress=False)
        log(page, lv, f"Captured: '{new_hotkey}'", "SUCCESS"); capture_successful = True
    except ImportError: log(page, lv, "Keyboard library not found.", "ERROR")
    except Exception as e: log(page, lv, f"Hotkey capture failed: {e}", "ERROR")
    try:
        if capture_successful and new_hotkey:
            # --- ADDED: Prevent excessively long hotkeys ---
            if len(new_hotkey) > 30: # Arbitrary limit
                 log(page, lv, f"Captured hotkey '{new_hotkey[:30]}...' seems too long, ignoring.", "WARNING")
                 capture_successful = False # Treat as failed
            else:
                current_hotkey = new_hotkey
                if save_hotkey(current_hotkey): log(page, lv, f"Hotkey '{current_hotkey}' saved.", "INFO")
                else: log(page, lv, f"Failed to save hotkey '{current_hotkey}'.", "ERROR")
        # If capture failed or was ignored, current_hotkey remains unchanged
        is_capturing_hotkey = False
        update_status_display(page, lv) # Update status and button states
    except Exception as e_update:
         log(page, lv, f"Error updating UI/saving post-capture: {e_update}", "ERROR")
         is_capturing_hotkey = False
         if set_hotkey_button.current: set_hotkey_button.current.text = "Set Hotkey"; set_hotkey_button.current.icon = ft.Icons.KEYBOARD_OPTION_KEY_ROUNDED
         if status_text.current: status_text.current.value = "Status: Error post-capture"
         if page and page.session: page.update()

# --- Start Hotkey Capture ---
# (Unchanged)
def start_hotkey_capture(e: ft.ControlEvent):
    global is_capturing_hotkey, capture_thread
    page = e.page; lv = log_view.current
    if not page or not lv: return
    if is_capturing_hotkey: log(page, lv, "Already capturing.", "WARNING"); return
    if is_listening: log(page, lv, "Stop listening first.", "WARNING"); return
    is_capturing_hotkey = True
    log(page, lv, "Preparing hotkey capture...", "INFO")
    update_status_display(page, lv) # Sets status text and disables buttons
    capture_thread = threading.Thread(target=capture_hotkey_thread_func, args=(page, lv), daemon=True)
    capture_thread.start()

# --- UI State Management Helpers ---
# (Unchanged, checks PYAUDIO_AVAILABLE now)
def _update_ui_controls_state(page: ft.Page, listening: bool, capturing: bool):
    if not page: return
    base_ready = SHERPA_AVAILABLE and PYAUDIO_AVAILABLE # Check PyAudio
    model_ready = base_ready and os.path.isfile(MODEL_FILE_PATH) and os.path.isfile(TOKENS_FILE_PATH)
    can_start = not listening and not capturing and model_ready
    can_stop = listening
    can_set_hotkey = not listening and not capturing
    chips_disabled = listening or capturing
    if start_button.current: start_button.current.disabled = not can_start
    if stop_button.current: stop_button.current.disabled = not can_stop
    if set_hotkey_button.current:
        set_hotkey_button.current.disabled = not can_set_hotkey
        if not capturing:
             set_hotkey_button.current.text = "Set Hotkey"; set_hotkey_button.current.icon = ft.Icons.KEYBOARD_OPTION_KEY_ROUNDED
             set_hotkey_button.current.tooltip = f"Set hotkey (Current: '{current_hotkey}')"
        else:
             set_hotkey_button.current.text = "Capturing..."; set_hotkey_button.current.icon = ft.Icons.RADIO_BUTTON_CHECKED
             set_hotkey_button.current.tooltip = "Press desired key combination"
    if language_chip_row.current:
         for chip in language_chip_row.current.controls: chip.disabled = chips_disabled

# --- Status Display (Refined) ---
# (Unchanged, checks PYAUDIO_AVAILABLE now)
def update_status_display(page: ft.Page, lv: ft.ListView):
    global is_listening, current_hotkey, is_capturing_hotkey, SHERPA_AVAILABLE, PYAUDIO_AVAILABLE
    if not page or not status_text.current: return
    status_val = "Status: Unknown"; status_col = ft.Colors.GREY_500
    model_files_ok = os.path.isfile(MODEL_FILE_PATH) and os.path.isfile(TOKENS_FILE_PATH)
    if is_capturing_hotkey: status_val = "Status: Press desired hotkey..."; status_col = ft.Colors.BLUE_500
    elif is_listening: status_val = f"Status: Listening for '{current_hotkey}' (Sherpa/PyAudio)"; status_col = ft.Colors.GREEN_500
    else: # Stopped or Error states
        if not SHERPA_AVAILABLE: status_val = "Status: Error (Sherpa-ONNX missing)"; status_col = ft.Colors.RED_700
        elif not PYAUDIO_AVAILABLE: status_val = "Status: Error (PyAudio missing/failed)"; status_col = ft.Colors.RED_700
        elif not model_files_ok: status_val = "Status: Error (Model/Tokens missing)"; status_col = ft.Colors.RED_700
        else: status_val = "Status: Stopped (Sherpa/PyAudio Ready)"; status_col = ft.Colors.ORANGE_500
    status_text.current.value = status_val; status_text.current.color = status_col
    _update_ui_controls_state(page, is_listening, is_capturing_hotkey)
    if page.session: page.update()


# --- MODIFIED: Start Listening (Uses PyAudio, keeps stream open) ---
def start_listening(e: ft.ControlEvent):
    global is_listening, listener_thread, model, p, audio_stream # Use global p and audio_stream
    page = e.page; lv = log_view.current

    # Pre-checks
    if not page or not lv: print("[ERROR] Log view or Page not ready."); return
    if not SHERPA_AVAILABLE: log(page, lv, f"Sherpa-ONNX unavailable.", "ERROR"); update_status_display(page, lv); return
    if not PYAUDIO_AVAILABLE: log(page, lv, "PyAudio unavailable.", "ERROR"); update_status_display(page, lv); return
    if is_listening: log(page, lv, "Listener already running.", "WARNING"); return
    if is_capturing_hotkey: log(page, lv, "Finish hotkey capture first.", "WARNING"); return

    # --- ADDED: Check Keyboard Root/Admin ---
    if sys.platform == 'win32':
        # On Windows, admin might be needed for low-level hooks
        # This is hard to check programmatically reliably without extra libs or UAC prompts.
        # We'll rely on potential errors during blocking/unblocking later.
        pass # Assume ok for now
    else: # Linux/macOS
        try:
            if os.geteuid() != 0:
                 log(page, lv, "WARNING: Not running as root/sudo.", "WARNING")
                 log(page, lv, "Hotkey blocking/unblocking might fail without root permissions.", "WARNING")
        except AttributeError:
            log(page, lv, "Could not check root status (non-Unix?). Permissions might be needed.", "WARNING")
        except Exception as e_perm:
            log(page, lv, f"Error checking permissions: {e_perm}", "WARNING")
    # ----------------------------------------

    # Disable Start/Set Hotkey buttons immediately
    start_disabled = False; hotkey_disabled = False
    if start_button.current: start_button.current.disabled = True; start_disabled = True
    if set_hotkey_button.current: set_hotkey_button.current.disabled = True; hotkey_disabled = True
    if start_disabled or hotkey_disabled: page.update()

    # Check model files
    if not os.path.isfile(MODEL_FILE_PATH) or not os.path.isfile(TOKENS_FILE_PATH):
        log(page, lv, f"Model/Tokens not found in {MODEL_DIR}!", "ERROR")
        update_status_display(page, lv); return # Also re-enables buttons

    # Initialize PyAudio if needed
    if p is None:
        log(page, lv, "Initializing PyAudio...", "INFO")
        try: p = pyaudio.PyAudio(); log(page, lv, "PyAudio initialized.", "DEBUG")
        except Exception as e_pa:
            log(page, lv, f"Failed to initialize PyAudio: {e_pa}", "ERROR"); traceback.print_exc()
            p = None; update_status_display(page, lv); return

    # Open PyAudio Stream if not already open
    if audio_stream is not None and audio_stream.is_active():
         log(page, lv, "Audio stream already active. Reusing.", "WARNING")
    elif audio_stream is None:
        log(page, lv, "Opening PyAudio stream...", "INFO")
        try:
            audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=CHUNK,
                                  stream_callback=None, start=False)
            audio_stream.start_stream() # Start the stream flow
            log(page, lv, f"PyAudio stream opened and started. Active: {audio_stream.is_active()}", "DEBUG")
        except OSError as e_os:
             log(page, lv, f"PyAudio OSError opening stream: {e_os}", "ERROR")
             log(page, lv, "Check if audio device is available/correct, or used by another app.", "ERROR")
             traceback.print_exc(); audio_stream = None; update_status_display(page, lv); return
        except Exception as e_pa_open:
            log(page, lv, f"Failed to open PyAudio stream: {e_pa_open}", "ERROR")
            traceback.print_exc(); audio_stream = None; update_status_display(page, lv); return

    # Load Sherpa Model if needed
    if model is None:
        log(page, lv, "Loading Sherpa-ONNX recognizer...", "INFO")
        current_lang = current_selected_language_value; log(page, lv, f"Model lang: '{current_lang}'", "INFO")
        load_start = time.perf_counter()
        try:
            model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=MODEL_FILE_PATH, tokens=TOKENS_FILE_PATH, language=current_lang if current_lang != "auto" else "",
                use_itn=True, num_threads=1, debug=False, provider="cpu", sample_rate=RATE, feature_dim=80, decoding_method="greedy_search",
            )
            log(page, lv, f"Recognizer loaded ({time.perf_counter()-load_start:.2f}s)", "SUCCESS")
        except Exception as e_model:
            log(page, lv, f"Recognizer loading failed: {e_model}", "ERROR"); traceback.print_exc()
            model = None
            if audio_stream is not None:
                try:
                    log(page, lv, "Closing audio stream due to model failure.", "WARNING")
                    if audio_stream.is_active(): audio_stream.stop_stream()
                    audio_stream.close()
                except Exception as e_close: log(page, lv, f"Error closing stream: {e_close}", "WARNING")
                finally: audio_stream = None
            update_status_display(page, lv); return

    # Start listener thread
    is_listening = True
    log(page, lv, f"Starting listener thread (Hotkey: '{current_hotkey}')...", "INFO")
    listener_thread = threading.Thread(target=listen_loop, args=(page, lv, audio_stream, p, model), daemon=True)
    listener_thread.start()
    update_status_display(page, lv) # Update UI


# --- MODIFIED: Stop Listening (Closes PyAudio stream) ---
def stop_listening(e: ft.ControlEvent):
    global is_listening, listener_thread, audio_stream # Need audio_stream
    page = e.page; lv = log_view.current

    if not page or not lv: print("[ERROR] Log view or Page not ready."); return
    if not is_listening: return # Already stopped

    if stop_button.current: stop_button.current.disabled = True; page.update()

    log(page, lv, "Stopping listener thread...", "INFO")
    is_listening = False # Signal thread

    # --- IMPORTANT: Unhook keyboard NOW if thread might be stuck ---
    # If the listener thread is somehow stuck *while a key is blocked*,
    # unblocking via the thread's finally block might not happen.
    # We *could* try a global unhook here as a safety measure, but it might
    # interfere if the user immediately starts another app that uses keyboard hooks.
    # Let's rely on the finally block within the thread first.
    # Consider adding `keyboard.unhook_all()` here ONLY IF persistent blocking issues occur.

    if listener_thread is not None and listener_thread.is_alive():
        listener_thread.join(timeout=1.5)
        if listener_thread.is_alive():
            log(page, lv, "Listener thread join timed out. May need manual cleanup.", "WARNING")
            # Force unhook as a last resort?
            # try:
            #     log(page, lv, "Force unhooking keyboard due to thread timeout.", "WARNING")
            #     keyboard.unhook_all()
            # except Exception as e_unhook_force:
            #     log(page, lv, f"Error during forced unhook: {e_unhook_force}", "ERROR")
    listener_thread = None

    # --- Close the PyAudio stream ---
    if audio_stream is not None:
        log(page, lv, "Stopping and closing PyAudio stream...", "INFO")
        try:
            if audio_stream.is_active(): audio_stream.stop_stream()
            audio_stream.close()
            log(page, lv, "PyAudio stream closed.", "DEBUG")
        except Exception as e_close:
            log(page, lv, f"Error closing PyAudio stream: {e_close}", "WARNING"); traceback.print_exc()
        finally:
            audio_stream = None # Set to None regardless of close success

    update_status_display(page, lv) # Update UI


# --- Clear Logs ---
# (Unchanged)
def clear_logs(e: ft.ControlEvent):
    page = e.page; lv = log_view.current
    if lv: lv.controls.clear(); log(page, lv, "Logs cleared.", "INFO")

# --- Language Selection ---
# (Unchanged)
def language_chip_selected(e: ft.ControlEvent):
    global current_selected_language_value, is_listening
    page = e.page; lv = log_view.current; selected_chip = e.control; selected_lang = selected_chip.data
    if not page or not lv or selected_chip.disabled or current_selected_language_value == selected_lang: return
    current_selected_language_value = selected_lang
    log(page, lv, f"Language preference: '{selected_lang}'", "INFO")
    if is_listening: log(page, lv, "NOTE: Model language won't change until restart.", "WARNING")
    else: log(page, lv, "Language will be used on next start.", "DEBUG")
    if language_chip_row.current:
        changed = False
        for chip in language_chip_row.current.controls:
             is_selected = (chip.data == selected_lang)
             if chip.selected != is_selected: chip.selected = is_selected; changed = True
        if changed and page.session: page.update()

# --- MODIFIED: Initial Condition Check (Checks PyAudio) ---
def run_initial_checks_async(page: ft.Page, lv: ft.ListView):
    global current_hotkey, SHERPA_AVAILABLE, PYAUDIO_AVAILABLE
    if not page or not lv: return
    log(page, lv, "Running initial checks...", "DEBUG"); start = time.perf_counter()
    all_ok = True

    # 1. Check Sherpa
    if not SHERPA_AVAILABLE: log(page, lv, f"Sherpa-ONNX failed import", "ERROR"); all_ok = False
    else: log(page, lv, "Sherpa-ONNX check passed.", "DEBUG")

    # 2. Check PyAudio
    if not PYAUDIO_AVAILABLE: log(page, lv, "PyAudio failed import/load.", "ERROR"); all_ok = False
    else: log(page, lv, "PyAudio check passed.", "DEBUG")

    # 3. Check Model/Tokens
    model_ok = os.path.isfile(MODEL_FILE_PATH)
    tokens_ok = os.path.isfile(TOKENS_FILE_PATH)
    if not model_ok: log(page, lv, f"Model NOT FOUND: {MODEL_FILE_PATH}", "ERROR"); all_ok = False
    else: log(page, lv, "Model file found.", "DEBUG")
    if not tokens_ok: log(page, lv, f"Tokens NOT FOUND: {TOKENS_FILE_PATH}", "ERROR"); all_ok = False
    else: log(page, lv, "Tokens file found.", "DEBUG")
    if not model_ok or not tokens_ok: log(page, lv, f"Ensure files are in 'assets/{MODEL_DIR_BASE}'.", "ERROR")

    # 4. Check Keyboard
    try: is_shift = keyboard.is_pressed('shift'); log(page, lv, f"Keyboard check passed.", "DEBUG")
    except ImportError: log(page, lv, "Keyboard import failed.", "ERROR"); all_ok = False
    except PermissionError: log(page, lv, "Keyboard permission denied (try sudo/admin?).", "ERROR"); all_ok = False
    except Exception as e_kbd: log(page, lv, f"Keyboard access failed: {e_kbd}", "ERROR"); all_ok = False

    log(page, lv, f"Initial checks complete ({time.perf_counter()-start:.3f}s). Updating UI...", "DEBUG")
    update_status_display(page, lv)

# --- Copy Logs to Clipboard ---
# (Unchanged)
def copy_logs_to_clipboard(page: ft.Page, lv: ft.ListView):
    if not page or not lv: return
    if not lv.controls: log(page, lv, "No logs to copy.", "INFO"); return
    log_texts = [ctrl.value for ctrl in lv.controls if isinstance(ctrl, ft.Text)]
    page.set_clipboard("\n".join(log_texts))
    log(page, lv, "Logs copied to clipboard.", "SUCCESS")

# --- Main Application Function ---
def main(page: ft.Page):
    global model, page_ref, p, audio_stream # Allow modification by cleanup
    page_ref.current = page

    # --- Page Setup ---
    page.title = "神色语音 SenseVox"; page.window.width = 380; page.window.height = 600
    page.window.min_width = 350; page.window.min_height = 500; page.padding = ft.padding.all(15)
    page.vertical_alignment = ft.MainAxisAlignment.START; page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH

    # --- UI Element Definitions ---
    # (Mostly unchanged, tooltips updated)
    status_label = ft.Text(ref=status_text, value="Status: Initializing...", weight=ft.FontWeight.BOLD, size=14, text_align=ft.TextAlign.CENTER)
    lang_chips = [ft.Chip(label=ft.Text(lang.upper()), data=lang, selected=(lang == current_selected_language_value), tooltip=f"Select '{lang}'", on_select=language_chip_selected, disabled=True) for lang in LANG_OPTIONS]
    lang_chip_container = ft.Row(ref=language_chip_row, controls=lang_chips, alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=5, run_spacing=5)
    start_btn = ft.ElevatedButton("Start 启动", ref=start_button, icon=ft.Icons.PLAY_ARROW_ROUNDED, on_click=start_listening, expand=True, tooltip="Start listening (keeps mic open)", disabled=True)
    stop_btn = ft.ElevatedButton("Stop 暂停", ref=stop_button, icon=ft.Icons.STOP_ROUNDED, on_click=stop_listening, expand=True, tooltip="Stop listening (closes mic)", disabled=True)
    clear_btn = ft.ElevatedButton("Clear Log", icon=ft.Icons.CLEAR_ALL_ROUNDED, on_click=clear_logs, expand=True, tooltip="Clear log display")
    hotkey_btn = ft.ElevatedButton("Set Hotkey", ref=set_hotkey_button, icon=ft.Icons.KEYBOARD_OPTION_KEY_ROUNDED, on_click=start_hotkey_capture, expand=True, tooltip=f"Set hotkey (Current: '{current_hotkey}')", disabled=True)
    logs_instance = ft.ListView(ref=log_view, expand=True, spacing=5, auto_scroll=False, padding=ft.padding.only(left=10, right=5, top=5, bottom=5))
    copy_logs_btn = ft.IconButton(icon=ft.Icons.CONTENT_COPY_ROUNDED, tooltip="Copy logs", on_click=lambda e: copy_logs_to_clipboard(page, logs_instance))
    log_container = ft.Container(content=logs_instance, border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=ft.border_radius.all(5), padding=ft.padding.only(top=5, bottom=5), expand=True)

    # --- Layout ---
    # (Unchanged)
    page.add(ft.Column([
        ft.Row([status_label], alignment=ft.MainAxisAlignment.CENTER), ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
        ft.Row([start_btn, stop_btn], spacing=10), ft.Row([hotkey_btn, clear_btn], spacing=10),
        ft.Divider(height=10, color=ft.Colors.TRANSPARENT), ft.Text("语言 Language:", weight=ft.FontWeight.BOLD, size=13), lang_chip_container,
        ft.Row([ft.Text("日志 Logs:", weight=ft.FontWeight.BOLD, size=13), ft.Container(expand=True), copy_logs_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        log_container
    ], expand=True, spacing=8))

    # --- MODIFIED: Cleanup Logic for PyAudio AND Keyboard ---
    def on_window_event(e: ft.ControlEvent):
        global is_listening, listener_thread, model, is_capturing_hotkey, capture_thread, p, audio_stream # Need p and audio_stream
        if e.data == "close":
            print("\n[INFO] Window closing. Starting cleanup...")
            start = time.perf_counter()

            # 1. Stop Hotkey Capture
            is_capturing_hotkey = False # Signal thread if running
            # No join needed here, capture thread should exit quickly after flag change

            # 2. Stop Listening Thread (Signal First)
            if is_listening:
                print("[INFO] Signaling listener thread to stop...")
                is_listening = False # Signal thread

            # --- 3. Unhook Keyboard EARLY ---
            # This is crucial. If the listener thread was blocking keys,
            # we need to unblock them *before* potentially waiting for the thread.
            # Unhooking all should release any blocks.
            try:
                print("[INFO] Unhooking keyboard...")
                keyboard.unhook_all()
                print("[DEBUG] Keyboard unhooked.")
            except Exception as e_unhook:
                print(f"[WARNING] Error unhooking keyboard: {e_unhook}")

            # 4. Join Listening Thread (After signaling and unhooking)
            if listener_thread and listener_thread.is_alive():
                print("[INFO] Waiting for listener thread to exit...")
                listener_thread.join(timeout=1.0) # Shorter timeout now
                if listener_thread.is_alive():
                     print("[WARNING] Listener thread did not exit gracefully.")
            listener_thread = None

            # 5. Close PyAudio Stream
            if audio_stream is not None:
                print("[INFO] Closing PyAudio stream...")
                try:
                    if audio_stream.is_active(): audio_stream.stop_stream()
                    audio_stream.close()
                    print("[DEBUG] PyAudio stream closed.")
                except Exception as e_sd_close: print(f"[WARNING] Error closing PyAudio stream: {e_sd_close}")
                finally: audio_stream = None

            # 6. Terminate PyAudio Instance
            if p is not None:
                print("[INFO] Terminating PyAudio instance...")
                term_start = time.perf_counter()
                try: p.terminate(); print(f"[DEBUG] PyAudio terminated ({time.perf_counter()-term_start:.3f}s)")
                except Exception as e_pa_term: print(f"[WARNING] Error terminating PyAudio: {e_pa_term}")
                finally: p = None

            # 7. Release Model
            if model: print("[INFO] Releasing model reference..."); model = None

            print(f"[INFO] Cleanup finished ({time.perf_counter()-start:.3f}s).")
            page.window_destroy()

    page.window_prevent_close = True
    page.on_window_event = on_window_event

    # Initial Render and Async Checks
    page.update(); page.window.center()
    initial_check_thread = threading.Thread(target=run_initial_checks_async, args=(page, logs_instance), daemon=True)
    initial_check_thread.start()

# --- Run App ---
if __name__ == "__main__":
    # Moved root check message to start_listening for better context
    print("[INFO] Starting Flet application (PyAudio Always-On, Hotkey Blocking)...")
    ft.app(target=main, assets_dir="assets")
    print("\n[INFO] Flet application exited.")

# --- END OF MODIFIED sensevox.py ---