import random
from enum import Enum, EnumMeta
from typing import Dict, List, Optional
from warnings import warn

from flet.utils import deprecated


class IconsDeprecated(EnumMeta):
    def __getattribute__(self, item):
        if not item.startswith("_") and item.isupper():
            warn(
                "icons enum is deprecated since version 0.25.0 and will be removed in version 0.28.0. "
                "Use Icons enum instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return EnumMeta.__getattribute__(self, item)


class icons(str, Enum, metaclass=IconsDeprecated):
    @staticmethod
    def random():
        return random.choice(list(icons))

    @staticmethod
    @deprecated(
        reason="Use icons.random() method instead.",
        version="0.25.0",
        delete_version="0.28.0",
    )
    def random_icon():
        return random.choice(list(icons))

    PLAY_ARROW_ROUNDED = "play_arrow_rounded"
    STOP_ROUNDED = "stop_rounded"
    KEYBOARD_OPTION_KEY_ROUNDED = "keyboard_option_key_rounded"
    MODEL_TRAINING_ROUNDED = "model_training_rounded"
    LANGUAGE_ROUNDED = "language_rounded"
    CLEAR_ALL_ROUNDED = "clear_all_rounded"
    CONTENT_COPY_ROUNDED = "content_copy_rounded"
    RADIO_BUTTON_CHECKED = "radio_button_checked"

class Icons(str, Enum):
    @staticmethod
    def random(
        exclude: Optional[List["Icons"]] = None,
        weights: Optional[Dict["Icons", int]] = None,
    ) -> Optional["Icons"]:

        choices = list(Icons)
        if exclude:
            choices = [member for member in choices if member not in exclude]
            if not choices:
                return None
        if weights:
            weights_list = [weights.get(c, 1) for c in choices]
            return random.choices(choices, weights=weights_list)[0]
        return random.choice(choices)
    PLAY_ARROW_ROUNDED = "play_arrow_rounded"
    STOP_ROUNDED = "stop_rounded"
    KEYBOARD_OPTION_KEY_ROUNDED = "keyboard_option_key_rounded"
    MODEL_TRAINING_ROUNDED = "model_training_rounded"
    LANGUAGE_ROUNDED = "language_rounded"
    CLEAR_ALL_ROUNDED = "clear_all_rounded"
    CONTENT_COPY_ROUNDED = "content_copy_rounded"
    RADIO_BUTTON_CHECKED = "radio_button_checked"
