from dataclasses import dataclass
import cv2
from pathlib import Path
from collections.abc import Callable

@dataclass
class SaveAttackContext:
    attacks: dict[str, cv2.typing.MatLike]
    key_path: Path
    output_dir: Path
    extract_function: Callable