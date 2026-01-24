"""ASR Hallucination Detection

A package for transcribing audio with multiple ASR systems and analyzing
hallucinations using time-aligned word-level transcripts.
"""

__version__ = "0.1.0"

from . import transcription
from . import audio
from . import alignment
from . import ami_utils

__all__ = ["transcription", "audio", "alignment", "ami_utils"]
