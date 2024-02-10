from enum import Enum

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Optional, List, Dict

@dataclass_json
@dataclass
class Contribution:
    """
    Represents a single speaker's continuous contribution to a transcription.

    Attributes:
        start (float): The start time of the contribution.
        end (float): The end time of the contribution.
        speaker (str): The identifier of the speaker.
        transcript (Optional[str]): The transcribed text of the contribution.
    """
    start: float
    end: float
    speaker: str
    transcript: Optional[str] = None


@dataclass_json
@dataclass
class Transcription:
    """
    Represents a transcription e.g., of a video composed of a list of individual contributions by possibly different speakers.

    Attributes:
        contributions (List[Contribution]): A list of Contribution instances.
    """
    contributions: List[Contribution] = field(default_factory=list)


@dataclass_json
@dataclass
class AnalysisObject:
    """
    Represents an object for storing various analysis data.

    """
    participants: Optional[list] = None
    global_data: Optional[dict] = None
    participant_data: Optional[dict] = None
    contribution_data: Optional[list] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    analyses_done: list[str] = field(default_factory=list)
    
    
class AnalysisMode(Enum):
    ENTIRE = 1
    AVERAGE_PER_PARTICIPANT = 2
    AVERAGE_PER_CONTRIBUTION = 3