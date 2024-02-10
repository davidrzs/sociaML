import pytest
from sociaml.datastructures import Contribution, Transcription, AnalysisObject

# Tests for Contribution class
def test_contribution_initialization():
    contribution = Contribution(start=0.0, end=10.0, speaker="Speaker1", transcript="Hello world")
    assert contribution.start == 0.0
    assert contribution.end == 10.0
    assert contribution.speaker == "Speaker1"
    assert contribution.transcript == "Hello world"

def test_contribution_optional_transcript():
    contribution = Contribution(start=0.0, end=10.0, speaker="Speaker1")
    assert contribution.transcript is None

# Tests for Transcription class
def test_transcription_initialization():
    transcription = Transcription()
    assert isinstance(transcription.contributions, list)
    assert len(transcription.contributions) == 0

def test_transcription_add_contribution():
    transcription = Transcription()
    contribution = Contribution(start=0.0, end=10.0, speaker="Speaker1")
    transcription.contributions.append(contribution)
    assert len(transcription.contributions) == 1
    assert transcription.contributions[0] == contribution

# Tests for AnalysisObject class
def test_analysis_object_initialization():
    analysis_obj = AnalysisObject()
    assert analysis_obj.participants is None
    assert analysis_obj.global_data is None
    assert analysis_obj.participant_data is None
    assert analysis_obj.contribution_data is None
    assert analysis_obj.audio_path is None
    assert analysis_obj.video_path is None
    assert isinstance(analysis_obj.analyses_done, list)
    assert len(analysis_obj.analyses_done) == 0