import pytest

from dotenv import load_dotenv

load_dotenv()

from sociaml.preprocessing import *

import tempfile
import os


@pytest.fixture
def analysis_object_with_obama_second_inauguration_video():
    return './tests/test_assets/President_Obama_takes_the_Oath_of_Office_(HD).ogv.720p.vp9.webm'
    

def test_audio_extractor(analysis_object_with_obama_second_inauguration_video):
    video_path = analysis_object_with_obama_second_inauguration_video
    audio_extractor = AudioExtractor()
    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        audio_path = f.name
        audio_extractor.process(video_path, audio_path=audio_path)
        
        assert os.path.getsize(audio_path) > 0

def test_annotation_diarization_anonymization(analysis_object_with_obama_second_inauguration_video):
    transcriber = TranscriberAndDiarizer(pyannote_api_key=os.getenv('PYANNOTE_API_KEY'),merge_consecutive_speakers=True)
    anonymizer = Anonymizer()
    
    trs = transcriber.process(analysis_object_with_obama_second_inauguration_video)
    trs = anonymizer.process(trs)
    
    assert len(trs.contributions) == 4
    assert trs.contributions[0] == Contribution(start=0.677, end=2.799, speaker='SPEAKER_01', transcript=' Please raise your right hand and repeat after me.')
    assert trs.contributions[1] == Contribution(start=2.819, end=37.599, speaker='SPEAKER_00', transcript='I, <PERSON>, do solemnly swear that I will faithfully execute the office of President of the United States.  and will, to the best of my ability, preserve, protect, and defend the Constitution of the United States. So help you God. Congratulations, Mr. President. Thank you, Mr. Chief Justice. Thank you so much. Thank you, sweetie.')
    assert trs.contributions[2] == Contribution(start=38.84, end=39.54, speaker='UNKNOWN', transcript='Thank you.')
    assert trs.contributions[3] == Contribution(start=41.76, end=45.982, speaker='SPEAKER_00', transcript='I did it. All right. Thank you, everybody.')
    
    
    
    
