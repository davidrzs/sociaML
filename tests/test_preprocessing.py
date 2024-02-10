import pytest

from dotenv import load_dotenv

load_dotenv()

from sociaml.preprocessing import *

import requests
import tempfile
import os


@pytest.fixture
def analysis_object_with_obama_second_inauguration_video():
    return './tests/test_assets/President_Obama_takes_the_Oath_of_Office_(HD).ogv.720p.vp9.webm'



# # test the diarization and transcription of the obama second inauguration video
# def test_obama_second_inauguration_video(analysis_object_with_obama_second_inauguration_video):
#     video_path = analysis_object_with_obama_second_inauguration_video
#     transcriber = TranscriberAndDiarizer(device=torch.device("cuda"),pyannote_api_key='PYANNOTE_KEY_HERE',merge_consecutive_speakers=True)
#     anonymizer = Anonymizer()
#     audio_extractor = AudioExtractor()


#     # audio_extractor.process(video_path, audio_path=audio_path)
#     trs = transcriber.process(video_path)
#     trs = anonymizer.process(trs)
    

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
    
    
    
