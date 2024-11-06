from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import torch
import whisper
import soundfile as sf
import tempfile
import numpy as np
import glob
from tqdm import tqdm
import json
import os
import gc 
        
from .datastructures import Contribution, Transcription
from .utils import get_device

class Preprocessor:
    """
    Base class for preprocessing operations.

    This class is intended to be subclassed by specific preprocessing
    implementations.
    """

    def process(self, input):
        """
        Process the input data.

        Parameters:
        input (various): The input data to be processed.

        Returns:
        Various: The result of the preprocessing operation.
        """
        pass


class Anonymizer(Preprocessor):
    """
    Anonymizer class that extends the Preprocessor class for anonymizing sensitive information.

    Attributes:
    target_entities (list of str): A list of entity types to be anonymized.
    model_config (dict): Configuration for the NLP engine models.
    nlp_engine (TransformersNlpEngine): The NLP engine for analysis.
    analyzer (AnalyzerEngine): The engine to analyze text.
    anonymizer (AnonymizerEngine): The engine to perform anonymization.
    """

    def __init__(self, target_entities=["PERSON", "ORG", "LOC", "NORP"]):
        """
        Initialize the Anonymizer class.

        Parameters:
        target_entities (list of str): A list of entity types to be anonymized.
        """
 
        self.analyzer  = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.target_entities = target_entities
    
    def process(self, transcription: Transcription):
        """
        Anonymize sensitive information in a transcription.

        Parameters:
        transcription (Transcription): The transcription object containing the text to be anonymized.

        Returns:
        Transcription: The anonymized transcription.
        """
        for segment in transcription.contributions:
            text = segment.transcript
            
            results = self.analyzer.analyze(text=text, language='en', entities=self.target_entities)
            
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)

            segment.transcript = anonymized_result.text

        return transcription


class AudioExtractor(Preprocessor):
    """
    This class is designed to extract audio data from video files. It extends the Preprocessor class 
    and provides functionality for reading video files, extracting the audio component, and then 
    saving or returning this audio data in a usable format.

    Attributes:
    None additional to those inherited from Preprocessor.

    Methods:
    process(video_path: str, audio_path: str = None): Extracts and returns the audio data from a video file.
        - video_path: A string representing the file path to the input video.
        - audio_path: An optional string representing the file path where the extracted audio should be saved. 
                      If not provided, a temporary file will be used.

    The process method is the core function of this class. It takes a video file as input and extracts the audio track. 
    If an audio_path is provided, the audio will be saved to this location. Otherwise, it will be saved to a temporary file. 
    The method then reads the audio file and returns the audio data along with the sample rate. This data can be used 
    for further audio processing tasks.

    Dependencies:
    This class relies on the moviepy library for reading and extracting audio from video files and on the soundfile (sf) 
    library for reading audio files into an array format.

    Example:
        # Creating an instance of AudioExtractor
        extractor = AudioExtractor()

        # Extracting audio from a video file and getting the audio data and samplerate
        audio_data, samplerate = extractor.process("path/to/video.mp4")

        # Using a custom path for the extracted audio file
        audio_data, samplerate = extractor.process("path/to/video.mp4", "path/to/save/audio.mp3")
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Additional initialization can be added here if needed
        
    def process(self, video_path: str, audio_path: str = None):
        video = VideoFileClip(video_path)

        if audio_path is None:
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "audio.mp3")
                video.audio.write_audiofile(temp_file, verbose=False, logger=None)
                # now we read the audio file with librosa
                audio_data, samplerate = sf.read(temp_file)
                return audio_data, samplerate
        else:
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio_data, samplerate = sf.read(audio_path)
            return audio_data, samplerate
    
    
    
    


class TranscriberAndDiarizer(Preprocessor):
    """
    Transcriber and Diarizer class for processing audio and video files.

    Attributes:
    merge_consecutive_speakers (bool): Flag to merge consecutive speakers.
    device (torch.device): Device to run the models on.
    pipeline (Pipeline): Pyannote audio pipeline for speaker diarization.
    """

    def __init__(self, pyannote_api_key=None, merge_consecutive_speakers=True, whisper_size="turbo", device=get_device()):
        """
        Initialize the TranscriberAndDiarizer class.

        Parameters:
        merge_consecutive_speakers (bool): Flag to merge consecutive speakers.
        device (torch.device): Device to run the models on.
        pyannote_api_key (str, optional): API key for Pyannote pipeline.
        """
        self.merge_consecutive_speakers = merge_consecutive_speakers
        self.whisper_module = whisper.load_model(self.whisper_size)
        self.device = device
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=pyannote_api_key if pyannote_api_key else os.getenv("HF_TOKEN")
        )
        
        self.pipeline.to(self.device)
        

    def __transcribe_segments(self, audio_data_segments, samplerate):
        """
        Transcribe audio segments.

        Parameters:
        audio_data_segments (list): A list of tuples containing audio segment information.
        samplerate (int): The samplerate of the audio.

        Returns:
        Transcription: The transcription of the audio segments.
        """
        ts = Transcription()

        for start, end, speaker, speaker_audio in audio_data_segments:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                sf.write(temp_file, speaker_audio, samplerate)
                temp_file.seek(0)
                result = self.whisper_module.transcribe(temp_file.name)

                ts.contributions.append(Contribution(
                    start=start,
                    end=end,
                    speaker=speaker,
                    transcription=result["text"]
                ))

        return ts
            
    def process(self, video_path: str):
        """
        Process a video file for transcription and diarization.

        Parameters:
        video_path (str): Path to the video file.

        Returns:
        Transcription: The transcription of the audio in the video.
        """
        video = VideoFileClip(video_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "audio.mp3")
            video.audio.write_audiofile(temp_file, verbose=False, logger=None)
            diarization = self.pipeline(temp_file)
            audio_data, samplerate = sf.read(temp_file)
            transcription = Transcription()

            last_speaker = None
            last_end = None

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end

                # Check if this speaker is the same as the last speaker and the option is set
                if self.merge_consecutive_speakers and speaker == last_speaker:
                    # Update the end time of the last contribution
                    transcription.contributions[-1].end = end
                else:
                    # Process a new speaker turn
                    speaker_audio = audio_data[int(start * samplerate):int(end * samplerate)]
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                        sf.write(temp_file, speaker_audio, samplerate)
                        temp_file.seek(0)
                        result = whisper.load_model("medium").transcribe(temp_file.name)

                        transcription.contributions.append(Contribution(
                            start=start,
                            end=end,
                            speaker=speaker,
                            transcript=result["text"]
                        ))

                # Update the last speaker and end time
                last_speaker = speaker
                last_end = end

            return transcription