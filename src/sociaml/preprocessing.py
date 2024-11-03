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
    """

    def __init__(self, pyannote_api_key, merge_consecutive_speakers=True, min_segment_duration=0.2, device=get_device()):
        """
        Initialize the TranscriberAndDiarizer class.

        Parameters:
        pyannote_api_key (str): Huggingface API key for Pyannote pipeline
        merge_consecutive_speakers (bool): Flag to merge consecutive speakers
        min_segment_duration (float): Minimum duration in seconds for a valid segment
        device (torch.device): Device to run the models on
        """
        self.merge_consecutive_speakers = merge_consecutive_speakers
        self.min_segment_duration = min_segment_duration
        self.device = device
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=pyannote_api_key
        )
        
        self.pipeline.to(self.device)
        self.whisper_model = whisper.load_model("medium")

    def _process_audio_segment(self, audio_data, start_sample, end_sample, samplerate):
        """
        Process a single audio segment with proper boundary handling.
        
        Parameters:
        audio_data (np.array): Full audio data
        start_sample (int): Starting sample index
        end_sample (int): Ending sample index
        samplerate (int): Audio sample rate
        
        Returns:
        str: Transcribed text
        """
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return ""
            
        speaker_audio = audio_data[start_sample:end_sample]
        
        num_channels = speaker_audio.shape[1] if len(speaker_audio.shape) > 1 else 1
        padding_duration = int(0.1 * samplerate)
        
        if num_channels > 1:
            padding = np.zeros((padding_duration, num_channels))
            speaker_audio = np.vstack([padding, speaker_audio, padding])
        else:
            padding = np.zeros(padding_duration)
            speaker_audio = np.concatenate([padding, speaker_audio, padding])
                
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            sf.write(temp_file, speaker_audio, samplerate)
            temp_file.seek(0)
            result = self.whisper_model.transcribe(temp_file.name)
            return result["text"].strip()

    def process(self, video_path: str):
        """
        Process a video file for transcription and diarization with improved timing handling.
        
        Parameters:
        video_path (str): Path to the video file
        
        Returns:
        Transcription: The transcription of the audio in the video
        """
        video = VideoFileClip(video_path)
        transcription = Transcription()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            temp_file = os.path.join(temp_dir, "audio.wav")  # Using WAV instead of MP3 for better timing accuracy
            video.audio.write_audiofile(temp_file, verbose=False, logger=None)
            
            # Load audio data
            audio_data, samplerate = sf.read(temp_file)
            
            # Run diarization
            diarization = self.pipeline(temp_file)
            
            # Process segments
            current_segment = None
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                duration = end - start
                
                # Skip very short segments that might be noise
                if duration < self.min_segment_duration:
                    continue
                
                # Convert time to sample indices
                start_sample = int(start * samplerate)
                end_sample = int(end * samplerate)
                
                if self.merge_consecutive_speakers and current_segment and current_segment.speaker == speaker:
                    # Merge with previous segment if same speaker
                    current_segment.end = end
                    # Process the new portion of audio
                    new_text = self._process_audio_segment(
                        audio_data, 
                        int(current_segment.end * samplerate), 
                        end_sample, 
                        samplerate
                    )
                    if new_text:
                        current_segment.transcript += " " + new_text
                else:
                    # Create new segment
                    text = self._process_audio_segment(audio_data, start_sample, end_sample, samplerate)
                    if text:
                        current_segment = Contribution(
                            start=start,
                            end=end,
                            speaker=speaker,
                            transcript=text
                        )
                        transcription.contributions.append(current_segment)
            
        return transcription