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
import whisperx
import gc 
        
from .datastructures import Contribution, Transcription


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

    def __init__(self, pyannote_api_key=None, merge_consecutive_speakers=True, device="cuda", batch_size: int = 16, compute_type: str = "float16"):
        """
        Initialize the TranscriberAndDiarizer class.

        Parameters:
        merge_consecutive_speakers (bool): Flag to merge consecutive speakers.
        device (torch.device): Device to run the models on.
        pyannote_api_key (str, optional): API key for Pyannote pipeline.
        batch_size (int, optional): _description_. Defaults to 16.
        compute_type (str, optional): _description_. Defaults to "float16", can also be "int8" if low on GPU mem (but reduces accuracy)
        """
        
        if device == "cpu":
            assert compute_type != "float16", "float16 is not supported on CPU, choose int8 instead"
        
        self.merge_consecutive_speakers = merge_consecutive_speakers
        self.device = device
        self.pyannote_api_key = pyannote_api_key
        self.batch_size = batch_size
        self.compute_type = compute_type
        

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
                result = whisper.load_model("medium").transcribe(temp_file.name)

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

        Args:
            video_path (str): _description_
            device (str, optional): _description_. Defaults to "cuda".
        """
        video = VideoFileClip(video_path)
        
        # code below adapted from https://github.com/m-bain/whisperX

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "audio.mp3")
            video.audio.write_audiofile(temp_file, verbose=False, logger=None)

            model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)


            audio = whisperx.load_audio(temp_file)
            result = model.transcribe(audio, batch_size=self.batch_size)

            gc.collect(); torch.cuda.empty_cache(); del model

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            gc.collect(); torch.cuda.empty_cache(); del model_a

            diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.pyannote_api_key, device=self.device)

            diarize_segments = diarize_model(audio)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            
            transcription = Transcription()

            if self.merge_consecutive_speakers:
                
                segments =  []
                previous_speaker = None
                
                for seg in result["segments"]:
                    speaker = seg.get('speaker',"UNKNOWN")
                    start = seg.get('start',"UNKNOWN")
                    end = seg.get('end',"UNKNOWN")
                    transcript = seg.get('text',"UNKNOWN")

                    # If the current speaker is the same as the previous speaker, extend the previous segment
                    if speaker == previous_speaker:
                        segments[-1] = Contribution(
                                # get the start of the previous one
                                start=segments[-1].start,
                                # as we extend we have a new end
                                end=end,
                                speaker=speaker,
                                # transcript is appended
                                transcript=segments[-1].transcript + " " + transcript
                            ) 
                    else:
                        segments.append(Contribution(
                            start=start,
                            end=end,
                            speaker=speaker,
                            transcript=transcript
                        ))

                    previous_speaker = speaker
                    
                transcription.contributions = segments
            
            else:

                for seg in result["segments"]:
                    transcription.contributions.append(Contribution(
                        start=seg.get('start',"UNKNOWN"),
                        end=seg.get('end',"UNKNOWN"),
                        speaker=seg.get('speaker',"UNKNOWN"),
                        transcript=seg.get('text',"UNKNOWN")
                    ))
                    
            
            
            
            return transcription

