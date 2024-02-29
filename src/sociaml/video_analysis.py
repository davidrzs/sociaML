from .analysis import GlobalAnalyzer, ParticipantAnalyzer, ContributionAnalyzer
from .text_analysis import ParticipantContributionCount, GlobalContributionCount, ParticipantContributionCount
from .datastructures import AnalysisObject

import moviepy.editor as mp
from tempfile import NamedTemporaryFile
from feat import Detector
import pandas as pd
import torch

class ContributionPyFeatVideoFeatureAnalyzer(ContributionAnalyzer):
    
    def __init__(self, skip_frames=None) -> None:
        self.skip_frames = skip_frames
        
    def analyze(self, ao):
        contributions = ao.contribution_data
        
        clip = mp.VideoFileClip(ao.video_path)
        detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

        for i, contribution in enumerate(contributions):
            
            sub_clip = clip.subclip(contribution['start'], contribution['end'])
            # write subclip to a temporary file
            with NamedTemporaryFile(suffix=".mp4") as temp_file:
                sub_clip.write_videofile(temp_file.name)
                video_prediction = detector.detect_video(temp_file.name, skip_frames=self.skip_frames)
                video_prediction = video_prediction.drop(columns=['input', 'frame', 'approx_time'])
                print(video_prediction.columns)
                video_prediction = video_prediction.mean(skipna=True).to_dict()
                contribution['pyfeat_video_features'] = video_prediction
                print(video_prediction)

        ao.analyses_done.append(self.__class__.__name__)



class ParticipantPyFeatVideoFeatureAnalyzer(ContributionAnalyzer):
    
    def __init__(self, skip_frames=None) -> None:
        self.skip_frames = skip_frames
    
    def analyze(self, ao):
        
        if ContributionPyFeatVideoFeatureAnalyzer.__name__ not in ao.analyses_done:
            ContributionPyFeatVideoFeatureAnalyzer(skip_frames=self.skip_frames).analyze(ao)

        for participant in ao.participants:
            
            list_of_video_features = []

            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    list_of_video_features.append(contribution['pyfeat_video_features'])
                    
            # average over all contributions of a participant
            ao.participant_data[participant]['pyfeat_video_features'] = pd.DataFrame(list_of_video_features).mean(skipna=True).to_dict()

        ao.analyses_done.append(self.__class__.__name__)
        
        
        
class GlobalPyFeatVideoFeatureAnalyzer(GlobalAnalyzer):
    
    def __init__(self, skip_frames=None) -> None:
        self.skip_frames = skip_frames
        
        
    def analyze(self, ao):
        detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu')


        video_prediction = detector.detect_video(ao.video_path, skip_frames=self.skip_frames)
        video_prediction = video_prediction.drop(columns=['input', 'frame', 'approx_time'])
        ao.global_data['pyfeat_video_features_mean'] = video_prediction.mean(skipna=True).to_dict()
        ao.global_data['pyfeat_video_features_std'] = video_prediction.std(skipna=True).to_dict()

        ao.analyses_done.append(self.__class__.__name__)
