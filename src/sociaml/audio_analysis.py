from .analysis import GlobalAnalyzer, ParticipantAnalyzer, ContributionAnalyzer
from .text_analysis import ParticipantContributionCount, GlobalContributionCount, ParticipantContributionCount
from .datastructures import AnalysisObject

import os
import urllib.request
import zipfile
import audonnx
import audeer
import numpy as np
import librosa

        
class ContributionAudioEmotionAnalyzer(ContributionAnalyzer):
    # https://github.com/audeering/w2v2-how-to/blob/main/notebook.ipynb
    def __init__(self):
        model_root = 'model'
        cache_root = 'cache'


        audeer.mkdir(cache_root)
        def cache_path(file):
            return os.path.join(cache_root, file)


        url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
        dst_path = cache_path('model.zip')

        if not os.path.exists(dst_path):
            audeer.download_url(
                url, 
                dst_path, 
                verbose=True,
            )
            
        if not os.path.exists(model_root):
            audeer.extract_archive(
                dst_path, 
                model_root, 
                verbose=True,
            )
        self.model = audonnx.load(model_root)




    def analyze(self, ao : AnalysisObject):
        sampling_rate = 16000
        signal = np.random.normal(size=sampling_rate).astype(np.float32)
        self.model(signal, sampling_rate)        
    
        for contribution in ao.contribution_data:
            relevant_audio = ao.audio[int(contribution['start']*ao.samplingrate):int(contribution['end']*ao.samplingrate)]
            relevant_audio_resampled = librosa.resample(relevant_audio, orig_sr=ao.samplingrate, target_sr=16000)
            clf = self.model(relevant_audio_resampled, 16000)['logits']
            contribution['audio_emotion'] = {}
            contribution['audio_emotion']['arousal'] = clf[0][0]
            contribution['audio_emotion']['dominance'] = clf[0][1]
            contribution['audio_emotion']['valence'] =  clf[0][2]
            
        
        ao.analyses_done.append(self.__class__.__name__)
            
        

   
        
class ParticipantAudioEmotionAnalyzer(ParticipantAnalyzer):   
    def analyze(self, ao):
        
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionAudioEmotionAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioEmotionAnalyzer().analyze(ao)
        
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
            
        for participant in ao.participants:
            ao.participant_data[participant]['audio_emotion'] = {}
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    for emotion in contribution['audio_emotion']:
                        if emotion not in ao.participant_data[participant]['audio_emotion']:
                            ao.participant_data[participant]['audio_emotion'][emotion] = 0
                        ao.participant_data[participant]['audio_emotion'][emotion] += contribution['audio_emotion'][emotion]

        # now we divide by the contribution count
        for participant in ao.participants:
            for emotion in ao.participant_data[participant]['audio_emotion']:
                ao.participant_data[participant]['audio_emotion'][emotion] /= ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)



class GlobalAudioEmotionAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        
        if ContributionAudioEmotionAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioEmotionAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)
            
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
            
        ao.global_data['audio_emotion'] = {}
        
        print(ao.contribution_data)
        for contribution in ao.contribution_data:
            print(contribution)
            for emotion in contribution['audio_emotion']:
                if emotion not in ao.global_data['audio_emotion']:
                        ao.global_data['audio_emotion'][emotion] = 0
                ao.global_data['audio_emotion'][emotion] += contribution['audio_emotion'][emotion]
                    
        # now we divide by the contribution count
        for emotion in ao.global_data['audio_emotion']:
            ao.global_data['audio_emotion'][emotion] /= ao.global_data['contribution_count']
            
        ao.analyses_done.append(self.__class__.__name__)





class ContributionAudioSpeakingTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao : AnalysisObject):
    
        for contribution in ao.contribution_data:
            contribution['speaking_time'] = contribution['start'] - contribution['end']
        
        ao.analyses_done.append(self.__class__.__name__)
            
        
class ParticipantAudioSpeakingTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao):
        
        if ContributionAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSpeakingTimeAnalyzer().analyze(ao)        
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)    
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
        
        for participant in ao.participants:
            ao.participant_data[participant]['speaking_time'] = 0
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                        ao.participant_data[participant]['speaking_time_sum'] += contribution['speaking_time_sum']
                        
        if self.reduction == "mean":
            for participant in ao.participants:
                ao.participant_data[participant]['speaking_time_average'] = ao.participant_data[participant]['speaking_time_sum'] /  ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)



class GlobalAudioSpeakingTimeAnalyzer(GlobalAnalyzer):

    def analyze(self, ao):
            
        if ContributionAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSpeakingTimeAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)            
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
                        
        ao.global_data['speaking_time_sum'] = 0
        
        for participant in ao.participants:
            ao.global_data['speaking_time_sum'] += ao.participant_data[participant]['speaking_time_sum']

        ao.global_data['speaking_time_average_over_contributions'] = ao.global_data['speaking_time_sum']  / ao.global_data['contribution_count']
        ao.global_data['speaking_time_average_over_participants'] = ao.global_data['speaking_time_sum']  / len(ao.participants)

        ao.analyses_done.append(self.__class__.__name__)
        
        
class ContributionMFCCAnalyzer(ContributionAnalyzer):
    def analyze(self, ao : AnalysisObject):
    
        for contribution in ao.contribution_data:
            relevant_audio = ao.audio[int(contribution['start']*ao.samplingrate):int(contribution['end']*ao.samplingrate)]
            mfccs = librosa.feature.mfcc(y=relevant_audio, sr=ao.samplingrate)
            contribution['mfcc'] = np.mean(mfccs, axis=1)
        
        ao.analyses_done.append(self.__class__.__name__)
            
class ParticipantMFCCAnalyzer(ParticipantAnalyzer):
    def analyze(self, ao):
        if ContributionMFCCAnalyzer.__name__ not in ao.analyses_done:
            ContributionMFCCAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)    

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
        
        for participant in ao.participants:
            ao.participant_data[participant]['mfcc'] = np.zeros_like(ao.contribution_data[0]['mfcc'])

            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    ao.participant_data[participant]['mfcc'] += contribution['mfcc']

                ao.participant_data[participant]['mfcc'] /= ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)


class GlobalMFCCAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        if ContributionMFCCAnalyzer.__name__ not in ao.analyses_done:
            ContributionMFCCAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)    
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        ao.global_data['mfcc'] = np.zeros_like(ao.contribution_data[0]['mfcc'])
        count = len(ao.contribution_data)

        for contribution in ao.contribution_data:
            ao.global_data['mfcc'] += contribution['mfcc']


        ao.global_data['mfcc'] /= ao.global_data['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)
        



class ContributionAudioSilenceTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao):
        contributions = ao.contribution_data
        for i, contribution in enumerate(contributions):
            silent_before = contributions[i]['start'] - contributions[i-1]['end'] if i > 0 else 0
            silent_after = contributions[i+1]['start'] - contributions[i]['end'] if i < len(contributions) - 1 else 0

            contribution['silent_time_before'] = silent_before
            contribution['silent_time_after'] = silent_after
            contribution['silent_time'] = (silent_before + silent_after) / 2 if silent_before and silent_after else 0

        ao.analyses_done.append(self.__class__.__name__)



class ParticipantAudioSilenceTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao):
        
        
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)    
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
        
        if ContributionAudioSilenceTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSilenceTimeAnalyzer().analyze(ao)

        for participant in ao.participants:
            ao.participant_data[participant]['silent_time_before_sum'] = 0
            ao.participant_data[participant]['silent_time_after_sum'] = 0
            ao.participant_data[participant]['silent_time_sum'] = 0

            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    ao.participant_data[participant]['silent_time_before_sum'] += contribution['silent_time_before']
                    ao.participant_data[participant]['silent_time_after_sum'] += contribution['silent_time_after']
                    ao.participant_data[participant]['silent_time_sum'] += contribution['silent_time']

            ao.participant_data[participant]['silent_time_before_avg'] = ao.participant_data[participant]['silent_time_before_sum'] / ao.participant_data[participant]['contribution_count']
            ao.participant_data[participant]['silent_time_after_avg'] = ao.participant_data[participant]['silent_time_after_sum'] / ao.participant_data[participant]['contribution_count']
            ao.participant_data[participant]['silent_time_avg'] = ao.participant_data[participant]['silent_time_sum'] / ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)
        
        
        
class GlobalAudioSilenceTimeAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        if ContributionAudioSilenceTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSilenceTimeAnalyzer().analyze(ao)

        ao.global_data['silent_time_before_sum'] = 0
        ao.global_data['silent_time_after_sum'] = 0
        ao.global_data['silent_time_avg_sum'] = 0

        for participant in ao.participants:
            ao.global_data['silent_time_before_sum'] += ao.participant_data[participant]['silent_time_before_sum']
            ao.global_data['silent_time_after_sum'] += ao.participant_data[participant]['silent_time_after_sum']
            ao.global_data['silent_time_avg_sum'] += ao.participant_data[participant]['silent_time_sum']

        contribution_count = len(ao.contribution_data)
        participant_count = len(ao.participants)
        
        ao.global_data['silent_time_before_avg_over_contributions'] = ao.global_data['silent_time_before_sum'] / contribution_count
        ao.global_data['silent_time_after_avg_over_contributions'] = ao.global_data['silent_time_after_sum'] / contribution_count
        ao.global_data['silent_time_avg_avg_over_contributions'] = ao.global_data['silent_time_avg_sum'] / contribution_count

        ao.global_data['silent_time_before_avg_over_participants'] = ao.global_data['silent_time_before_sum'] / participant_count
        ao.global_data['silent_time_after_avg_over_participants'] = ao.global_data['silent_time_after_sum'] / participant_count
        ao.global_data['silent_time_avg_avg_over_participants'] = ao.global_data['silent_time_avg_sum'] / participant_count

        ao.analyses_done.append(self.__class__.__name__)