from sociaml.analysis import Analysis
from sociaml.datastructures import Transcription, AnalysisMode

from sociaml.text_analysis import GlobalSentimentAnalyzer, GlobalEkmanEmotionAnalyzer, ParticipantEkmanEmotionAnalyzer, ParticipantSentimentAnalyzer, ParticipantNLTKTokenCountAnalyzer, ParticipantContributionCount, ParticipantSentenceEmbeddingAnalyzer
from sociaml.audio_analysis import ParticipantAudioSpeakingTimeAnalyzer, ParticipantMFCCAnalyzer, ParticipantAudioSilenceTimeAnalyzer, GlobalAudioEmotionAnalyzer,ParticipantAudioEmotionAnalyzer
from sociaml.video_analysis import ParticipantPyFeatVideoFeatureAnalyzer, GlobalPyFeatVideoFeatureAnalyzer


from sociaml.utils import features_to_df

from pathlib import Path
import glob
import tqdm


import json


pitch_folder = '/home/david/data/entrepreneurship_videos/pitches/'
audio_folder = '/home/david/data/entrepreneurship_videos/audios'
json_folder = '/home/david/data/entrepreneurship_videos/transcriptions'




video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
files = []

for extension in video_extensions:
    files.extend(glob.glob(pitch_folder + extension))


print(files)

global_data_dict = {}
participant_data_dict = {}
contribution_data_dict = {}


import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



for video_file in (progress_bar := tqdm.tqdm(files)):
        
    
    filename = Path(video_file).stem
    audio_file = str(Path(audio_folder) / (filename + '.mp3'))
    transcription_file = str(Path(json_folder) / (filename + '.json'))

    if not Path(video_file).exists() or not Path(audio_file).exists():
        print(f"Skipping {filename} because video or audio file does not exist")
        print(f"Video: {video_file}")
        print(f"Audio: {audio_file}")
        continue

    progress_bar.set_description(f"Processing {filename}")


    with open(transcription_file, 'r') as f:
        transcription = json.loads(json.load(f))
        transcription = Transcription.from_dict(transcription)


    analysis = Analysis(
                        
                        # textual features
                        GlobalSentimentAnalyzer(mode=AnalysisMode.ENTIRE), 
                        GlobalEkmanEmotionAnalyzer(mode=AnalysisMode.ENTIRE),
                        ParticipantEkmanEmotionAnalyzer(),
                        ParticipantSentimentAnalyzer(),
                        ParticipantNLTKTokenCountAnalyzer(),
                        ParticipantContributionCount(),
                        ParticipantSentenceEmbeddingAnalyzer(),
                        
                        
                        # audio features
                        GlobalAudioEmotionAnalyzer(mode=AnalysisMode.ENTIRE),
                        ParticipantAudioEmotionAnalyzer(),
                        ParticipantAudioSpeakingTimeAnalyzer(),
                        ParticipantMFCCAnalyzer(),
                        ParticipantAudioSilenceTimeAnalyzer(),
                        
                        
                        # visual features
                        # ParticipantPyFeatVideoFeatureAnalyzer(),
                        GlobalPyFeatVideoFeatureAnalyzer(skip_frames=500)
                        
                )

    global_data, participant_data, contribution_data = analysis.analyze(transcription, str(video_file), str(audio_file))



    # print("------")
    # print("GLOBAL LEVEL")
    # print(global_data)
    # print("------")
    # print("PARTICIPANT LEVEL")
    # print(participant_data)
    # print("------")
    # print("CONTRIBUTION LEVEL")
    # print(contribution_data)
    global_data_dict[filename] = global_data
    participant_data_dict[filename] = participant_data
    contribution_data_dict[filename] = contribution_data
    
    
    global_df = features_to_df(global_data_dict, type="global")
    participant_df = features_to_df(participant_data_dict, type="participant")
    contribution_df = features_to_df(contribution_data_dict, type="contribution")
    
    # store them
    global_df.to_csv(f'global.csv')
    participant_df.to_csv(f'participant.csv')
    contribution_df.to_csv(f'contribution.csv')

    
    
    # save
    with open('global_data.json', 'w') as f:
        json.dump(global_data_dict, f, cls=NumpyEncoder)
        
    with open('participant_data.json', 'w') as f:
        json.dump(participant_data_dict, f, cls=NumpyEncoder)
        
    with open('contribution_data.json', 'w') as f:
        json.dump(contribution_data_dict, f, cls=NumpyEncoder)
