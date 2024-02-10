from .datastructures import Transcription, AnalysisObject


class Analysis:
    """
    Represents an analysis process composed of multiple analyzers.

    Analyzers are dynamically passed as arguments and stored for processing contributions.
    """

    def __init__(self, *args):
        """
        Initializes the Analysis with a variable number of analyzers.

        Parameters:
            *args: Variable length argument list representing different analyzers.
        """
        self.analyzers = args
    
    def analyze(self, transcription: Transcription, video_path: str, audio_path: str):

        participants = []

        for contribution in transcription.contributions:
            if contribution.speaker not in participants:
                participants.append(contribution.speaker)
                
        participant_data : dict = {}
        for participant in participants:
            participant_data[participant] = {}
            
        global_data : dict = {
            "transcript": ""
        }
        
        for contribution in transcription.contributions:
            if contribution.transcript:
                global_data["transcript"] += contribution.transcript + " "
        
        
        
        ao : AnalysisObject = AnalysisObject(participants, global_data, participant_data, transcription.to_dict()['contributions'], audio_path, video_path)
        
        for analyzer in self.analyzers:
            # check if the analyzer class is not already in the list of analyses done
            if analyzer.__class__.__name__ not in ao.analyses_done:
                # if not, analyze the data
                analyzer.analyze(ao)
                
        return ao.global_data, ao.participant_data, ao.contribution_data

class Analyzer:
    """
    Base class for different types of analyzers.

    This class should be subclassed by specific analysis implementations.
    """

    def analyze(self, analysis_object):
        """
        Analyzes the given analysis object.

        This method is intended to be overridden in subclasses.

        Parameters:
            analysis_object (AnalysisObject): The analysis object to be analyzed.
        """
        pass


class ParticipantAnalyzer(Analyzer):
    """
    Analyzer subclass focusing on participant-specific analysis.
    """
    pass


class GlobalAnalyzer(Analyzer):
    """
    Analyzer subclass focusing on global-level analysis.
    """
    pass


class ContributionAnalyzer(Analyzer):
    """
    Analyzer subclass focusing on individual contributions.
    """
    pass