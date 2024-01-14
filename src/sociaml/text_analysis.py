from .analysis import GlobalAnalyzer, ParticipantAnalyzer, ContributionAnalyzer
from .datastructures import AnalysisObject

import numpy as np


class GlobalContributionCount(GlobalAnalyzer):

    def analyze(self,  ao : AnalysisObject):
        ao.global_data['contribution_count'] = len(ao.contribution_data)
        ao.analyses_done.append(self.__class__.__name__)


class ParticipantContributionCount(ParticipantAnalyzer):
    
    def analyze(self, ao : AnalysisObject):
        for participant in ao.participants:
            ao.participant_data[participant]['contribution_count'] = 0
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    ao.participant_data[participant]['contribution_count'] += 1
        
        ao.analyses_done.append(self.__class__.__name__)

class GlobalNLTKTokenCountAnalyzer(GlobalAnalyzer):
    
    def analyze(self, ao : AnalysisObject):
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ParticipantNLTKTokenCountAnalyzer.__name__ not in ao.analyses_done:
            ParticipantNLTKTokenCountAnalyzer().analyze(ao)
        
        ao.global_data['token_count'] = 0
        for participant in ao.participants:
            ao.global_data['token_count'] += ao.participant_data[participant]['token_count'] 
        
        ao.analyses_done.append(self.__class__.__name__)
        

class ParticipantNLTKTokenCountAnalyzer(GlobalAnalyzer):

    
    def analyze(self, ao : AnalysisObject):
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionNLTKTokenCountAnalyzer.__name__ not in ao.analyses_done:
            ContributionNLTKTokenCountAnalyzer().analyze(ao)
        for participant in ao.participants:
            ao.participant_data[participant]['token_count'] = 0
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    ao.participant_data[participant]['token_count'] += contribution['token_count']
        
        
        ao.analyses_done.append(self.__class__.__name__)
    
    
class ContributionNLTKTokenCountAnalyzer(GlobalAnalyzer):
    def __init__(self):
        import nltk
    
    def analyze(self, ao : AnalysisObject):
        import nltk        
        
        for contribution in ao.contribution_data:
            token_count = len(nltk.word_tokenize(contribution['transcript']))
            # attach the classification to the contribution in the data
            contribution['token_count'] = token_count
        
        ao.analyses_done.append(self.__class__.__name__)


class GlobalTopicModelAnalyzer(GlobalAnalyzer):
    def __init__(self):
        pass
    
    def analyze(self, data):
        pass
    
    
class ParticipantTopicModelAnalyzer(ParticipantAnalyzer):
    def __init__(self):
        pass
    
    def analyze(self, data):
        pass


class ContributionEkmanEmotionAnalyzer(ContributionAnalyzer):
    # https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

    def analyze(self, ao : AnalysisObject):
        
        from transformers import pipeline

        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

        
        for contribution in ao.contribution_data:
            cls = classifier(contribution['transcript'])[0]
            # attach the classification to the contribution in the data
            emotion_dict = {item['label']: item['score'] for item in cls}
            contribution['ekmanemotion'] = emotion_dict
        
        ao.analyses_done.append(self.__class__.__name__)



class ParticipantEkmanEmotionAnalyzer(ParticipantAnalyzer):   
    def analyze(self, ao):
        
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionEkmanEmotionAnalyzer.__name__ not in ao.analyses_done:
            ContributionEkmanEmotionAnalyzer().analyze(ao)
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
            
        for participant in ao.participants:
            ao.participant_data[participant]['ekmanemotion'] = {}
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    for emotion in contribution['ekmanemotion']:
                        if emotion not in ao.participant_data[participant]['ekmanemotion']:
                            ao.participant_data[participant]['ekmanemotion'][emotion] = 0
                        ao.participant_data[participant]['ekmanemotion'][emotion] += contribution['ekmanemotion'][emotion]

        # now we divide by the contribution count
        for participant in ao.participants:
            for emotion in ao.participant_data[participant]['ekmanemotion']:
                ao.participant_data[participant]['ekmanemotion'][emotion] /= ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)



class GlobalEkmanEmotionAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionEkmanEmotionAnalyzer.__name__ not in ao.analyses_done:
            ContributionEkmanEmotionAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)
            
        ao.global_data['ekmanemotion'] = {}
        

        for contribution in ao.contribution_data:
            for emotion in contribution['ekmanemotion']:
                if emotion not in ao.global_data['ekmanemotion']:
                        ao.global_data['ekmanemotion'][emotion] = 0
                ao.global_data['ekmanemotion'][emotion] += contribution['ekmanemotion'][emotion]
                    
        # now we divide by the contribution count
        for emotion in ao.global_data['ekmanemotion']:
            ao.global_data['ekmanemotion'][emotion] /= ao.global_data['contribution_count']
            
        ao.analyses_done.append(self.__class__.__name__)


class ContributionSentimentAnalyzer(ContributionAnalyzer):
    # https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
    def analyze(self, ao : AnalysisObject):
        
        from transformers import pipeline

        classifier = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)
        processed_contributions = []
        
        for contribution in ao.contribution_data:
            cls = classifier(contribution['transcript'])[0]
            # attach the classification to the contribution in the data
            emotion_dict = {item['label']: item['score'] for item in cls}
            contribution['sentiment'] = emotion_dict
        
        ao.analyses_done.append(self.__class__.__name__)
            
        

   
        
class ParticipantSentimentAnalyzer(ParticipantAnalyzer):   
    def analyze(self, ao):
        
        if ContributionSentimentAnalyzer.__name__ not in ao.analyses_done:
            ContributionSentimentAnalyzer().analyze(ao)
            
        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)
            
        for participant in ao.participants:
            ao.participant_data[participant]['sentiment'] = {}
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    for sentiment in contribution['sentiment']:
                        if sentiment not in ao.participant_data[participant]['sentiment']:
                            ao.participant_data[participant]['sentiment'][sentiment] = 0
                        ao.participant_data[participant]['sentiment'][sentiment] += contribution['sentiment'][sentiment]

        # now we divide by the contribution count
        for participant in ao.participants:
            for sentiment in ao.participant_data[participant]['sentiment']:
                ao.participant_data[participant]['sentiment'][sentiment] /= ao.participant_data[participant]['contribution_count']

        ao.analyses_done.append(self.__class__.__name__)



class GlobalSentimentAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        
        if ContributionSentimentAnalyzer.__name__ not in ao.analyses_done:
            ContributionSentimentAnalyzer().analyze(ao)
            
        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)
            
        ao.global_data['sentiment'] = {}
        

        for contribution in ao.contribution_data:
            for sentiment in contribution['sentiment']:
                if sentiment not in ao.global_data['sentiment']:
                        ao.global_data['sentiment'][sentiment] = 0
                ao.global_data['sentiment'][sentiment] += contribution['sentiment'][sentiment]
                    
        # now we divide by the contribution count
        for sentiment in ao.global_data['sentiment']:
            ao.global_data['sentiment'][sentiment] /= ao.global_data['contribution_count']
            
        ao.analyses_done.append(self.__class__.__name__)
        
        
        
        
        
        
        
class ContributionSentenceEmbeddingAnalyzer(ContributionAnalyzer):
    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    
    def __init__(self, sentence_transformer="sentence-transformers/all-mpnet-base-v2") -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(sentence_transformer)
        
        import spacy
        self.spacy_nlp = spacy.load("en_core_web_sm")

    
    def analyze(self, ao : AnalysisObject):
        
    
        for contribution in ao.contribution_data:
            
            import numpy as np

            # we need to do this per sentence        
            spacy_doc = self.spacy_nlp(contribution['transcript'])
            
            embeddings = []
            
            for sentence in spacy_doc.sents:
                sentence_embeddings = self.model.encode(sentence.text)
                embeddings.append(sentence_embeddings)
            
            # average over the embeddings (are numpy!)
            contribution['sentence_embeddings'] = np.mean(embeddings, axis=0)
        
        ao.analyses_done.append(self.__class__.__name__)        
        


        
class ParticipantSentenceEmbeddingAnalyzer(ParticipantAnalyzer):   
    def analyze(self, ao):
        
        
        if ContributionSentenceEmbeddingAnalyzer.__name__ not in ao.analyses_done:
            ContributionSentenceEmbeddingAnalyzer().analyze(ao)
            
            
        for participant in ao.participants:
            embeddings = []
            for contribution in ao.contribution_data:
                if contribution['speaker'] == participant:
                    embeddings.append(contribution['sentence_embeddings'])
            ao.participant_data[participant]['sentence_embeddings'] = np.mean(embeddings, axis=0)
                    
        ao.analyses_done.append(self.__class__.__name__)



class GlobalSentenceEmbeddingAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        
        if ContributionSentenceEmbeddingAnalyzer.__name__ not in ao.analyses_done:
            ContributionSentenceEmbeddingAnalyzer().analyze(ao)

            
        embeddings = []
        

        for participant in ao.participants:
            embeddings = []
            for contribution in ao.contribution_data:
                    embeddings.append(contribution['sentence_embeddings'])
                    
        ao.global_data['sentence_embeddings'] = np.mean(embeddings, axis=0)
            
        ao.analyses_done.append(self.__class__.__name__)
        