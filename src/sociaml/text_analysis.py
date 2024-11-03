from .analysis import GlobalAnalyzer, ParticipantAnalyzer, ContributionAnalyzer
from .datastructures import AnalysisObject, AnalysisMode
from .utils import get_device

import numpy as np
from transformers import pipeline
import nltk
from sentence_transformers import SentenceTransformer
import spacy
import torch


class GlobalContributionCount(GlobalAnalyzer):

    def analyze(self, ao: AnalysisObject):
        ao.global_data["contribution_count"]: int = len(ao.contribution_data)
        ao.analyses_done.append(self.__class__.__name__)


class ParticipantContributionCount(ParticipantAnalyzer):

    def analyze(self, ao: AnalysisObject):
        for participant in ao.participants:
            ao.participant_data[participant]["contribution_count"] = 0
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    ao.participant_data[participant]["contribution_count"] += 1

        ao.analyses_done.append(self.__class__.__name__)


class GlobalNLTKTokenCountAnalyzer(GlobalAnalyzer):

    def analyze(self, ao: AnalysisObject):
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ParticipantNLTKTokenCountAnalyzer.__name__ not in ao.analyses_done:
            ParticipantNLTKTokenCountAnalyzer().analyze(ao)

        ao.global_data["token_count"]: int = 0
        for participant in ao.participants:
            ao.global_data["token_count"] += ao.participant_data[participant][
                "token_count"
            ]

        ao.analyses_done.append(self.__class__.__name__)


class ParticipantNLTKTokenCountAnalyzer(GlobalAnalyzer):

    def analyze(self, ao: AnalysisObject):
        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionNLTKTokenCountAnalyzer.__name__ not in ao.analyses_done:
            ContributionNLTKTokenCountAnalyzer().analyze(ao)
        for participant in ao.participants:
            ao.participant_data[participant]["token_count"]: int = 0
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    ao.participant_data[participant]["token_count"] += contribution[
                        "token_count"
                    ]

        ao.analyses_done.append(self.__class__.__name__)


class ContributionNLTKTokenCountAnalyzer(GlobalAnalyzer):
    def __init__(self):
        super().__init__()
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")

    def analyze(self, ao: AnalysisObject):

        for contribution in ao.contribution_data:
            token_count = len(nltk.word_tokenize(contribution["transcript"]))
            # attach the classification to the contribution in the data
            contribution["token_count"] = token_count

        ao.analyses_done.append(self.__class__.__name__)


class ContributionEkmanEmotionAnalyzer(ContributionAnalyzer):
    def __init__(self, model='j-hartmann/emotion-english-distilroberta-base', device=get_device()):
        self.classifier = pipeline("text-classification", model=model, return_all_scores=True, device=device)
        self.device = device



    def analyze(self, ao: AnalysisObject):


        for contribution in ao.contribution_data:
            cls = self.classifier(contribution["transcript"], padding=True, truncation=True)[
                0
            ]
            # attach the classification to the contribution in the data
            emotion_dict = {item["label"]: item["score"] for item in cls}
            contribution["ekmanemotion"] = emotion_dict

        ao.analyses_done.append(self.__class__.__name__)


class ParticipantEkmanEmotionAnalyzer(ParticipantAnalyzer):

    def __init__(self, model='j-hartmann/emotion-english-distilroberta-base', device=get_device()):
        self.model = model
        self.device = device

    def analyze(self, ao):

        # make sure the ContributionEkmanEmotionAnalyzer already ran
        if ContributionEkmanEmotionAnalyzer.__name__ not in ao.analyses_done:
            ContributionEkmanEmotionAnalyzer().analyze(ao)

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        for participant in ao.participants:
            ao.participant_data[participant]["ekmanemotion"] = {}
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    for emotion in contribution["ekmanemotion"]:
                        if (
                            emotion
                            not in ao.participant_data[participant]["ekmanemotion"]
                        ):
                            ao.participant_data[participant]["ekmanemotion"][
                                emotion
                            ] = 0
                        ao.participant_data[participant]["ekmanemotion"][
                            emotion
                        ] += contribution["ekmanemotion"][emotion]

        # now we divide by the contribution count
        for participant in ao.participants:
            for emotion in ao.participant_data[participant]["ekmanemotion"]:
                ao.participant_data[participant]["ekmanemotion"][
                    emotion
                ] /= ao.participant_data[participant]["contribution_count"]

        ao.analyses_done.append(self.__class__.__name__)


class GlobalEkmanEmotionAnalyzer(GlobalAnalyzer):

    def __init__(self, mode, model='j-hartmann/emotion-english-distilroberta-base', device=get_device()):
        self.mode = mode
        self.model = model
        self.device = device


    def analyze(self, ao: AnalysisObject):

        if self.mode == AnalysisMode.ENTIRE:
            self.classifier = pipeline("text-classification", model=self.model, return_all_scores=True, device=self.device)

            cls = self.classifier(
                ao.global_data["transcript"], padding=True, truncation=True
            )[0]
            # attach the classification to the contribution in the data
            emotion_dict = {item["label"]: item["score"] for item in cls}

            ao.global_data["ekmanemotion"] = emotion_dict

            ao.analyses_done.append(self.__class__.__name__)

        elif self.mode == AnalysisMode.AVERAGE_PER_PARTICIPANT:

            raise NotImplementedError(
                "AnalysisMode.AVERAGE_PER_PARTICIPANT not implemented yet"
            )

        else:

            assert self.mode == AnalysisMode.AVERAGE_PER_CONTRIBUTION

            if ContributionEkmanEmotionAnalyzer.__name__ not in ao.analyses_done:
                ContributionEkmanEmotionAnalyzer().analyze(ao)

            if GlobalContributionCount.__name__ not in ao.analyses_done:
                GlobalContributionCount().analyze(ao)

            ao.global_data["ekmanemotion"] = {}

            for contribution in ao.contribution_data:
                for emotion in contribution["ekmanemotion"]:
                    if emotion not in ao.global_data["ekmanemotion"]:
                        ao.global_data["ekmanemotion"][emotion] = 0
                    ao.global_data["ekmanemotion"][emotion] += contribution[
                        "ekmanemotion"
                    ][emotion]

            # now we divide by the contribution count
            for emotion in ao.global_data["ekmanemotion"]:
                ao.global_data["ekmanemotion"][emotion] /= ao.global_data[
                    "contribution_count"
                ]

            ao.analyses_done.append(self.__class__.__name__)


class ContributionSentimentAnalyzer(ContributionAnalyzer):

    def __init__(self, model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=get_device()):
        self.classifier = pipeline("text-classification", model=model, return_all_scores=True, device=device)


    def analyze(self, ao: AnalysisObject):

        for contribution in ao.contribution_data:
            cls = self.classifier(contribution["transcript"], truncation=True)[
                0
            ]
            # attach the classification to the contribution in the data
            emotion_dict = {item["label"]: item["score"] for item in cls}
            contribution["sentiment"] = emotion_dict

        ao.analyses_done.append(self.__class__.__name__)


class ParticipantSentimentAnalyzer(ParticipantAnalyzer):

    def __init__(self, model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=get_device()):
        super().__init__()
        self.model = model
        self.device = device

    def analyze(self, ao):

        if ContributionSentimentAnalyzer.__name__ not in ao.analyses_done:
            ContributionSentimentAnalyzer(model=self.model, device=self.device).analyze(ao)

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        for participant in ao.participants:
            ao.participant_data[participant]["sentiment"] = {}
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    for sentiment in contribution["sentiment"]:
                        if (
                            sentiment
                            not in ao.participant_data[participant]["sentiment"]
                        ):
                            ao.participant_data[participant]["sentiment"][sentiment] = 0
                        ao.participant_data[participant]["sentiment"][
                            sentiment
                        ] += contribution["sentiment"][sentiment]

        # now we divide by the contribution count
        for participant in ao.participants:
            for sentiment in ao.participant_data[participant]["sentiment"]:
                ao.participant_data[participant]["sentiment"][
                    sentiment
                ] /= ao.participant_data[participant]["contribution_count"]

        ao.analyses_done.append(self.__class__.__name__)


class GlobalSentimentAnalyzer(GlobalAnalyzer):

    def __init__(self, mode: AnalysisMode = AnalysisMode.ENTIRE, model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=get_device()):
        self.mode = mode
        self.device = device
        self.model = model

    def analyze(self, ao):

        if self.mode == AnalysisMode.AVERAGE_PER_CONTRIBUTION:

            if ContributionSentimentAnalyzer.__name__ not in ao.analyses_done:
                ContributionSentimentAnalyzer(model=self.model, device=self.device).analyze(ao)

            if GlobalContributionCount.__name__ not in ao.analyses_done:
                GlobalContributionCount().analyze(ao)

            ao.global_data["sentiment"] = {}

            for contribution in ao.contribution_data:
                for sentiment in contribution["sentiment"]:
                    if sentiment not in ao.global_data["sentiment"]:
                        ao.global_data["sentiment"][sentiment] = 0
                    ao.global_data["sentiment"][sentiment] += contribution["sentiment"][
                        sentiment
                    ]

            # now we divide by the contribution count
            for sentiment in ao.global_data["sentiment"]:
                ao.global_data["sentiment"][sentiment] /= ao.global_data[
                    "contribution_count"
                ]

            ao.analyses_done.append(self.__class__.__name__)

        elif self.mode == AnalysisMode.AVERAGE_PER_PARTICIPANT:
            raise NotImplementedError("AnalysisMode.ENTIRE not implemented yet")

        else:

            assert self.mode == AnalysisMode.ENTIRE

            classifier = pipeline("text-classification", model=self.model, return_all_scores=True, device=self.device)


            cls = classifier(
                ao.global_data["transcript"], padding=True, truncation=True
            )[0]
            # attach the classification to the contribution in the data
            emotion_dict = {item["label"]: item["score"] for item in cls}
            ao.global_data["sentiment"] = emotion_dict

            ao.analyses_done.append(self.__class__.__name__)


class ContributionSentenceTransformerEmbeddingAnalyzer(ContributionAnalyzer):
    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2

    def __init__(
        self, sentence_transformer="sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        super().__init__()

        self.model = SentenceTransformer(sentence_transformer)

        self.spacy_nlp = spacy.load("en_core_web_sm")

    def analyze(self, ao: AnalysisObject):

        for contribution in ao.contribution_data:

            # As the sentence transformer only transforms sentences we loop over them
            spacy_doc = self.spacy_nlp(contribution["transcript"])

            embeddings = []

            for sentence in spacy_doc.sents:
                sentence_embeddings = self.model.encode(sentence.text)
                embeddings.append(sentence_embeddings)

            # average over the embeddings (are numpy!)
            contribution["sentence_embeddings"] = np.mean(embeddings, axis=0)

        ao.analyses_done.append(self.__class__.__name__)


class ParticipantSentenceTransformerEmbeddingAnalyzer(ParticipantAnalyzer):
    def analyze(self, ao):

        if (
            ContributionSentenceTransformerEmbeddingAnalyzer.__name__
            not in ao.analyses_done
        ):
            ContributionSentenceTransformerEmbeddingAnalyzer().analyze(ao)

        for participant in ao.participants:
            embeddings = []
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    embeddings.append(contribution["sentence_embeddings"])
            ao.participant_data[participant]["sentence_embeddings"] = np.mean(
                embeddings, axis=0
            )

        ao.analyses_done.append(self.__class__.__name__)


class GlobalSentenceTransformerEmbeddingAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):

        if (
            ContributionSentenceTransformerEmbeddingAnalyzer.__name__
            not in ao.analyses_done
        ):
            ContributionSentenceTransformerEmbeddingAnalyzer().analyze(ao)

        embeddings = []

        for participant in ao.participants:
            embeddings = []
            for contribution in ao.contribution_data:
                embeddings.append(contribution["sentence_embeddings"])

        ao.global_data["sentence_embeddings"] = np.mean(embeddings, axis=0)

        ao.analyses_done.append(self.__class__.__name__)
