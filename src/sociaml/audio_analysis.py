import os
import urllib.request
import zipfile
import librosa
import numpy as np

from .analysis import GlobalAnalyzer, ParticipantAnalyzer, ContributionAnalyzer
from .text_analysis import (
    ParticipantContributionCount,
    GlobalContributionCount,
    ParticipantContributionCount,
)
from .datastructures import AnalysisObject, AnalysisMode
from .models.wav2vec2LargeRobust12ftEmotionMspDim import (
    Wav2vec2LargeRobust12ftEmotionMspDim,
)


class AudioEmotionModel:
    def __init__(self) -> None:
        # implicitly default to the best model https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
        self.model = Wav2vec2LargeRobust12ftEmotionMspDim()

    def process_audio(self, audio, sampling_rate):
        resampled = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        clf = self.model.process(resampled, 16000)

        return {"arousal": clf[0][0], "dominance": clf[0][1], "valence": clf[0][2]}


class ContributionAudioEmotionAnalyzer(ContributionAnalyzer):
    def __init__(self):
        self.model = AudioEmotionModel()

    def analyze(self, ao: AnalysisObject):
        audio, samplingrate = librosa.load(ao.audio_path)

        for contribution in ao.contribution_data:
            relevant_audio = audio[
                int(contribution["start"] * samplingrate) : int(
                    contribution["end"] * samplingrate
                )
            ]
            contribution["audio_emotion"] = self.model.process_audio(
                relevant_audio, samplingrate
            )

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
            ao.participant_data[participant]["audio_emotion"] = {}
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    for emotion in contribution["audio_emotion"]:
                        if (
                            emotion
                            not in ao.participant_data[participant]["audio_emotion"]
                        ):
                            ao.participant_data[participant]["audio_emotion"][
                                emotion
                            ] = 0
                        ao.participant_data[participant]["audio_emotion"][
                            emotion
                        ] += contribution["audio_emotion"][emotion]

        # now we divide by the contribution count
        for participant in ao.participants:
            for emotion in ao.participant_data[participant]["audio_emotion"]:
                ao.participant_data[participant]["audio_emotion"][
                    emotion
                ] /= ao.participant_data[participant]["contribution_count"]

        ao.analyses_done.append(self.__class__.__name__)


class GlobalAudioEmotionAnalyzer(GlobalAnalyzer):

    def __init__(self, mode: AnalysisMode = AnalysisMode.ENTIRE):
        self.mode = mode
        self.model = AudioEmotionModel()

    def analyze(self, ao):

        if self.mode == AnalysisMode.AVERAGE_PER_CONTRIBUTION:

            if ContributionAudioEmotionAnalyzer.__name__ not in ao.analyses_done:
                ContributionAudioEmotionAnalyzer().analyze(ao)

            if GlobalContributionCount.__name__ not in ao.analyses_done:
                GlobalContributionCount().analyze(ao)

            if ParticipantContributionCount.__name__ not in ao.analyses_done:
                ParticipantContributionCount().analyze(ao)

            ao.global_data["audio_emotion"] = {}

            for contribution in ao.contribution_data:
                print(contribution)
                for emotion in contribution["audio_emotion"]:
                    if emotion not in ao.global_data["audio_emotion"]:
                        ao.global_data["audio_emotion"][emotion] = 0
                    ao.global_data["audio_emotion"][emotion] += contribution[
                        "audio_emotion"
                    ][emotion]

            # now we divide by the contribution count
            for emotion in ao.global_data["audio_emotion"]:
                ao.global_data["audio_emotion"][emotion] /= ao.global_data[
                    "contribution_count"
                ]

            ao.analyses_done.append(self.__class__.__name__)

        elif self.mode == AnalysisMode.AVERAGE_PER_PARTICIPANT:
            raise NotImplementedError("Not implemented yet")

        else:

            assert self.mode == AnalysisMode.ENTIRE

            audio, samplingrate = librosa.load(ao.audio_path)

            ao.global_data["audio_emotion"] = self.model.process_audio(
                audio, samplingrate
            )

            ao.analyses_done.append(self.__class__.__name__)


class ContributionAudioSpeakingTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao: AnalysisObject):

        for contribution in ao.contribution_data:
            contribution["speaking_time"] = contribution["end"] - contribution["start"]

        ao.analyses_done.append(self.__class__.__name__)


class ParticipantAudioSpeakingTimeAnalyzer(ContributionAnalyzer):

    def __init__(self, reduction="mean"):

        self.reduction = reduction

    def analyze(self, ao):

        if ContributionAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSpeakingTimeAnalyzer().analyze(ao)

        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        for participant in ao.participants:
            ao.participant_data[participant]["speaking_time_sum"] = 0
            ao.participant_data[participant]["speaking_time_average"] = 0
            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    ao.participant_data[participant][
                        "speaking_time_sum"
                    ] += contribution["speaking_time"]

        if self.reduction == "mean":
            for participant in ao.participants:
                ao.participant_data[participant]["speaking_time_average"] = (
                    ao.participant_data[participant]["speaking_time_sum"]
                    / ao.participant_data[participant]["contribution_count"]
                )

        ao.analyses_done.append(self.__class__.__name__)


class GlobalAudioSpeakingTimeAnalyzer(GlobalAnalyzer):

    def analyze(self, ao):

        if ContributionAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSpeakingTimeAnalyzer().analyze(ao)

        if ParticipantAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ParticipantAudioSpeakingTimeAnalyzer().analyze(ao)

        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        if ParticipantAudioSpeakingTimeAnalyzer.__name__ not in ao.analyses_done:
            ParticipantAudioSpeakingTimeAnalyzer().analyze(ao)

        ao.global_data["speaking_time_sum"] = 0

        for participant in ao.participants:
            ao.global_data["speaking_time_sum"] += ao.participant_data[participant][
                "speaking_time_sum"
            ]

        ao.global_data["speaking_time_average_over_contributions"] = (
            ao.global_data["speaking_time_sum"] / ao.global_data["contribution_count"]
        )
        ao.global_data["speaking_time_average_over_participants"] = ao.global_data[
            "speaking_time_sum"
        ] / len(ao.participants)

        ao.analyses_done.append(self.__class__.__name__)


class ContributionMFCCAnalyzer(ContributionAnalyzer):

    def analyze(self, ao: AnalysisObject):

        audio, samplingrate = librosa.load(ao.audio_path)

        for contribution in ao.contribution_data:
            relevant_audio = audio[
                int(contribution["start"] * samplingrate) : int(
                    contribution["end"] * samplingrate
                )
            ]
            mfccs = librosa.feature.mfcc(y=relevant_audio, sr=samplingrate)
            contribution["mfcc"] = np.mean(mfccs, axis=1)

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
            ao.participant_data[participant]["mfcc"] = np.zeros_like(
                ao.contribution_data[0]["mfcc"]
            )

            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    ao.participant_data[participant]["mfcc"] += contribution["mfcc"]

                ao.participant_data[participant]["mfcc"] /= ao.participant_data[
                    participant
                ]["contribution_count"]

        ao.analyses_done.append(self.__class__.__name__)


class GlobalMFCCAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        if ContributionMFCCAnalyzer.__name__ not in ao.analyses_done:
            ContributionMFCCAnalyzer().analyze(ao)

        if GlobalContributionCount.__name__ not in ao.analyses_done:
            GlobalContributionCount().analyze(ao)

        if ParticipantContributionCount.__name__ not in ao.analyses_done:
            ParticipantContributionCount().analyze(ao)

        ao.global_data["mfcc"] = np.zeros_like(ao.contribution_data[0]["mfcc"])
        count = len(ao.contribution_data)

        for contribution in ao.contribution_data:
            ao.global_data["mfcc"] += contribution["mfcc"]

        ao.global_data["mfcc"] /= ao.global_data["contribution_count"]

        ao.analyses_done.append(self.__class__.__name__)


class ContributionAudioSilenceTimeAnalyzer(ContributionAnalyzer):
    def analyze(self, ao):
        contributions = ao.contribution_data
        for i, contribution in enumerate(contributions):
            silent_before = (
                contributions[i]["start"] - contributions[i - 1]["end"] if i > 0 else 0
            )
            silent_after = (
                contributions[i + 1]["start"] - contributions[i]["end"]
                if i < len(contributions) - 1
                else 0
            )

            contribution["silent_time_before"] = silent_before
            contribution["silent_time_after"] = silent_after
            contribution["silent_time"] = (
                (silent_before + silent_after) / 2
                if silent_before and silent_after
                else 0
            )

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
            ao.participant_data[participant]["silent_time_before_sum"] = 0
            ao.participant_data[participant]["silent_time_after_sum"] = 0
            ao.participant_data[participant]["silent_time_sum"] = 0

            for contribution in ao.contribution_data:
                if contribution["speaker"] == participant:
                    ao.participant_data[participant][
                        "silent_time_before_sum"
                    ] += contribution["silent_time_before"]
                    ao.participant_data[participant][
                        "silent_time_after_sum"
                    ] += contribution["silent_time_after"]
                    ao.participant_data[participant]["silent_time_sum"] += contribution[
                        "silent_time"
                    ]

            ao.participant_data[participant]["silent_time_before_avg"] = (
                ao.participant_data[participant]["silent_time_before_sum"]
                / ao.participant_data[participant]["contribution_count"]
            )
            ao.participant_data[participant]["silent_time_after_avg"] = (
                ao.participant_data[participant]["silent_time_after_sum"]
                / ao.participant_data[participant]["contribution_count"]
            )
            ao.participant_data[participant]["silent_time_avg"] = (
                ao.participant_data[participant]["silent_time_sum"]
                / ao.participant_data[participant]["contribution_count"]
            )

        ao.analyses_done.append(self.__class__.__name__)


class GlobalAudioSilenceTimeAnalyzer(GlobalAnalyzer):
    def analyze(self, ao):
        if ContributionAudioSilenceTimeAnalyzer.__name__ not in ao.analyses_done:
            ContributionAudioSilenceTimeAnalyzer().analyze(ao)

        if ParticipantAudioSilenceTimeAnalyzer.__name__ not in ao.analyses_done:
            ParticipantAudioSilenceTimeAnalyzer().analyze(ao)

        ao.global_data["silent_time_before_sum"] = 0
        ao.global_data["silent_time_after_sum"] = 0
        ao.global_data["silent_time_avg_sum"] = 0

        for participant in ao.participants:
            ao.global_data["silent_time_before_sum"] += ao.participant_data[
                participant
            ]["silent_time_before_sum"]
            ao.global_data["silent_time_after_sum"] += ao.participant_data[participant][
                "silent_time_after_sum"
            ]
            ao.global_data["silent_time_avg_sum"] += ao.participant_data[participant][
                "silent_time_sum"
            ]

        contribution_count = len(ao.contribution_data)
        participant_count = len(ao.participants)

        ao.global_data["silent_time_before_avg_over_contributions"] = (
            ao.global_data["silent_time_before_sum"] / contribution_count
        )
        ao.global_data["silent_time_after_avg_over_contributions"] = (
            ao.global_data["silent_time_after_sum"] / contribution_count
        )
        ao.global_data["silent_time_avg_avg_over_contributions"] = (
            ao.global_data["silent_time_avg_sum"] / contribution_count
        )

        ao.global_data["silent_time_before_avg_over_participants"] = (
            ao.global_data["silent_time_before_sum"] / participant_count
        )
        ao.global_data["silent_time_after_avg_over_participants"] = (
            ao.global_data["silent_time_after_sum"] / participant_count
        )
        ao.global_data["silent_time_avg_avg_over_participants"] = (
            ao.global_data["silent_time_avg_sum"] / participant_count
        )

        ao.analyses_done.append(self.__class__.__name__)
