import os

import numpy

from pyaudioanalysis.audioSegmentation import speaker_diarization

numpy.random.seed(123)  # for reproducibility
ROOT_FLD = os.path.dirname(os.path.dirname(__file__))


def test_speaker_diarization_no_lda():
    file_path = os.path.join(ROOT_FLD, 'data', 'diarizationExample.wav')
    cls = speaker_diarization(filepath=file_path, num_of_speakers=4, lda_dim=0, plot_=True)
    print(cls)
