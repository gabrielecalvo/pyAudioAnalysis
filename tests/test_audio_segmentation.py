import os
import numpy as np
import json
from pyaudioanalysis.audioSegmentation import speaker_diarization

np.random.seed(999)  # for reproducibility
ROOT_FLD = os.path.dirname(os.path.dirname(__file__))


def print_json(x):
    print(json.dumps(x, indent=2))


def export_speaker_labels(labels, timestamps):
    speaker_labels = []
    current_speaker = -1
    for t, speaker in zip(timestamps, labels):
        t = float(t)            # to de-numpifying
        speaker = str(speaker)  # to de-numpifying
        if speaker != current_speaker:
            if speaker_labels:
                speaker_labels[-1]['to'] = t
            current_speaker = speaker
            speaker_labels.append({'speaker': current_speaker, "from": t, "final": False})
        else:
            speaker_labels[-1]['to'] = t
    speaker_labels[-1]['final'] = True

    return {"speaker_labels": speaker_labels}


def test_speaker_diarization_no_lda():
    file_path = os.path.join(ROOT_FLD, 'data', 'diarizationExample.wav')
    assert os.path.isfile(file_path)
    ts, cls = speaker_diarization(filepath=file_path, num_of_speakers=4, plot_=True)
    print(cls)

    json_labels = export_speaker_labels(labels=cls, timestamps=ts)
    print(json_labels)


def test_export_speaker_labels():
    cls = np.array([1]*5+[3]*3+[1]*2+[2]*4)
    ts = np.linspace(0, 30, len(cls))

    expected = {"speaker_labels": [
        {"from": 0, "to": 11.53846154, "speaker": '1', "final": False},
        {"from": 11.53846154, "to": 18.46153846, "speaker": '3', "final": False},
        {"from": 18.46153846, "to": 23.07692308, "speaker": '1', "final": False},
        {"from": 23.07692308, "to": 30, "speaker": '2', "final": True},
    ]}

    actual = export_speaker_labels(cls, ts)
    for act, exp in zip(actual["speaker_labels"], expected["speaker_labels"]):
        assert act["speaker"] == exp["speaker"]
        assert act["final"] == exp["final"]
        np.testing.assert_almost_equal(act["from"], exp["from"])
        np.testing.assert_almost_equal(act["to"], exp["to"])
