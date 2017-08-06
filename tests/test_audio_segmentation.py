import os
import numpy as np
from pyaudioanalysis.audioSegmentation import *

np.random.seed(123)  # for reproducibility
get_test_file = lambda x: os.path.join(os.path.dirname(__file__), 'input', x)


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


def test_speaker_diarization():
    result = speaker_diarization(get_test_file('diarizationExample.wav'), 4)
    expected = {'speaker_labels': [
        {'speaker': '2.0', 'from': 0.1, 'final': False, 'to': 9.9},
        {'speaker': '3.0', 'from': 9.9, 'final': False, 'to': 19.9},
        {'speaker': '0.0', 'from': 19.9, 'final': False, 'to': 28.1},
        {'speaker': '1.0', 'from': 28.1, 'final': True, 'to': 41.9}
    ]}
    assert_speaker_labels_equal(result, expected)


# === aux testing functions ===
def assert_speaker_labels_equal(actual, expected):
    assert list(actual.keys()) == list(expected.keys()) == ['speaker_labels']
    assert len(actual['speaker_labels']) == len(expected['speaker_labels'])
    for i, j in zip(actual['speaker_labels'], expected['speaker_labels']):
        assert i['speaker'] == j['speaker']
        np.testing.assert_almost_equal(i['from'], j['from'])
        np.testing.assert_almost_equal(i['to'], j['to'])
        assert i['final'] == j['final']
