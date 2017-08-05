import numpy as np
from pyaudioanalysis.audioSegmentation import export_speaker_labels


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
