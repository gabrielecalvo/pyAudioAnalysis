# Disclaimer
This is a forked version of the [original pyAudioAnalysis project](https://github.com/tyiannak/pyAudioAnalysis).

I forked this code to port it to python 3 and add some custom functionality I'm interested in.
I am not currently looking to port and test all features.
For more please check the original documentation.

# Usage
```
from pyaudioanalysis.audioSegmentation import speaker_diarization
cluster_labels = speaker_diarization(filepath=fpath, num_of_speakers=num_of_speakers)
```