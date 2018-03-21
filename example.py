import bird_call as bc
from matplotlib import pyplot as plt


rec = bc.Recording("./STE-096.wav", trim=False)

# display a spectrogram of the audio
rec.spectrogram(logy=False)
plt.show()

# grab the calls and syllables and diplay/highlight the calls
rec.get_calls(min_amps=10, max_interval=.2)
plt.show()
