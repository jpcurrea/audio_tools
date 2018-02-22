import bird_call as bc
from matplotlib import pyplot as plt


rec = bc.Recording("./STE-004.wav", trim=False)
rec.spectrogram(logy=False)
plt.show()
