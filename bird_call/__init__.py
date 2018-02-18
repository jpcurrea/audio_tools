import os
import time
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sbn

from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
from scipy.signal import butter, lfilter
from pygame import mixer, sndarray


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y


def printProgress(part, whole):
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()


def playAudio(output, sample_rate=44100):
    mixer.init(sample_rate)
    preview = output.copy(order='C')
    # preview = np.array([output, output]).T.copy(order='C')
    preview = sndarray.make_sound(preview.astype(np.int16))

    preview.play()
    return mixer, preview


def get_changes(bool):
    # changes = np.where(np.diff(bool))[0]
    changes = np.where(bool[1:] != bool[:-1])[0]
    if bool[0]:
        changes = changes[1:]
    if bool[-1]:
        changes = changes[:-1]
    ins = changes[::2]
    outs = changes[1::2]
    return ins, outs

# class Syllable():
#     def __init__(self, audio_reference, start, stop, sample_rate=44100):


class Syllable():
    """Class that points to the start of syllables in a
    recording without having to copy the section."""
    def __init__(self, Recording, start, stop):
        self.recording = Recording
        self.start = start
        self.stop = stop

    def play(self):
        self.recording.play(start=self.start, stop=self.stop)

    def plot(self):
        self.recording.plot(start=self.start, stop=self.stop)

    def dur(self):
        return self.stop - self.start

    def max_fund_freq(self, ret=True):
        self.get_frequencies()
        return self.fr_range[np.argmax(self.frequency, axis=0)]

    def get_frequencies(self):
        start = self.start * self.recording.sample_rate
        stop = self.stop * self.recording.sample_rate
        self.frequency = fft(
            self.recording.audio[start:stop], axis=0)
        self.fr_range = np.fft.fftfreq(self.dur())
        self.fr_range = abs(self.fr_range*self.sample_rate)

    def plot_frequencies(self, res_fr=None):  # res_fr gives the plotting acuity
        """Plot the wave in frequency space."""
        if res_fr is None:
            res_fr = max(len(self.audio)/10000, 1)
        self.get_frequencies()
        fig = plt.gcf()
        plt.semilogx(self.fr_range[::res_fr], self.frequency[::res_fr])
        plt.title("{} Frequencies".format(self.name))
        plt.xlabel("temporal fr. (Hz)")
        plt.ylabel("amplitude (dB)")
        plt.tight_layout()
        plt.show()

    def spectrogram(self, disp=True):
        start = self.start*self.recording.sample_rate
        stop = self.stop*self.recording.sample_rate
        self.freqs, self.times, self.spec = signal.spectrogram(
            self.recording.audio[start:stop].T.mean(axis=0),
            self.sample_rate)
        if disp:
            plt.pcolormesh(
                self.times, self.freqs, self.spec, cmap='viridis',
                norm=colors.LogNorm(self.spec.min(), self.spec.max()))
            plt.title("Syllable Spectrogram")
            plt.xlabel("time (s)")
            plt.ylabel("frequency (Hz)")

    def copy(self):
        start = int(
            self.start*self.recording.sample_rate)
        stop = int(
            self.stop*self.recording.sample_rate)
        return self.recording.audio.copy()[start:stop]


class Audio():
    def __init__(self, arr, sample_rate=44100, name=None):
        self.audio = arr        # shape=(#samples, #channels)
        self.sample_rate = sample_rate
        self.name = name
        if self.audio.ndim > 1:
            self.channels = self.audio.shape[1]
            if self.channels == 1:
                self.audio = self.audio.reshape(self.audio.shape[0])
        else:
            self.channels = 1
        self.frequency = None
        self.syllables = None

    def dur(self):
        return self.audio.shape[0]/float(self.sample_rate)

    def max_fund_freq(self, ret=True):
        self.get_frequencies()
        return self.fr_range[np.argmax(self.frequency, axis=0)]

    def get_frequencies(self):
        self.frequency = fft(self.audio, axis=0)
        self.fr_range = np.fft.fftfreq(len(self.audio))
        self.fr_range = abs(self.fr_range*self.sample_rate)

    def plot(self, start=0, stop=None, res_fr=None):
        """Plot the sound wave with appropriate axes, labels, and title.
        """
        if res_fr is None:
            res_fr = max(len(self.audio)/10000, 1)
        if stop is None:
            stop = self.dur()
        stop = int(self.sample_rate*stop)
        start = int(self.sample_rate*start)
        time = np.arange(self.audio.shape[0])/float(self.sample_rate)
        fig = plt.gcf()
        plt.plot(time[start:stop:res_fr], self.audio[start:stop:res_fr])
        plt.title(self.name)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude (dB)")
        fig.tight_layout()
        # plt.show()

    def plot_frequencies(self, res_fr=None): #res_fr gives the plotting acuity
        """Plot the wave in frequency space."""
        if res_fr is None:
            res_fr = max(len(self.audio)/10000, 1)
        self.get_frequencies()
        fig = plt.gcf()
        plt.semilogx(self.fr_range[::res_fr], self.frequency[::res_fr])
        plt.title("{} Frequencies".format(self.name))
        plt.xlabel("temporal fr. (Hz)")
        plt.ylabel("amplitude (dB)")
        fig.tight_layout()
        # plt.show()

    def play(self, start=0, stop=None):  # start and stop are in seconds
        """Use pygame to play the audio. If the audio hasn't been read yet,
        do that first.
        """
        if stop is None:
            stop = self.dur()
        start = int(start*self.sample_rate)
        stop = int(stop*self.sample_rate)
        while mixer.get_busy():
            pass
        playAudio(self.audio[start:stop], self.sample_rate)

    def fr_filter(self, low_freq=None, high_freq=None):  # Hz
        """Use the fft to remove frequencies outside of our range of
        interest."""
        self.fr_range = np.fft.fftfreq(len(self.audio))
        self.fr_range = abs(self.fr_range*self.sample_rate)

        if low_freq is None:
            low_freq = 0
        if high_freq is None:
            high_freq = max(self.fr_range)

        self.audio = butter_bandpass_filter(
            self.audio, low_freq, high_freq, self.sample_rate)

    def spectrogram(self, disp=True, logy=True):
        if self.channels > 1:
            self.freqs, self.times, self.spec = signal.spectrogram(
                self.audio.T.mean(axis=0), self.sample_rate)
        else:
            self.freqs, self.times, self.spec = signal.spectrogram(
                self.audio, self.sample_rate)
        if disp:
            plt.pcolormesh(
                self.times, self.freqs, self.spec, cmap='viridis',
                norm=colors.LogNorm(self.spec.min(), self.spec.max()))
            if logy:
                plt.semilogy()
            plt.title("{} Spectrogram".format(self.name))
            plt.xlabel("time (s)")
            plt.ylabel("frequency (Hz)")
            plt.tight_layout()
            # plt.show()

    def get_syllables(self, smooth=11, min_amps=200, disp=True, disp_res=5):
        """Using a spectrogram and some assumptions about "silence" in the recording,
        seperate out the syllables of the recording.
        """
        self.syllables = []

        self.spectrogram(disp=False)
        self.max_freqs = self.spec.argmax(axis=0)  # max fundamental freqs
        self.max_amps = self.spec.max(axis=0)      # amplitude

        # self.max_fund_freq(ret=False)
        # gets the max frequencies and corresponding amps
        self.max_amps = signal.medfilt(self.max_amps, smooth)

        # get the ins, outs and pause lengths of the times when the the mff
        # is greater than min_amps
        b = self.max_amps > min_amps
        self.ins, self.outs = get_changes(b)
        self.ins = self.times[self.ins]
        self.outs = self.times[self.outs]

        # create an syllable object for every syllable
        for x in range(len(self.ins)):
            self.syllables += [
                Syllable(Recording=self, start=self.ins[x],
                         stop=self.outs[x])]

        if disp:
            first = True
            self.plot(res_fr=disp_res)
            for i, o in zip(self.ins, self.outs):
                if first:
                    plt.axvspan(i, o, color='b', alpha=.3, label="syllables")
                    first = False
                else:
                    plt.axvspan(i, o, color='b', alpha=.3)

    def get_calls(self, smooth=11, min_amps=1000, max_interval=.01,
                  disp=True, disp_res=5):
        """min_interval is in seconds."""
        if self.syllables is None:
            self.get_syllables(smooth, min_amps)

        self.calls = []

        self.intervals = self.ins[1:] - self.outs[:-1]

        ind = self.intervals > max_interval
        self.call_ins = np.append(self.ins[0], self.ins[1:][ind])
        self.call_outs = np.append(self.outs[:-1][ind], self.outs[-1])

        for i, o in zip(self.call_ins, self.call_outs):
            self.calls += [
                Syllable(Recording=self, start=i,
                         stop=o)]
        if disp:
            first = True
            self.plot(res_fr=disp_res)
            for i, o in zip(self.call_ins, self.call_outs):
                if first:
                    plt.axvspan(i, o, color='r', alpha=.3, label="calls")
                    first = False
                else:
                    plt.axvspan(i, o, color='r', alpha=.3)

    def get_durs(self):
        if self.syllables is None:
            self.get_syllables()
        self.durs = []
        for s in self.syllables:
            self.durs += [s.dur()]
        self.durs = np.array(self.durs)

    def get_avg_freqs(self):
        if self.syllables is None:
            self.get_syllables()
        self.avg_freqs = []
        for s in self.syllables:
            self.avg_freqs += [s.max_fund_freq()]
        self.avg_freqs = np.array(self.avg_freqs)


class Recording(Audio):
    """An audio object for wave recordings on file.
    """
    def __init__(self, filename, trim=True):
        self.name = filename
        self.getAudio(trim=trim)
        Audio.__init__(self, self.audio, self.sample_rate, name=self.name)

    def getAudio(self, trim=True):
        """Use scipy's wavfile to read the recording as an array and store its sampling
        rate.
        """
        self.sample_rate, self.audio = wavfile.read(self.name)
        self.original = Audio(self.audio)
        if trim:
            l = len(self.audio)
            approx = int(np.log2(l))
            self.audio = self.audio[2**(approx-1):-2**(approx-1)]

# to do:
# -bird object that can get average stats on each of the birds recordings
# -incorporate entropy and amplitude measures for recording
# -finally, run the analysis for each bird


class Bird():
    """A collective object for handling multiple recordings over different
    days for a single bird.
    """
    def __init__(self, subject_id, recording_info="test_info.csv",
                 bird_info="info.csv", recordings=None):
        self.subject_id = subject_id
        if isinstance(bird_info, str):
            self.bird_info = pd.read_csv(bird_info)
            self.bird_info = self.bird_info[self.bird_info.id ==
                                            self.subject_id]
        else:
            self.bird_info = bird_info
        if isinstance(recording_info, str):
            self.recordings_info = pd.read_csv(recording_info)
            self.recordings_info = self.recordings_info[
                self.recordings_info.id == self.subject_id]
            self.recordings_info.index = range(len(self.recordings_info))
        else:
            self.recordings_info = recording_info

        if recordings is None:
            self.recordings = self.recordings_info.file.values
        else:
            self.recordings = recordings
        self.get_info()
        self.get_recordings()

    def get_recordings(self):
        if isinstance(self.recordings, str):
            self.recordings = Recording(self.recordings)
        if isinstance(self.recordings, (np.ndarray, list, tuple)):
            recs = []
            for x in range(len(self.recordings)):
                rec = Recording(self.recordings[x])

                rec.weight = self.recordings_info.weight[x] - self.tag_weight
                rec.notes = self.recordings_info.notes[x]
                rec.age = - self.hatch_time + pd.to_datetime(
                    time.ctime(
                        os.path.getmtime(self.recordings[x])))
                recs += [rec]
            self.recordings = recs

    def fr_filter(self, low=600, high=10000):
        x = 0
        for rec in self.recordings:
            rec.fr_filter(low, high)
            x += 1
            printProgress(x, len(self.recordings))

    def get_syllables(self, min_amps=200):
        x = 0
        for rec in self.recordings:
            rec.get_syllables(min_amps=200)
            x += 1
            printProgress(x, len(self.recordings))

    def get_info(self):
        self.batch = self.bird_info.batch.values[0]
        self.hatch_date = self.bird_info.hatch_date.values[0]
        self.hatch_time = self.bird_info.hatch_time.values[0]
        self.hatch_time = pd.to_datetime(
            self.hatch_date + " " + self.hatch_time)
        self.tag_weight = self.bird_info.tag_weight.values[0]
        self.rearing_bin = self.bird_info.rearing_bin.values[0]

    def get_avg_durs(self, ret=True):
        self.avg_durs = []
        for rec in self.recordings:
            rec.get_durs()
            self.avg_durs += [np.median(rec.durs)]
        if ret:
            return self.avg_durs

    def get_ages(self, ret=True):
        self.ages = []
        for rec in self.recordings:
            self.ages += [rec.age]
        if ret:
            return self.ages

    def get_avg_freqs(self, ret=True):
        self.avg_freqs = []
        for rec in self.recordings:
            rec.get_avg_freqs()
            self.avg_freqs += [np.median(rec.avg_freqs)]
        if ret:
            return self.avg_freqs

    def get_num_syllables(self, ret=True):
        self.num_syllables = []
        for rec in self.recordings:
            self.num_syllables += [len(rec.syllables)]
        if ret:
            return self.num_syllables

    def plot_durations(self):
        durs = []
        ages = []
        for rec in self.recordings:
            rec.get_durs()
            l = len(rec.durs)
            durs += [rec.durs]
            ages += [np.repeat(rec.age.days, l)]
        ages = np.concatenate(ages)
        durs = np.concatenate(durs)
        sbn.violinplot(x=ages, y=durs)
        plt.xlabel("age (days)")
        plt.ylabel("duration (s)")
