#!/usr/bin/env python
# coding: utf-8

import pyaudio
import struct
import wave
import numpy as np
import matplotlib.pyplot as plt


p = pyaudio.PyAudio()

CHUNK = 1024 * 8
FORMAT = pyaudio.paInt16
BITWIDTH = 16
CHANNELS = 1
RATE = 48000
DURATION = 10                   # seconds

# def callback(in_data, frame_count, time_info, status):
#     data = stream.read(CHUNK)
#     wavefile.writeframes(data)
#     return (data, pyaudio.paContinue)

def grab_frame():
    for channel, stream in enumerate(streams):
        frames[channel].append(stream.read(CHUNK))

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
input_devices = []
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels'))>0:
        print ("Input Device id"), i, "-", p.get_device_info_by_host_api_device_index(0, i).get('name')
        input_devices.append(i)
        

p= pyaudio.PyAudio()
streams = []
for i in input_devices:
    streams += [p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=i,
        output=True,
        frames_per_buffer=CHUNK
    )]

nchunks = int(DURATION * RATE / CHUNK)
frames = []
for channel in range(len(streams)):
    frames.append([])

for i in range(0, nchunks):
    grab_frame()

for stream in streams:
    stream.stop_stream()
    stream.close()
p.terminate()

for num, channel in enumerate(frames):
    wavfile = wave.open("./test_{}.wav".format(num), 'w')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(p.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(channel))
    wavfile.close()

