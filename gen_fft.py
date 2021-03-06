#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Take some wav files, get the spectrum
"""
from __future__ import division
import sys
# import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.fftpack import fftfreq
from scipy.io import wavfile
import pandas
import numpy as np

'''
    Split the music into sample_time size pieces.
    A fourier transform is done over each piece and its output
    it reported.

    music: numpy.ndarray the raw music data
    bitdepth: int the resolution of the wave file
    bitrate: int the sampling rate of the wave file
    sample_time: float the size of the peice to analyze
    return: a list of numpy.ndarray. each array has rows of frequency
    and amplitude pairs, one array per peice analyzed.
'''
def analyze(music, bitdepth=None, bitrate=44100, sample_time=1/6.0):
    if bitdepth is None:
        bitdepth = 8 * music.dtype.itemsize
        print("Automatically picking bit depth as", bitdepth)

    # Normalize to mono
    print("Normalizing music to mono")
    mono = music
    if len(music.shape) > 1:
        mono = np.mean(music, axis = 1)

    # Normalize to [-1, 1)
    print("Normalizing music to standard volume")
    normed = mono / (1 << (bitdepth - 1))

    music_time = len(normed) / bitrate

    samples = []

    n = int(sample_time * bitrate)
    for sample in range(0, int(np.ceil(music_time / sample_time))):
        start = int(sample * n)
        end = int(start + n)

        # print("Analyzing [%d, %d)" % (start, end), file=sys.stderr)
        print "Analyzing [%d, %d)" % (start, end)
        spectrum = fft(normed[start:end])
        spectrum = 2 / n * np.abs(spectrum[0:int(n / 2)])[1:]

        freqs = fftfreq(n, d=(1 / bitrate))[1:int(n/2)]

        lower = None
        upper = None
        for i, freq in enumerate(freqs):
            if freq >= 20 and lower is None:
                lower = i
            if freq > 20000 and upper is None:
                upper = i
        spectrum = spectrum[lower:upper]
        freqs = freqs[lower:upper]

        pairs = np.array([freqs, spectrum]).T
        samples.append(pairs)

    return samples[:-1]

'''
    Load a wave file into memory.

    path: string the file path of the wav file
    return: a tuple (sample rate, wave data)
'''
def load_wav(path):
    return wavfile.read(path)

'''
    get the middle 30 seconds of the song
'''
def middle_30(music, samplerate):
    print("Getting the middle 30 seconds of the song")
    samples = 10 * samplerate * len(music.shape)
    start = int((len(music) / 2) - (samples / 2))
    stop = int(start + (samples / 2))
    return music[start:stop, :]

'''
    Create a test sin wave.
    This wave is built up of a 50Hz and an 80Hz signal.

    return: a tuple (sample rate, data)
'''
def make_sin():
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = 10 + np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    print(len(y))
    sf = 1 / T
    return sf, y

'''
    Get the frequency for a given piano key
'''
def f(n):
    return 440 * 2 ** ((float(n)-49) / 12)

ranges = [f(n) for n in xrange(0,89)]
log2 = np.log(2)
thirteenlog2 = 13 * log2

'''
    Get the piano key for a given frequency
'''
def finv(freq):
    return (12 * np.log(freq/55) + thirteenlog2) / log2

'''
    sample is an array of (freq, amp)
'''
def bucketSample(sample, id=None):
    processed = np.zeros(90)
    for freq, amp in sample:
        key = min(max(int(np.round((finv(freq)))), 0), 89)
        processed[key] = processed[key] + amp
    if id is not None:
        print "Done with sample", id
    return processed

'''
    bitrate is the rate for the output wave file
'''
def unbucketSample(processed, keymap, nfreqs=3674, id=None):
    unbucket = np.zeros(nfreqs)
    for key, amp in enumerate(processed):
        unbucket[keymap[key]] = amp
    return unbucket

def reproduceSong(series):
    song = []
    for spectrum in series:
        ift = ifft(spectrum)
        # print ift.shape --> 3674 aka 1/6 our new bitrate
        song = np.concatenate([song, ift])
    song = np.array((np.array(song).real) * (1 << 15), dtype='int8')
    return song

if __name__ == "__main__":
    import md5
    sample_time = 1/6.0

    bitrate, music = load_wav(sys.argv[1])
    music = middle_30(music, bitrate)

    n = int(bitrate * sample_time)
    freqs = fftfreq(n, d=(1 / bitrate))[1:int(n/2)]
    print len(freqs)

    analysis = analyze(music, bitrate=bitrate, sample_time=sample_time)

    bucketed = [bucketSample(sample, id) for id, sample in enumerate(analysis)]

    hash_file = md5.new()
    hash_file.update(sys.argv[1])
    file_name = hash_file.digest().encode('base64').replace("/", "").replace("=", "").strip()
    inputData = pandas.DataFrame(bucketed)
    inputData.to_csv(file_name + ".csv", index=False, index_label=False, header=False)

    keys = [x for x in xrange(0, 90)]
    keymap = {}
    for key in keys:
        index = None
        keyfreq = f(key)
        for i, freq in enumerate(freqs):
            if freq > keyfreq:
                index = i
                break
        if index is None:
            print 'NOOOOOOOOO'
            exit(100)
        keymap[key] = index

    print keymap
    unbucketed = [unbucketSample(sample, keymap, nfreqs=len(freqs), id=id) for sample in bucketed]
    outputData = pandas.DataFrame(unbucketed)
    song = reproduceSong(unbucketed)
    print song
    wavfile.write("NEW_" + file_name + ".wav", 3674*6, song)
