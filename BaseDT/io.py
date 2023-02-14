import pyaudio
import numpy as np
import matplotlib.pyplot as plt

class MicroPhone(object):
    def __init__(self):
        pass

    def record_audio(self, time = 2, rate = 16000, channels = 1, chunk = 1024, normalize = True):
        p = pyaudio.PyAudio()
        CHUNK = chunk
        CHANNELS = channels
        RATE = rate

        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(RATE / CHUNK * time)):
            data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wave = np.concatenate(frames)
        if normalize: return wave / np.iinfo(np.int16).max
        else: return wave

class Camera(object):
    def  __init__(self):
        pass

    def record_camera(self):
        pass


