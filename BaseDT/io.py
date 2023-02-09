import pyaudio
import numpy as np

# def get_audio_data(time = 2, rate = 16000, channels = 1):
#     p = pyaudio.PyAudio()
#     CHANNELS = channels
#     RATE = rate
#     INPUT_BLOCK_TIME = 0.05
#     INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
#
#     stream = p.open(
#         format=pyaudio.paInt16,
#         channels=CHANNELS,
#         rate=RATE,
#         input=True,
#         frames_per_buffer=INPUT_FRAMES_PER_BLOCK
#         )
#
#     audio_data = []
#     for i in range(0, int(RATE * time / INPUT_FRAMES_PER_BLOCK)):
#         data = stream.read(INPUT_FRAMES_PER_BLOCK)
#         audio_data.append(data)
#
#     # Stop recording audio
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#     audio_samples = librosa.core.samples_to_frames(
#         np.frombuffer(b''.join(audio_data), dtype=np.int16),
#         CHANNELS,
#         hop_length=INPUT_FRAMES_PER_BLOCK
#         )
#
#     return audio_samples
import pyaudio
import numpy as np
import matplotlib.pyplot as plt


def record_audio(time = 2, rate = 16000, channels = 1, chunk = 1024, normalize = True):
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

def record_camera():
    pass

print(record_audio(2))
plt.plot(record_audio(2))
plt.show()
