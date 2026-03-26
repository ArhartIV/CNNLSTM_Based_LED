import numpy as np
import wave
import random


# Bit-traversal for FFT taken from my other project
def bit_reverse_indices(N):
  bits = int(np.log2(N))
  return np.array([int(f'{i:0{bits}b}'[::-1], 2) for i in range(N)])

# Pre-emphasis for the framing to extract gigher-frequency features
def preemph(signal, alpha=0.97):
  return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def resample(signal, new_length):
  return np.interp(
    np.linspace(0, len(signal), new_length, endpoint=False),
    np.arange(len(signal)),
    signal
  )

def lowpass_filter(singal_freq_domain, fr):
  indicies_keep_first = singal_freq_domain[:fr]
  indicies_keep_last = singal_freq_domain[-fr:]
  filtered_signal = np.concatenate((indicies_keep_first, indicies_keep_last))
  return filtered_signal

def downsample(signal, current_sr, target_sr):
  if current_sr == target_sr:
    return signal

  new_length = int(len(signal) * (target_sr / current_sr))
  downsampled_signal = resample(signal, new_length)

  return downsampled_signal.astype(np.float32)

# Divides the the signal into overlapping frames
def frame(signal, f_size, f_stride, sample_rate):
  f_length = int(round(f_size * sample_rate))
  f_step = int(round(f_stride * sample_rate))
  signal_length = len(signal)

  frames_count = 1 + int(np.ceil((signal_length - f_length) / f_step))

  padded_length = (frames_count - 1) * f_step + f_length
  padded_signal = np.zeros(padded_length, dtype=signal.dtype)
  padded_signal[:signal_length] = signal
  
  index = np.arange(f_length)[None, :] + np.arange(frames_count)[:, None] * f_step
  frames = padded_signal[index]

  return frames 

# Hamming window
def HammingWindow(frames):
  num_frames, frame_length = frames.shape
  return frames * np.hamming(frame_length)


def FFT(signal):
  N = len(signal)

  if not(N & (N - 1) == 0) and N != 0:
    new_N = PowerOfTwo(N)
    new_signal = np.zeros(new_N, dtype=complex)
    new_signal[:N] = signal
    signal = new_signal
    N = new_N

  new_signal = np.array(signal, dtype=complex)

  indices = bit_reverse_indices(N)
  new_signal = new_signal[indices]
  stages = int(np.log2(N))
  for stage in range(1, stages + 1):
    step = 2 ** stage
    half_step = step // 2
    twiddle_factors = np.exp(-2j * np.pi * np.arange(half_step) / step)

    for k in range(0, N, step):
      for n in range(half_step):
        a = new_signal[k + n]
        b = twiddle_factors[n] * new_signal[k + n + half_step]
        new_signal[k + n] = a + b
        new_signal[k + n + half_step] = a - b
  return new_signal



# FFT and Power Spectrum, The FFT implementation is based on the Cooley-Turkey algorithm and also taken from my other project
def FFTandPower(frames, NFFT):

  num_frames, frame_length = frames.shape
  power_spectrum = np.zeros((num_frames, NFFT // 2 + 1))

  for i in range(num_frames):
    signal = np.array(frames[i], dtype=complex)

    if len(signal) < NFFT:
      signal = np.append(signal, np.zeros(NFFT - len(signal)))
    elif len(signal) > NFFT:
      signal = signal[:NFFT]

    N = NFFT
    indices = bit_reverse_indices(N)
    signal = signal[indices]

    stages = int(np.log2(N))
    for stage in range(1, stages + 1):
      step = 2 ** stage
      half_step = step // 2
      twiddle_factors = np.exp(-2j * np.pi * np.arange(half_step) / step)

      for k in range(0, N, step):
        for n in range(half_step):
          a = signal[k + n]
          b = twiddle_factors[n] * signal[k + n + half_step]
          signal[k + n] = a + b
          signal[k + n + half_step] = a - b

    mag = np.abs(signal[:NFFT // 2 + 1]) ** 2
    power_spectrum[i] = (1.0 / NFFT) * mag

  return power_spectrum

def PowerOfTwo(x):
  return 1 << (x - 1).bit_length()

def Inverse_FFT(signal):
  N = len(signal)

  if not (N & (N - 1) == 0) and N != 0:
    new_N = PowerOfTwo(N)
    new_signal = np.zeros(new_N, dtype=complex)
    new_signal[:N] = signal
    signal = new_signal
    N = new_N

  new_signal = np.array(signal, dtype=complex)
  indices = bit_reverse_indices(N)
  new_signal = new_signal[indices]

  stages = int(np.log2(N))
  for stage in range(1, stages + 1):
    step = 2 ** stage
    half_step = step // 2
    twiddle_factors = np.exp(2j * np.pi * np.arange(half_step) / step)

    for k in range(0, N, step):
      for n in range(half_step):
        a = new_signal[k + n]
        b = twiddle_factors[n] * new_signal[k + n + half_step]
        new_signal[k + n] = a + b
        new_signal[k + n + half_step] = a - b

  return new_signal / N




# Reads a WAV file and returns audio data and sampling rate( Changed to support 8-bit and mono/stereo files, since some of the 
# datasets were apparently in 8-bit format)
def readWaveFile(filename):
  with wave.open(filename, 'rb') as wf:
    num_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    num_frames = wf.getnframes()
    raw_data = wf.readframes(num_frames)

  if sample_width == 1:
    dtype = np.uint8
    audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
    audio = audio - 128.0
    audio /= 128.0
  elif sample_width == 2:
    dtype = np.int16
    audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
    audio /= 32768.0
  else:
    raise ValueError("Unsupported sample width: {}".format(sample_width))

  if num_channels > 1:
    audio = audio.reshape(-1, num_channels).mean(axis=1)

  if sample_rate != 20000:
    audio = downsample(audio, sample_rate, 20000)
    sample_rate = 20000

  return audio, sample_rate




#Mel banks to mimic human ear's response to varying frequencies
def melBanks(frames, sample_rate, NFFT, filter_N=40):
  mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
  mel_points = np.linspace(0, mel, filter_N + 2)
  hz_points = 700 * (10**(mel_points / 2595) - 1)
  bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

  f_banks = np.zeros((filter_N, NFFT // 2 + 1))

  
  for m in range(1, filter_N + 1):
    f_m_left, f_m_center, f_m_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
    if f_m_center <= f_m_left:
      f_m_center = f_m_left + 1
    if f_m_right <= f_m_center:
      f_m_right = f_m_center + 1

    denom1 = (f_m_center - f_m_left) if (f_m_center - f_m_left) > 0 else 1
    denom2 = (f_m_right - f_m_center) if (f_m_right - f_m_center) > 0 else 1

    f_banks[m - 1, f_m_left:f_m_center] = (
      (np.arange(f_m_left, f_m_center) - f_m_left) / denom1
    )
    f_banks[m - 1, f_m_center:f_m_right] = (
      (f_m_right - np.arange(f_m_center, f_m_right)) / denom2
    )

  filtered_frames = np.dot(frames, f_banks.T)
  filtered_frames = np.where(filtered_frames == 0, np.finfo(float).eps, filtered_frames)
  filtered_frames = np.log(filtered_frames)
  return filtered_frames

# Discrete Cosine Transform to get MFFC
def DCT(mel_frames, coeff_N=13):
  _, frames_N = mel_frames.shape
  j = np.arange(frames_N)
  i = np.arange(coeff_N).reshape(-1, 1)


  transform_Matrix = np.cos(np.pi * i * (2 * j + 1) / (2 * frames_N))

  transform_Matrix *= np.sqrt(2 / frames_N)
  transform_Matrix[0, :] *= 1 / np.sqrt(2)

  return np.dot(mel_frames, transform_Matrix.T)


# Calculating Delta of the features
def CalculateDelta(mel_coeffs, order=2):
  if order < 1:
    raise ValueError("Order must be at least 1")

  denom = 2 * sum([n**2 for n in range(1, order+1)])
 
  kernel = np.arange(-order, order + 1, dtype=np.float32)
  kernel[order] = 0.0
  kernel /= denom
  padded = np.pad(mel_coeffs.astype(np.float32), ((order, order), (0, 0)), mode="edge")

  delta = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded)
  return delta

def fixShape(features, target_frames=120, training=False):
  def fix_3d(features, target_frames, training):
    if num_frames < target_frames:
      pad_width = target_frames - num_frames
      features = np.pad(features, 
        ((0, 0), (0, pad_width), (0, 0)), 
        mode='constant', 
        constant_values=0)
    elif num_frames > target_frames:
      if training:
        start = np.random.randint(0, num_frames - target_frames)
      else:
        start = (num_frames - target_frames) // 2
      features = features[:, start:start + target_frames, :]       
    return features
  
  def fix_2d(features, target_frames, training):
    if num_frames < target_frames:
      pad_width = target_frames - num_frames
      features = np.pad(features, 
        ((0, pad_width), (0, 0)), 
        mode='constant', 
        constant_values=0)
    elif num_frames > target_frames:
      if training:
        start = np.random.randint(0, num_frames - target_frames)
      else:
        start = (num_frames - target_frames) // 2
      features = features[start:start + target_frames, :]       
    return features

  if features.ndim == 2:
    num_frames, num_coeffs = features.shape
    features = fix_2d(features, target_frames, training)
  elif features.ndim == 3:
    channels, num_frames, num_coeffs = features.shape
    features = fix_3d(features, target_frames, training)
  else:
    raise ValueError("Features have to be a 2D or 3D array")
  
  return features
  

# Data augmentation by adding noise, time-stretching, pitch-shifting, and volume changes to avoid overfitting
def augment_audio(signal, sr):
  if random.random() < 0.5:
    noise = np.random.normal(0, 0.005, signal.shape)
    signal = signal + noise

  if random.random() < 0.3:
    gain = np.random.uniform(0.8, 1.2)
    signal = signal * gain

  if random.random() < 0.3:
    stretch = np.random.uniform(0.9, 1.1)
    new_len = int(len(signal) * stretch)
    signal = resample(signal, new_len)
    '''if len(signal) > sr:
      signal = signal[:sr]
    else:
      pad = np.zeros(sr - len(signal))
      signal = np.concatenate([signal, pad])'''

  if random.random() < 0.3:
    max_shift = int(0.1 * sr)
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
      signal = np.concatenate([np.zeros(shift), signal[:-shift]])
    elif shift < 0:
      signal = np.concatenate([signal[-shift:], np.zeros(-shift)])

  return signal.astype(np.float32)  

# Extracts MFFC feature from a Wav file using all functions from above
def extract_mel(filename, signal_given, target_frames=120, filter_N=64, NFFT=512, augment=False):
  if (signal_given is None) or (len(signal_given) == 0):
    signal, sr = readWaveFile(filename)
  else:
    sr = 20000
    signal = signal_given

  if augment:
    signal = augment_audio(signal, sr)
  
  signal = preemph(signal, alpha=0.97)
  frames = frame(signal, 0.025, 0.01, sr)
  frames = HammingWindow(frames)
  power_spectrum = FFTandPower(frames, NFFT)
  mel_frames = melBanks(power_spectrum, sr, NFFT, filter_N)
  
  delta = CalculateDelta(mel_frames, order=1)
  delta_delta = CalculateDelta(mel_frames, order=2)
  mel_frames = np.stack((mel_frames, delta, delta_delta), axis=0)
  features = fixShape(features=mel_frames, target_frames=target_frames, training=augment)

  return features

#print(extract_mel('test_audio.wav', None, target_frames=120, filter_N=80, NFFT=1024, augment=True).shape)