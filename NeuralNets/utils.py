import numpy as np

class BatchModifier:
  def __init__(self):
    pass
  
  def modify_batch(self, batch_data, batch_labels):
    pass

class BatchShuffle(BatchModifier):
  def __init__(self):
    super().__init__()
  
  def modify_batch(self, batch_data, batch_labels):
    N = batch_data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    return batch_data[indices], batch_labels[indices]
  
class BatchMask(BatchModifier):
  
  def __init__(self, chance, amount):
    self.chance = chance
    self.amount = amount

  def modify_batch(self, batch_data, batch_labels):
      
    ndim = batch_data.ndim
    if ndim == 3:
      B_Size, Time, Feat = batch_data.shape
        
      if isinstance(self.amount, (tuple, list)):
        amount_t, amount_f = self.amount
      else:
        amount_t = amount_f = self.amount

      masked_data = np.copy(batch_data)
      for i in range(B_Size):
        if np.random.rand() < self.chance:
          max_t_cut = max(1, Time // amount_t)
          t_width = np.random.randint(1, max_t_cut+1)         
          if Time - t_width > 0:
            t_start = np.random.randint(0, Time - t_width)
            masked_data[i, t_start : t_start + t_width, :] = 0

        if np.random.rand() < self.chance:
          max_f_cut = max(1, Feat // amount_f)
          f_width = np.random.randint(1, max_f_cut+1)
            
          if Feat - f_width > 0:
            f_start = np.random.randint(0, Feat - f_width)
            masked_data[i, :, f_start : f_start + f_width] = 0        
      return masked_data, batch_labels
      

    elif ndim == 4:
      B_Size, C, H, W = batch_data.shape

      if isinstance(self.amount, (tuple, list)):
        amount_h, amount_w = self.amount
      else:
        amount_h = amount_w = self.amount

      masked_data = np.copy(batch_data)
      for i in range(B_Size):

        if np.random.rand() < self.chance:
          max_w_cut = max(1, W // amount_w)
          f_width = np.random.randint(0, max_w_cut) 
          if W - f_width > 0:
            f_start = np.random.randint(0, W - f_width)
            masked_data[i, :, :, f_start : f_start + f_width] = 0
          

        if np.random.rand() < self.chance:
          max_h_cut = max(1, H // amount_h)
          t_width = np.random.randint(0, max_h_cut)
          if H - t_width > 0:
            t_start = np.random.randint(0, H - t_width)
            masked_data[i, :, t_start : t_start + t_width, :] = 0

      return masked_data, batch_labels
      
    else:
      raise ValueError(f"BatchMask expects 3D or 4D input, got {ndim}D")
  
class BatchNoise(BatchModifier):
  def __init__(self, noise_ratio, chance):
    self.noise_ratio = noise_ratio
    self.chance = chance

  def modify_batch(self, batch_data, batch_labels):
    B_Size = batch_data.shape[0]
    sample_shape = batch_data.shape[1:] 

    masked_data = np.copy(batch_data)
    for i in range(B_Size):
      if np.random.rand() < self.chance:
        noise = np.random.normal(0, self.noise_ratio, size=sample_shape)
        masked_data[i] += noise
        
    return masked_data, batch_labels  

class BatchMix(BatchModifier):
  def __init__(self, alpha, chance):
    self.alpha = alpha
    self.chance = chance

  def modify_batch(self, batch_data, batch_labels):     
    if np.random.rand() > self.chance:
      return batch_data, batch_labels
    
    B_size = batch_data.shape[0]
    indices = np.random.permutation(B_size)

    lambd = np.random.beta(self.alpha, self.alpha)

    mixed_data = lambd * batch_data + (1 - lambd) * batch_data[indices]
    mixed_labels = lambd * batch_labels + (1 - lambd) * batch_labels[indices]
      
    return mixed_data, mixed_labels

    
    

#class_names = ["ANG","DIS","FEA","HAP","NEU","SAD","CAL","SUR"]

def calculate_weights_dictionary(dataset):
  unique, sample_count = np.unique(dataset, return_counts=True)
  print(f"unique_labels={unique.tolist()}, counts={sample_count.tolist()}")
  if -1 in unique:
    print("WARNING: -1 label found in split")

  mean_class_count = np.mean(sample_count)
  weights_dictionary = {int(label): float(mean_class_count / count)
                        for label, count in zip(unique, sample_count)}
  return weights_dictionary

def upsample_missing_classes(batch_data, batch_labels):
  batch_labels_extracted = np.argmax(batch_labels, axis=1)
  labels, sample_count = np.unique(batch_labels_extracted, return_counts=True)
  if len(labels) == 0:
    raise ValueError("No labels found in batch_labels")
  if len(labels) <= 1:
    return batch_data, batch_labels

  max_count = np.max(sample_count)
  upsampled_data_list = []
  upsampled_labels_list = []

  for label, count in zip(labels, sample_count):
    indices = np.where(batch_labels_extracted == label)[0]
    label_data = batch_data[indices]
    label_labels = batch_labels[indices]

    if count == max_count:
      upsampled_data_list.append(label_data)
      upsampled_labels_list.append(label_labels)
    else:
      repeats = max_count // count
      remainder = max_count % count
      tile_shape = [1] * label_data.ndim
      tile_shape[0] = repeats
            
      repeated_data = np.tile(label_data, tile_shape)
      repeated_labels = np.tile(label_labels, (repeats, 1))
      if remainder > 0:
        remainder_indices = np.random.choice(indices, size=remainder, replace=False)
                
        remainder_data = batch_data[remainder_indices]
        remainder_labels_part = batch_labels[remainder_indices]
        repeated_data = np.vstack((repeated_data, remainder_data))
        repeated_labels = np.vstack((repeated_labels, remainder_labels_part))

        upsampled_data_list.append(repeated_data)
        upsampled_labels_list.append(repeated_labels)
  return np.vstack(upsampled_data_list), np.vstack(upsampled_labels_list)
      
'''def shuffle_batch_data(batch_data, batch_labels):
  N = batch_data.shape[0]
  indices = np.arange(N)
  np.random.shuffle(indices)
  return batch_data[indices], batch_labels[indices]

def mask_batch_data(batch_data, batch_labels):
  if len(batch_data.shape) != 4:
    raise ValueError("mask_batch_data expects batch_data with 4 dimensions (B_Size,C,H,W)")       
  B_Size, C, H, W = batch_data.shape

  masked_data = np.copy(batch_data)



  for i in range(B_Size):
    if np.random.rand() < 0.6:
      f_width = np.random.randint(0, W // 3) 
      if W - f_width > 0:
        f_start = np.random.randint(0, W - f_width)
        masked_data[i, :, :, f_start : f_start + f_width] = 0
    if np.random.rand() < 0.6:
      t_width = np.random.randint(0, H // 3)
      if H - t_width > 0:
        t_start = np.random.randint(0, H - t_width)
        masked_data[i, :, t_start : t_start + t_width, :] = 0

  return masked_data, batch_labels


def add_noise_to_batch(batch_data, batch_labels):
  if len(batch_data.shape) != 4:
    raise ValueError("add_noise_to_batch expects batch_data with 4 dimensions (B_Size,C,H,W)")       
  B_Size, C, H, W = batch_data.shape

  masked_data = np.copy(batch_data)

  noise_ratio = 0.1


  for i in range(B_Size):
    if np.random.rand() < 0.6:
      noise = np.random.normal(0, noise_ratio, size=(C,H,W))
      masked_data[i] += noise
  return masked_data, batch_labels

def mix_batch(batch_data, batch_labels, alpha=0.35):
  if np.random.rand() > 0.3:
    return batch_data, batch_labels
  
  B_size = batch_data.shape[0]
  indices = np.random.permutation(B_size)

  lambd = np.random.beta(alpha, alpha)

  mixed_data = lambd * batch_data + (1 - lambd) * batch_data[indices]
  mixed_labels = lambd * batch_labels + (1 - lambd) * batch_labels[indices]
    
  return mixed_data, mixed_labels'''

def create_near_identity_matrix(rows, cols, alpha=1.0, eps=0.01):
  W = eps * np.random.randn(rows, cols)
  d = min(rows, cols)
  W[:d, :d] += alpha * np.eye(d)
  return W
  

def remove_gaps(signal, threshold=0.02, window_duration=0.25, sr = 40000):
  window_length = int(sr * window_duration)
  if len(signal) < window_length:
    return signal
  remainder = (window_length - (len(signal) % window_length)) % window_length

  if remainder != 0:
    signal_pad = np.pad(signal, (0, remainder), constant_values=0)
  else:
    signal_pad = signal

  windowed_signal = np.reshape(signal_pad, (-1, window_length))
  rms_signal = np.sqrt(np.mean(windowed_signal**2, axis=1))
  non_silence_indices = rms_signal > threshold
    
  mask = np.repeat(non_silence_indices, window_length)

  return signal_pad[mask]

def quantize_value(val, factor):
  if factor == 32: return val, None
  if factor == 16: return val.astype(np.float16), None
  if factor == 8:
    max = np.max(np.abs(val))
    scaling_factor  = 127/max if max > 0 else 1
    val = np.round(val * scaling_factor).astype(np.int8)
    return val, scaling_factor
  else: raise ValueError(f"Unsupported quantization factor: {factor}! Expected 32, 16 or 8.")

  


       