from NeuralNets.CNN_Classes import Adam, CrossEntropyLoss
from NeuralNets.utils import upsample_missing_classes, BatchMask, BatchMix, BatchShuffle, BatchNoise, remove_gaps, calculate_weights_dictionary
from NeuralNets.audio_data import extract_mel
from NeuralNets.model import Model as M
from NeuralNets.Modules import LSTMModule, CNNModule, BranchModule
import numpy as np
import matplotlib.pyplot as plt

# Extraction example
'''def prepare_dataset_clean(example):
  signal_trimmed = remove_gaps("<filename>", signal, threshold=0.005, window_duration=0.5, sr=20000)
  if len(signal_trimmed) == 0: 
    signal_trimmed = signal 
  mel = extract_mel("None", signal, target_frames=240, filter_N=40, NFFT=2048, augment=False)
  mel = (mel - mel_mean) / (mel_std + 1e-6)
  example["mel"] = mel
  return example'''

# Model Architecture
'''batches_per_epoch = len(X_train) // BATCH_SIZE
total_steps = batches_per_epoch * EPOCHS
adam_crnn = Adam(lr=0.0003, b1=0.9, b2=0.999, weight_decay=0.00005)  
adam_lstm = Adam(lr=0.00005, b1=0.9, b2=0.999, weight_decay=0.00005) 
adam_class = Adam(lr=0.0003, b1=0.9, b2=0.999, weight_decay=0.00005)

adam_crnn.T_max = total_steps
adam_lstm.T_max = total_steps
adam_class.T_max = total_steps

model = M()
cnn_module = CNNModule()
cnn_module.set_optimizer(adam_crnn)

cnn_module.add_layer("Convolution", input_shape=(3, 240, 40), kernel_N=3, kernel_count=48, stride=(2, 2), padding=1)
cnn_module.add_layer("BatchNorm", 48, is_conv=True)
cnn_module.add_layer("GELU")

cnn_module.add_layer("Convolution", input_shape=(48, 120, 20), kernel_N=3, kernel_count=96, stride=(2, 2), padding=1)
cnn_module.add_layer("BatchNorm", 96, is_conv=True)
cnn_module.add_layer("GELU")
cnn_module.add_layer("Dropout", rate=0.2)

cnn_module.add_layer("SpatialAttribution", dims=4, rows=60, columns=10)
cnn_module.add_layer("BatchNorm", 96, is_conv=True)

cnn_module.add_layer("Convolution", input_shape=(96, 60, 10), kernel_N=3, kernel_count=128, stride=1, padding=1)
cnn_module.add_layer("BatchNorm", 128, is_conv=True)
cnn_module.add_layer("GELU")
cnn_module.add_layer("Dropout", rate=0.2)

cnn_module.add_layer("Pooling", pool_size=(1, 2), stride=(1, 2), mode='max')
cnn_module.add_layer("Dropout", rate=0.2)

cnn_module.add_layer("Permute", axes=(0, 2, 1, 3)) 
cnn_module.add_layer("Reshape", input_shape=(60, 128, 5), output_shape=(60, 640))

cnn_module.add_layer("Dense", 640, 512) 
cnn_module.add_layer("GELU")
cnn_module.add_layer("Dropout", rate=0.2)

lstm_module = LSTMModule(input_N=512, hidden_N=256, optimizer=adam_lstm, bidirectional=True, use_sequence=True)
dense_module = CNNModule()
dense_module.set_optimizer(adam_class)

dense_module.add_layer("CoupledPooling", axis=1)
dense_module.add_layer("Dropout", rate=0.5)
dense_module.add_layer("Dense", 1024, 128)
dense_module.add_layer("GELU")
dense_module.add_layer("Dropout", rate=0.5)
dense_module.add_layer("Dense", 128, 8)
dense_module.add_layer("Softmax")

model.add_module(cnn_module)
model.add_module(lstm_module)
model.add_module(dense_module)


model.add_batch_modifier(BatchNoise(chance=0.2, noise_ratio=0.03)) 
model.add_batch_modifier(BatchMask(chance=0.3, amount=(4,4)))
model.add_batch_modifier(BatchMix(chance=0.2, alpha=0.2))
model.add_batch_modifier(BatchShuffle())'''
