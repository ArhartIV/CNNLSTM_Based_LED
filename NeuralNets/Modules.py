
from .utils import create_near_identity_matrix, quantize_value
import numpy as np
from .CNN_Classes import Layer, Dense, Pooling, LeakyReLU, Softmax, Adam, Sigmoid, BatchNorm, Dropout, GlobalAvgPooling, CrossEntropyLoss, GELU, AdamWrapper
from .CNN_Classes import Convolve, Reshape, Optimizer, Permute, SpatialAttribution, GlobalMaxPooling, CoupledPooling
from copy import deepcopy

class Module:
  def __init__(self):
    pass

  def forward(self, input, b_training: bool):
    pass

  def backward(self, optimizer=None):
    pass

  def inference(self, input):
    pass

  def get_config(self, get_weights: bool):
    pass
  
  @staticmethod
  def load_module():
    pass


class CNNModule(Module):
  def __init__(self):
    self.available_layers = {
      "Dense": Dense,
      "Convolution": Convolve,
      "LeakyReLU": LeakyReLU,
      "Softmax": Softmax,
      "Sigmoid": Sigmoid,
      "GELU": GELU,
      "BatchNorm": BatchNorm,
      "Reshape": Reshape,
      "Pooling": Pooling,
      "Dropout": Dropout,
      "GlobalAvgPooling": GlobalAvgPooling,
      "GlobalMaxPooling": GlobalMaxPooling,
      "CoupledPooling": CoupledPooling,
      "Permute": Permute, 
      "SpatialAttribution": SpatialAttribution
      
    }
    self.optimizer = None
    self.layers = []
    self.epoch = 0
    pass

  def forward(self, input, b_training : bool):
    out = input
    for layer in self.layers:
      out = layer.forwardProp(out, b_training)
    return out
  
  def backward(self, grad):
    if not self.optimizer:
      raise ValueError("[CNNModule, backward]: No optimizer given")
    for layer in reversed(self.layers):
      grad = layer.backwardProp(grad, self.optimizer)
    return grad

  
  
  def add_layer(self, layer_name : str, *args, **kwargs):
    if layer_name not in self.available_layers:
      raise ValueError(f"Layer '{layer_name}' is not available. Choose from {list(self.available_layers.keys())}")
    
    layer_class = self.available_layers[layer_name]
    layer = layer_class(*args, **kwargs)
    self.layers.append(layer)

  def get_config(self, get_weights : bool, quantize_factor : int = 32):
    configs = []
    weights = {}
    
    for i, layer in enumerate(self.layers):
      configs.append(layer.getConfig())
      w = layer.getParams()
      if w is not None and get_weights:
        for key, val in w.items():
          quantized_val, scale = quantize_value(val, quantize_factor)
          weights[f"{i}:{key}"] = quantized_val
          if scale is not None:
            weights[f"{i}:{key}_scale"] = scale

    return configs, weights
  
  @classmethod
  def load_module(cls, configs, weights, file_name=None):
    instance = cls()

    if not configs or not weights:
      raise ValueError("[load module]: Invalid configs or weights loaded")
    for i, cfg in enumerate(configs):
      layer_type = cfg["layer"]

      if layer_type not in instance.available_layers:
        raise ValueError(f"[load module]: Unknown layer type: {layer_type}")
      
      layer_class = instance.available_layers[layer_type]

      init_args = {k: v for k, v in cfg.items() if k != "layer"}
      layer = layer_class(**init_args)

      prefix = f"{i}:"
      layer_weights = {k.split(":", 1)[1]: weights[k] for k in weights if k.startswith(prefix)}
            
      if layer_weights:
        layer.setParams(layer_weights)
            
      instance.layers.append(layer)
            
    return instance
  
  def get_optimizer(self):
    return self.optimizer
  
  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  def save_optimizer_state(self, filename):
    if self.optimizer:
      self.optimizer.SaveOptimizer(filename, self.layers)



class LSTMModule(Module):
  def __init__(self, hidden_N=0, input_N=0, optimizer=None, bidirectional=False, use_sequence=False):
    self.hidden_N = hidden_N
    self.input_N = input_N
    self.lr = 0.0001
    self.optimizer = optimizer
    self.bidirectional = False
    self.use_sequence = use_sequence


    if self.optimizer: self.optimizer_wrappers = []

    if input_N != 0 and hidden_N !=0:
      self.initialize()

    if bidirectional:
      self.backward_module = deepcopy(self)

    self.bidirectional = bidirectional  

  @staticmethod
  def tanh(x):
    return np.tanh(x)
  
  @staticmethod
  def tanh_deriv(tanh_x):
    return 1 - tanh_x**2

  @staticmethod
  def sigmoid(x):
    return np.where(
    x >= 0,
    1 / (1 + np.exp(-x)),
    np.exp(x) / (1 + np.exp(x))
  )
  
  @staticmethod
  def sigmoid_deriv(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

  def update_wo_optimizer(self):

    lr = self.lr
    self.Wf -= lr * self.dWf
    self.Uf -= lr * self.dUf
    self.bf -= lr * self.dbf

    self.Wi -= lr * self.dWi
    self.Ui -= lr * self.dUi
    self.bi -= lr * self.dbi

    self.Wnc -= lr * self.dWnc
    self.Unc -= lr * self.dUnc
    self.bnc -= lr * self.dbnc

    self.Wo -= lr * self.dWo
    self.Uo -= lr * self.dUo
    self.bo -= lr * self.dbo

    self.Wci -= lr * self.dWci
    self.bci -= lr * self.dbci

  def initialize(self):
    input_N = self.input_N
    hidden_N = self.hidden_N

    def xavier_init(fan_in, fan_out):
      limit = np.sqrt(6 / (fan_in + fan_out))
      return np.random.uniform(-limit, limit, (fan_out, fan_in))

    self.Wf = xavier_init(input_N, hidden_N)  # forget weights matrix
    self.Wi = xavier_init(input_N, hidden_N)  # input weights matrix
    self.Wnc = xavier_init(input_N, hidden_N) # next cell weights matrix
    self.Wo = xavier_init(input_N, hidden_N)  # output weights matrix
    self.Wci = xavier_init(hidden_N, hidden_N) # Cell influence on next hidden state weights matrix

    '''self.Wf = create_near_identity_matrix(hidden_N, input_N)  # forget weights matrix
    self.Wi = create_near_identity_matrix(hidden_N, input_N)  # input weights matrix
    self.Wnc = create_near_identity_matrix(hidden_N, input_N) # next cell weights matrix
    self.Wo = create_near_identity_matrix(hidden_N, input_N)  # output weights matrix
    self.Wci = create_near_identity_matrix(hidden_N,  hidden_N) # Cell influence on next hidden state weights matrix'''

    self.Uf = create_near_identity_matrix(hidden_N, hidden_N) # forget hidden state weights matrix
    self.Ui = create_near_identity_matrix(hidden_N, hidden_N) # input hidden state weights matrix
    self.Unc = create_near_identity_matrix(hidden_N, hidden_N) # next cell hidden state weights matrix
    self.Uo = create_near_identity_matrix(hidden_N, hidden_N) # output hidden state weights matrix

    if self.optimizer:

      self.bf = np.ones((hidden_N, 1)) 
      self.bi = np.zeros((hidden_N, 1))
      self.bnc = np.zeros((hidden_N, 1))
      self.bo = np.zeros((hidden_N, 1))
      self.bci = np.zeros((hidden_N, 1))

      self.dummy_bias = np.zeros((hidden_N, 1))

      self.wrap_Wf = AdamWrapper(self.Wf, self.bf)     
      self.wrap_Wi = AdamWrapper(self.Wi, self.bi)
      self.wrap_Wnc = AdamWrapper(self.Wnc, self.bnc)
      self.wrap_Wo = AdamWrapper(self.Wo, self.bo)
      self.wrap_Wci = AdamWrapper(self.Wci, self.bci)

      self.wrap_Uf = AdamWrapper(self.Uf, self.dummy_bias)
      self.wrap_Ui = AdamWrapper(self.Ui, self.dummy_bias)
      self.wrap_Unc = AdamWrapper(self.Unc, self.dummy_bias)
      self.wrap_Uo = AdamWrapper(self.Uo, self.dummy_bias)

      self.optimizer_wrappers = [
        self.wrap_Wf, self.wrap_Uf,
        self.wrap_Wi, self.wrap_Ui,
        self.wrap_Wnc, self.wrap_Unc,
        self.wrap_Wo, self.wrap_Uo,
        self.wrap_Wci
      ]

  def forward(self, input, b_training: bool):
    if input.ndim > 3:
      if input.shape[-1] != 1 and input.shape[-2] != 1:
        input = np.squeeze(input)
      else:
        raise ValueError(f"[LSTM Module]: Got input with unexpected dimensions {input.shape}")
    elif input.ndim < 3:
      raise ValueError(f"[LSTM Module]: Got input with unexpected dimensions {input.shape}")
    
    B_size, time_steps, input_N = input.shape

    h_t = np.zeros((self.hidden_N, B_size))
    C_t = np.zeros((self.hidden_N, B_size))

    self.cache_h = []
    self.cache_C = []
    
    if b_training:
      self.cache_X = []
      self.cache_gates = [] # Order is: F, I, C_pot, O, Ci

    for t in range(time_steps):
      x_t_extracted = input[:,t,:]
      x_t = x_t_extracted.T

      F = self.sigmoid(self.Wf @ x_t + self.Uf @ h_t + self.bf)
      I = self.sigmoid(self.Wi @ x_t + self.Ui @ h_t + self.bi)
      C_pot = self.tanh(self.Wnc @ x_t + self.Unc @ h_t + self.bnc)
      O = self.sigmoid(self.Wo @ x_t + self.Uo @ h_t + self.bo)

      C_hat = C_t * F
      C_new = (C_pot * I) + C_hat

      Ci = self.tanh(self.Wci @ C_new + self.bci)

      h_new = Ci * O
      h_t = h_new
      C_t = C_new

      if b_training:
        self.cache_X.append(x_t)
        self.cache_gates.append((F,  I, C_pot, O, Ci))

      self.cache_h.append(h_new)
      self.cache_C.append(C_new)
    
    output_stack = np.array(self.cache_h) 

    if self.use_sequence:
      output_final = output_stack.transpose(2, 0, 1) # Shape: (B_size, time_steps, hidden_N)
    else:
      output_final = output_stack.transpose(2, 0, 1)[:, -1, :]

    if self.bidirectional:
      backward_output = self.backward_module.forward(np.flip(input, axis=1), b_training=b_training)

      if self.use_sequence:
        backward_output = np.flip(backward_output, axis=1)
        output_final = np.concatenate((output_final, backward_output), axis=2)
      else:
        output_final = np.concatenate((output_final, backward_output), axis=1)

    return output_final

  def backward(self, grad):

    B_size = grad.shape[0]
    time_steps = len(self.cache_X)

    if self.bidirectional:
      if self.use_sequence:
        bwd_grad = grad[:, :, self.hidden_N:]
        grad = grad[:, :, :self.hidden_N]
        bwd_grad = np.flip(bwd_grad, axis=1)
      else:
        bwd_grad = grad[:, self.hidden_N:]
        grad = grad[:, :self.hidden_N]
      dx_backward = self.backward_module.backward(bwd_grad)

    if grad.ndim == 2:
      # grad_3d shape is (Batch_Size, Time_Steps, hidden_N)
      grad_3d = np.zeros((B_size, time_steps, self.hidden_N))
      grad_3d[:, -1, :] = grad
      grad = grad_3d


    dh_next = np.zeros((self.hidden_N, grad.shape[0])) 
    d_C_next = np.zeros((self.hidden_N, grad.shape[0]))
    dx = np.zeros((B_size, time_steps, self.input_N))
    self.grad_h = grad

    dWf = dWi = dWnc = dWo = dWci = 0
    dUf = dUi = dUnc = dUo = 0
    dbf = dbi = dbnc = dbo = dbci = 0


    for t in reversed(range(len(self.cache_X))):
      x_t = self.cache_X[t].T
      C_t = self.cache_C[t]    
      F, I, C_nc , O, Ci = self.cache_gates[t]


      if t != 0:
        h_t_prev = self.cache_h[t-1].T
        C_t_prev = self.cache_C[t-1]
      else:
        h_t_prev = np.zeros((self.hidden_N, grad.shape[0])).T
        C_t_prev = np.zeros((self.hidden_N, grad.shape[0]))


      dh_t =  grad[:,t,:].T + dh_next

      dh_O = dh_t * O
      dMci = dh_O * self.tanh_deriv(Ci)
      dWci += dMci @ C_t.T

      dh_Ci = dh_t * Ci
      dMo = dh_Ci * self.sigmoid_deriv(O)
      dWo += dMo @ x_t
      dUo += dMo @ h_t_prev

      dCt = self.Wci.T @ dMci + d_C_next
      d_C_next = dCt * F 

      dC_I = dCt * I
      dMnc = dC_I * self.tanh_deriv(C_nc)
      dWnc += dMnc @ x_t
      dUnc += dMnc @ h_t_prev

      dC_nc = dCt * C_nc
      dMi = dC_nc * self.sigmoid_deriv(I)
      dWi += dMi @ x_t
      dUi += dMi @ h_t_prev

      dC_f = dCt * C_t_prev
      dMf = dC_f * self.sigmoid_deriv(F)
      dWf += dMf @ x_t
      dUf += dMf @ h_t_prev

      dh_next = (self.Uf.T @ dMf) + (self.Ui.T @ dMi) + (self.Unc.T @ dMnc) + (self.Uo.T @ dMo)
      dx_t = (self.Wf.T @ dMf) + (self.Wi.T @ dMi) + (self.Wnc.T @ dMnc) + (self.Wo.T @ dMo)

      dx[:,t,:] = dx_t.T

      dbf += np.sum(dMf, axis=1, keepdims=True)
      dbi += np.sum(dMi, axis=1, keepdims=True)
      dbnc += np.sum(dMnc, axis=1, keepdims=True)
      dbo += np.sum(dMo, axis=1, keepdims=True)
      dbci += np.sum(dMci, axis=1, keepdims=True)

    self.dWf, self.dUf, self.dbf = dWf, dUf, dbf
    self.dWi, self.dUi, self.dbi = dWi, dUi, dbi
    self.dWnc, self.dUnc, self.dbnc = dWnc, dUnc, dbnc
    self.dWo, self.dUo, self.dbo = dWo, dUo, dbo
    self.dWci, self.dbci = dWci, dbci

    if self.optimizer:
      clip_v = 1.0
      clip = lambda g: np.clip(g, -clip_v, clip_v)
        
      self.optimizer.updateParams(self.wrap_Wf, clip(dWf), clip(dbf))
      self.optimizer.updateParams(self.wrap_Wi, clip(dWi), clip(dbi))
      self.optimizer.updateParams(self.wrap_Wnc, clip(dWnc), clip(dbnc))
      self.optimizer.updateParams(self.wrap_Wo, clip(dWo), clip(dbo))
      self.optimizer.updateParams(self.wrap_Wci, clip(dWci), clip(dbci))

      dummy_grad_bias = np.zeros_like(self.dummy_bias)
      self.optimizer.updateParams(self.wrap_Uf, clip(dUf), dummy_grad_bias)
      self.optimizer.updateParams(self.wrap_Ui, clip(dUi), dummy_grad_bias)
      self.optimizer.updateParams(self.wrap_Unc, clip(dUnc), dummy_grad_bias)
      self.optimizer.updateParams(self.wrap_Uo, clip(dUo), dummy_grad_bias)
    else:
      self.update_wo_optimizer()

    if self.bidirectional:
      dx_backward = np.flip(dx_backward, axis=1)
      dx = dx + dx_backward
    
    return dx

  def get_config(self, get_weights: bool, quantize_factor=32):
    configs = [{
      "layer": "LSTM",
      "input_N": self.input_N,
      "hidden_N": self.hidden_N,
      "lr": self.lr,
      "bidirectional": self.bidirectional,
      "use_sequence": self.use_sequence
    }]

    weights = {}
    if get_weights:
      weight_names = [
        "Wf", "Uf", "bf",
        "Wi", "Ui", "bi",
        "Wnc", "Unc", "bnc",
        "Wo", "Uo", "bo",
        "Wci", "bci"
      ]
      for name in weight_names:
        original_array = getattr(self, name)
        quantized_val, scale = quantize_value(original_array, quantize_factor)
        weights[name] = quantized_val
        if scale is not None:
          weights[f"{name}_scale"] = scale
      if self.bidirectional:
        bwd_weights = self.backward_module.get_config(get_weights=True, quantize_factor=quantize_factor)[1]
        
        for key, val in bwd_weights.items():
          weights[f"bwd_{key}"] = val

    return configs, weights
  
  @classmethod
  def load_module(cls, configs, weights, file_name=None):
    optimizer = Optimizer()


    cfg = configs[0]
    bidirectional = cfg.get("bidirectional", False)
    use_sequence = cfg.get("use_sequence", False)
    instance = cls(hidden_N=cfg["hidden_N"], input_N=cfg["input_N"], 
      optimizer=optimizer, bidirectional=bidirectional, use_sequence=use_sequence)
    instance.lr = cfg.get("lr", 0.001) # Default to 0.001 if missing

    try:
      instance.Wf = weights["Wf"]
      instance.Uf = weights["Uf"]
      instance.bf = weights["bf"]

      instance.Wi = weights["Wi"]
      instance.Ui = weights["Ui"]
      instance.bi = weights["bi"]

      instance.Wnc = weights["Wnc"]
      instance.Unc = weights["Unc"]
      instance.bnc = weights["bnc"]

      instance.Wo = weights["Wo"]
      instance.Uo = weights["Uo"]
      instance.bo = weights["bo"]

      instance.Wci = weights["Wci"]
      instance.bci = weights["bci"]

      if bidirectional:
          
        instance.backward_module.Wf = weights["bwd_Wf"]
        instance.backward_module.Uf = weights["bwd_Uf"]
        instance.backward_module.bf = weights["bwd_bf"]

        instance.backward_module.Wi = weights["bwd_Wi"]
        instance.backward_module.Ui = weights["bwd_Ui"]
        instance.backward_module.bi = weights["bwd_bi"]

        instance.backward_module.Wnc = weights["bwd_Wnc"]
        instance.backward_module.Unc = weights["bwd_Unc"]
        instance.backward_module.bnc = weights["bwd_bnc"]

        instance.backward_module.Wo = weights["bwd_Wo"]
        instance.backward_module.Uo = weights["bwd_Uo"]
        instance.backward_module.bo = weights["bwd_bo"]

        instance.backward_module.Wci = weights["bwd_Wci"]
        instance.backward_module.bci = weights["bwd_bci"] 

    except KeyError as e:
      raise KeyError(f"[LSTM Load]: Missing weight in save file: {e}")

    return instance
  
  def get_optimizer(self):
    return self.optimizer

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  def save_optimizer_state(self, filename):
    if self.optimizer:
      self.optimizer.SaveOptimizer(filename, self.optimizer_wrappers)
      if self.bidirectional:
        self.backward_module.optimizer.SaveOptimizer("bwd_" + filename,
        self.backward_module.optimizer_wrappers)

  def load_optimizer_state(self, filename):
    if self.optimizer:
      self.optimizer = Adam.LoadOptimizer(filename, self.optimizer_wrappers)
      if self.bidirectional:
        self.backward_module.optimizer = Adam.LoadOptimizer("bwd_" + filename, 
        self.backward_module.optimizer_wrappers)


class BranchModule(Module):
  def __init__(self):
    self.branch_modules = []
    self.output_sizes = []

  def add_branch(self, module):
    self.branch_modules.append(module)

  def forward(self, input, b_training):
    self.outputs = []
    self.output_sizes = []

    for module in self.branch_modules:
      out = module.forward(input, b_training)
      self.outputs.append(out)
      self.output_sizes.append(out.shape[1])

    return np.concatenate(self.outputs, axis=1)

  def backward(self, grad):
        
    current_idx = 0
    input_grads = 0
        
    for i, module in enumerate(self.branch_modules):

      feat_size = self.output_sizes[i]
      grad_slice = grad[:, current_idx : current_idx + feat_size]
      current_idx += feat_size

      branch_grad = module.backward(grad_slice)
      input_grads += branch_grad
            
    return input_grads
        
  def get_optimizer(self):
    return MultiOptimizer(self.branch_modules)

  def get_config(self, get_weights):
    return [{"layer": "BranchingModule", "branches_count": len(self.branch_modules)}], {}
  
  @classmethod
  def load_module(cls, configs, weights, file_name=None):
    print("Warning: BranchModule loaded. Ensure branches are added manually before loading weights if using strict loading.")
    print("Breanches to load:", configs[0].get("branches_count"))
    return cls()
  
  def save_optimizer_state(self, filename):
    multi_opt = MultiOptimizer(self.branch_modules)
    multi_opt.save_optimizer_state(filename)


class MultiOptimizer:
  """Helper to step multiple optimizers at once"""
  def __init__(self, modules):
    self.modules = modules
    
  def step(self):
    for m in self.modules:
      opt = m.get_optimizer()
      if opt: 
        opt.step()

  def save_optimizer_state(self, filename):
    for i, m in enumerate(self.modules):
      m.save_optimizer_state(f"{filename}_branch_{i}")
  
  def load_optimizer_state(self, filename):
    for i, m in enumerate(self.modules):
      m.load_optimizer_state(f"{filename}_branch_{i}")
