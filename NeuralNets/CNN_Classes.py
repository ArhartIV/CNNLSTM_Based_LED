import numpy as np

class Layer:
  '''Base Layer class that all other layers inherit from. Contains the basic structure and function definitions of a layer
     Contains no functionality and is not meant to be instantiated on its own'''
  def __init__(self):
    self.input = None
    self.output = None

  def forwardProp(self, input, b_training=True):
    ''' Performs the forward pass through the layer
        Parameters:
          - input: The input data to the layer, can be any shape depending on the layer type and initialization parameters.
          - b_training: a boolean flag indicating the use of the training mode
        returns:
          - the output of the layer after applying all calculations and transformations. Shape and type depend on the layer type and initialization parameters.'''
    
    pass
  
  def backwardProp(self, gradient, optimizer):
    ''' Performs the backward pass through the layer and updates the parameters using the provided optimizer. calculates the gradient to be passed further.
        Parameters:
          - gradient: The incoming gradient from the previous(counting backwards) layer(or loss function). Can be any shape depending on the layer type.
          - optimizer: The optimizer object that is used to update the parameters of the layer. (See: Adam class below)
        returns: 
          - the gradient to be passed to the next layer in the backward pass. Shape and type depend on the layer type'''
        
    pass

  def getConfig(self):
    ''' Returns a dictionary containing the configuration of the layer such as initialization parameters and internal variables. 
        Example return: {"layer": "Dense", "input_size": self.input_size, "output_size": self.output_size}'''
    pass

  def getParams(self): 
    ''' Returns the dictionary containing the parameters(usually weights and biases) for saving the model. 
        Example return: {"W": self.weights, "b": self.biases}'''
    pass

  def SetParams(self, weights):
    ''' Sets the parameters of the layer from the provided dictionary. Used for loading the model.
        Parameters:
          - weights: A dictionary containing the parameters of the layer. Expected to be in the same format as the output of getParams()'''
    pass


# Dense Layer Class
class Dense(Layer):
  def __init__(self, input_size, output_size, initialization_type = "XG"):
    ''' Initializes a fully connected dense layer with the given input and output sizes. the weights are initialized using either Xavier/Glorot or He initialization methods
        Parameters:
          - input_size: tuple, the size of the data that will be forwarded through the layer using the forwardProp.
          - output_size: tuple, the size of the data that is expected to be returned by the layer
          - initialization_type: string, either "XG" or "HE" for Xavier/Glorot or He initialization respectively. Use "XG" for layers with sigmoid or tanh functions
          and "HE" for layers with any LU functions. Default is "XG" '''
    self.input_size = input_size
    self.output_size = output_size

    #Chose some random Initialization methods.
    if initialization_type == "XG":
      limit = np.sqrt(6 / input_size)
    elif initialization_type == "HE":
      limit = np.sqrt(2 / input_size)
    else :
      print(f"Unknown initialization type {initialization_type}, defaulting to Xavier/Glorot")
      limit = np.sqrt(6 / input_size)


    self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
    self.bias = np.random.randn(output_size, 1)
  
  def forwardProp(self, input, b_training=True):
    self.original_shape = input.shape
    self.input = input 

    if input.ndim == 3:
      B, H, W = input.shape
      input_reshaped = input.reshape(-1, W)
      out = np.dot(input_reshaped, self.weights.T) + self.bias.reshape(1, -1)
      return out.reshape(B, H, self.output_size)

    else:
      return np.dot(input, self.weights.T) + self.bias.reshape(1, -1)
  
  def backwardProp(self, gradient, optimizer):
    if gradient.ndim == 3:
      B, H, W = gradient.shape

      grad_reshaped = gradient.reshape(-1, W)
      input_reshaped = self.input.reshape(-1, self.input_size)
  
      weights_grad = np.dot(grad_reshaped.T, input_reshaped)
      bias_grad = np.sum(grad_reshaped, axis=0, keepdims=True).T
      optimizer.updateParams(self, weights_grad, bias_grad)

      grad_input = np.dot(grad_reshaped, self.weights)

      return grad_input.reshape(B, H, self.input_size)
    
    else:
      weights_grad = np.dot(gradient.T, self.input)
      bias_grad = np.sum(gradient, axis=0, keepdims=True).T

      optimizer.updateParams(self, weights_grad, bias_grad)

      return np.dot(gradient, self.weights)
  
  def getConfig(self):
    return {"layer": "Dense", "input_size": self.input_size, "output_size": self.output_size}
  
  def getParams(self):
    return {"W": self.weights, "b": self.bias}   

  def setParams(self, weights):
    self.weights = weights["W"]
    self.bias = weights["b"] 
  

''' 
Below are Convolve and Reshape Layers.
They extend the model from NN to CNN
'''


class Convolve(Layer):
  # kernel_N denotes the Size in each Dimension. e.g. (N x N) Matrix
  def __init__(self, input_shape, kernel_N, kernel_count, stride = 1, padding=0):
    ''' Initializes a convolution with the given kernels and input shape. The kernels are initialized using Xavier/Glorot initialization.
        Parameters:
          - input_shape: tuple, shape of the incoming data
          - kernel_N: int or a tuple, size of the convolutional kernel
          - kernel_count: int, number of kernels/filters, consequentially - depth of the output
          - stride: int or tuple. Defaults to 1.
          - padding: int or tuple. Amount of padding added to both sides. Defaults to 0.'''
    if isinstance(kernel_N, (tuple, list)):
      self.kernel_height, self.kernel_width = kernel_N
    else: 
      self.kernel_height = self.kernel_width = kernel_N

    if isinstance(stride, (tuple, list)):
      self.stride_height, self.stride_width = stride
    else:
      self.stride_height = self.stride_width = stride

    if isinstance(padding, (tuple, list)):
      self.padding_height, self.padding_width = padding
    else:
      self.padding_height = self.padding_width = padding

    self.kernel_count = kernel_count
    self.input_shape = input_shape
    self.kernel_N = kernel_N
    input_depth, input_height, input_width = input_shape
    self.input_depth = input_depth
    self.stride = stride
    self.padding = padding

    # (Size of kernel) x Depth of input x Amount of Kernels in layer
    self.kernel_shape = (kernel_count, input_depth, self.kernel_height, self.kernel_width)

    # (Size of input + 2 * padding - Size of  Output + 1) x Amount of Kernels
    self.output_shape = (kernel_count,
      (input_height + 2*self.padding_height - self.kernel_height) //self.stride_height + 1,
      (input_width + 2*self.padding_width - self.kernel_width) // self.stride_width + 1)


    size_in = input_depth * self.kernel_height * self.kernel_width
    size_out = kernel_count * self.kernel_height * self.kernel_width
    limit = np.sqrt(6 / size_in)
    self.kernels = np.random.uniform(-limit, limit, size=self.kernel_shape)
    self.biases = np.zeros((kernel_count, 1, 1))


  #Similar function to the one I used in Pooling layer
  @staticmethod
  def kernelIndices(height, width, kernel_height, kernel_width, stride_h, stride_w):
    row_starts = np.arange(0, height - kernel_height + 1, stride_h)
    col_starts = np.arange(0, width - kernel_width + 1, stride_w)
    grid_r, grid_c = np.meshgrid(row_starts, col_starts, indexing="ij")

    row_offsets = np.arange(kernel_height).reshape(-1, 1)
    col_offsets = np.arange(kernel_width).reshape(1, -1)

    rows = grid_r.reshape(-1, 1, 1) + row_offsets
    cols = grid_c.reshape(-1, 1, 1) + col_offsets
    return rows, cols 
  
  @staticmethod
  def handle_padding(input, padding_h, padding_w):
    if padding_h > 0 or padding_w > 0:
      return np.pad(input, ((0,0), (0,0), (padding_h, padding_h), (padding_w, padding_w)), mode='constant')
    return input

  def forwardProp(self, input, b_training=True):
    self.input = input
    input = self.handle_padding(input, self.padding_height, self.padding_width)
    self.padded_input = input
    stride_height = self.stride_height
    stride_width = self.stride_width

    batch_size, input_depth, input_height, input_width = input.shape
    output_height, output_width = self.output_shape[1:]
    self.output = np.zeros((batch_size, self.kernel_count, output_height, output_width,))

    rows, cols = self.kernelIndices(input_height, input_width,
      self.kernel_height, self.kernel_width, stride_height, stride_width)
    num_windows = rows.shape[0]

    patches = input[:, :, rows, cols]
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(batch_size * num_windows, -1)

    kernels_flat = self.kernels.reshape(self.kernel_count, -1)
    out = patches @ kernels_flat.T + self.biases.ravel()

    out = out.reshape(batch_size, num_windows, self.kernel_count).transpose(0, 2, 1)
    self.output = out.reshape(batch_size, self.kernel_count, output_height, output_width)
    return self.output
  
  def backwardProp(self, gradient, optimizer):
    batch_size = gradient.shape[0]
    input_depth, input_height, input_width = self.input_shape
    pad_height = input_height + 2 * self.padding_height
    pad_width = input_width + 2 * self.padding_width
    stride_height = self.stride_height
    stride_width = self.stride_width
    
    kernel_count, kernel_depth, kernel_height, kernel_width = self.kernels.shape
    if not hasattr(self, 'padded_input'):
      if self.padding_height > 0 or self.padding_width > 0:
        self.padded_input = np.pad(self.input, 
          ((0,0), (0,0), (self.padding_height, self.padding_height), (self.padding_width, self.padding_width)), 
          mode='constant')
      else:
        self.padded_input = self.input

    rows, cols = self.kernelIndices(pad_height, pad_width, self.kernel_height, self.kernel_width, stride_height, stride_width)
    num_windows = rows.shape[0]

    patches = self.padded_input[:, :, rows, cols]
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches_flat = patches.reshape(batch_size * num_windows, -1)
    
    grad_trans = gradient.transpose(0, 2, 3, 1).reshape(batch_size * num_windows, kernel_count)

    kernels_grad_flat = grad_trans.T @ patches_flat
    kernels_grad = kernels_grad_flat.reshape(self.kernels.shape)
    kernels_flat = self.kernels.reshape(kernel_count, -1)
    patches_grad_flat = grad_trans @ kernels_flat
    patches_grad = patches_grad_flat.reshape(batch_size, num_windows, input_depth, kernel_height, kernel_width)

    input_grad_padded = np.zeros((batch_size, input_depth, pad_height, pad_width), dtype=gradient.dtype)
    patches_grad_t = patches_grad.transpose(0, 2, 1, 3, 4)

    for r in range(kernel_height):
      for c in range(kernel_width):
        grad_slice = patches_grad_t[:, :, :, r, c]
        r_global = rows[:, r, 0] 
        c_global = cols[:, 0, c]
        np.add.at(input_grad_padded, (slice(None), slice(None), r_global, c_global), grad_slice)

    if self.padding_height > 0:
        h_slice = slice(self.padding_height, -self.padding_height)
    else:
        h_slice = slice(None)

    if self.padding_width > 0:
        w_slice = slice(self.padding_width, -self.padding_width)
    else:
        w_slice = slice(None)

    input_grad = input_grad_padded[:, :, h_slice, w_slice]

    bias_grad = np.sum(gradient, axis=(0, 2, 3)).reshape((kernel_count, 1, 1))

    optimizer.updateParams(self, kernels_grad, bias_grad)
    return input_grad
  
  def getConfig(self):
    return {"layer": "Convolution", "input_shape": self.input_shape, "kernel_N": self.kernel_N,
      "kernel_count": self.kernel_count, "stride": self.stride, "padding":self.padding
      }

  def getParams(self):
    return {"W": self.kernels, "b": self.biases}

  def setParams(self, weights):
    self.kernels = weights["W"]
    self.biases = weights["b"]


'''Reshape Layer needed to pass and reshape output 
from Convolve Layer to Dense layer'''

class Reshape:
  def __init__(self, input_shape, output_shape):
    self.input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
    self.output_shape = tuple(output_shape) if isinstance(output_shape, list) else output_shape

  def forwardProp(self, input, b_training=True):
    self.input = input
    batch_size = input.shape[0]
    return input.reshape((batch_size,) + self.output_shape)

  def backwardProp(self, gradient, _):
    batch_size = gradient.shape[0]
    return gradient.reshape((batch_size,) + self.input_shape)
  
  def getConfig(self):
    return {"layer": "Reshape", "input_shape": self.input_shape, "output_shape": self.output_shape}

  def getParams(self):
    return None

  def setParams(self, weights):
    pass
  
class Permute(Layer):
  def __init__(self, axes):
    self.axes = axes
        
  def forwardProp(self, input, b_training=True):
    self.input_shape = input.shape
    return input.transpose(self.axes)
        
  def backwardProp(self, gradient, _):
    return gradient.transpose(self.axes)

  def getConfig(self): return {"layer": "Permute", "axes": self.axes}
  def getParams(self): return None
  def setParams(self, p): pass
    
''' 
Below Are Activation Layer classes and all that inherit from it
They are to be used as Activation Functions.
Currently has 
- Sigmoid
- LeakyReLU
- Softmax

'''
 #Activation Layer Class
class Activation(Layer):
  def __init__(self, activ_f, activ_deriv):
    self.activ_f = activ_f
    self.activ_deriv = activ_deriv
  
  def forwardProp(self, input, b_training=True):
    self.input = input
    return self.activ_f(input)
  
  def backwardProp(self, gradient, optimizer):
    return np.multiply(gradient, self.activ_deriv(self.input))

# Leaky ReLU function
class LeakyReLU(Activation):
  def __init__(self):
    LReLU = lambda x: np.maximum(0.1 * x, x)
    LReLU_deriv = lambda x: np.where(x > 0, 1, 0.1)
    super().__init__(LReLU, LReLU_deriv)

  def getConfig(self):
    return {"layer": "LeakyReLU"}

  def getParams(self):
    return None

  def setParams(self, weights):
    pass

# Sigmoid Function 
class Sigmoid(Activation):
  def __init__(self):
    def sigmoid_f(x):
      return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(x):
      s = sigmoid_f(x)
      return s * (1 - s)
    
    super().__init__(sigmoid_f, sigmoid_deriv)

  def forwardProp(self, input, b_training=True):
    self.output = 1 / (1 + np.exp(-input))  # cache output
    return self.output

  def backwardProp(self, gradient, optimizer):
    return gradient * (self.output * (1 - self.output))
  
class Tanh(Activation):
  def __init__(self):
    def tanh_f(x):
      return np.tanh(x)
    def tanh_deriv(x):
      return 1 - np.tanh(x)**2
    super().__init__(tanh_f, tanh_deriv)
    
class Softmax(Activation):
  def __init__(self):
    softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    softmax_deriv = lambda x: np.ones_like(x)
    super().__init__(softmax, softmax_deriv)
  def forwardProp(self, input, b_training=True):
    e_x = np.exp(input - np.max(input, axis=1, keepdims=True))
    self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
    return self.output
    
  def backwardProp(self, gradient, optimizer):
    return gradient
  
  def getConfig(self):
    return {"layer": "Softmax"}
  
  def getParams(self):
    return None

  def setParams(self, weights):
    pass

# 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³))) + 0.5x * sech²(√(2/π) * (x + 0.044715 * x³)) * (√(2/π) * (1 + 3 * 0.044715 * x²))
class GELU(Activation):
  def __init__(self):
    super().__init__(lambda x: x, lambda x: x) 
    self.SQRT_VAL = np.sqrt(2 / np.pi)
    self.COEFF = 0.044715

  def forwardProp(self, input, b_training=True):
    self.input = input
    inner_part = self.SQRT_VAL * (input + self.COEFF * np.power(input, 3))
    self.tanh_inner = np.tanh(inner_part)
    return 0.5 * input * (1 + self.tanh_inner)

  def backwardProp(self, gradient, optimizer):
    x = self.input
    tanh_val = self.tanh_inner
    term1 = 0.5 * (1 + tanh_val)
    inner_deriv = self.SQRT_VAL * (1 + 3 * self.COEFF * np.power(x, 2))
    sech_squared = 1 - np.square(tanh_val)
    term2 = 0.5 * x * sech_squared * inner_deriv
    return np.multiply(gradient, (term1 + term2))

  def getConfig(self):
    return {"layer": "GELU"}

  def getParams(self):
    return None

  def setParams(self, weights):
    pass


'''Below Are additional Layer classes that further help imporve the performance and reduce
the computational cost of the model
- Batch Normalization Layer (Now used for both Dense and Convolutional Layers)
- Admam Optimizer Class
- Pooling Layer'''

class BatchNorm(Layer):
  def __init__(self, dim, momentum=0.9, epsilon=1e-5, is_conv=False):
    self.is_conv = is_conv
    if is_conv:
      self.weights = np.ones((1, dim, 1, 1))
      self.bias = np.zeros((1, dim, 1, 1))
      self.running_mean = np.zeros((1, dim, 1, 1))
      self.running_var = np.ones((1, dim, 1, 1))
    else:
      self.weights = np.ones((1, dim))
      self.bias = np.zeros((1, dim))
      self.running_mean = np.zeros((1, dim))
      self.running_var = np.ones((1, dim))
    
    self.dim = dim
    self.epsilon = epsilon
    self.momentum = momentum

  def forwardProp(self, input, b_training=True):
    self.input = input

    if self.is_conv:
      axes = (0, 2, 3)
      if b_training:
        self.batch_mean = np.mean(input, axis=axes, keepdims=True)
        self.batch_var = np.var(input, axis=axes, keepdims=True)

        self.input_hat = (input - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        out = self.weights * self.input_hat + self.bias

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
      else:
        input_hat = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        out = self.weights * input_hat + self.bias
      return out

    else:
      if b_training:
        self.batch_mean = np.mean(input, axis=0, keepdims=True)
        self.batch_var = np.var(input, axis=0, keepdims=True)

        self.input_hat = (input - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        out = self.weights * self.input_hat + self.bias

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
      else:
        input_hat = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        out = self.weights * input_hat + self.bias
      return out

  def backwardProp(self, grad, optimizer):
    if self.is_conv:
      N, C, H, W = grad.shape
      axes = (0, 2, 3)
      batch_size = N * H * W
      gamma_grad = np.sum(grad * self.input_hat, axis=axes, keepdims=True)
      beta_grad  = np.sum(grad, axis=axes, keepdims=True)

      dx_hat = grad * self.weights

      var_grad = np.sum(dx_hat * (self.input - self.batch_mean) * -0.5 *
        (self.batch_var + self.epsilon) ** -1.5, axis=axes, keepdims=True)
      mean_grad = np.sum(dx_hat * -1 / np.sqrt(self.batch_var + self.epsilon),
        axis=axes, keepdims=True) + \
        var_grad * np.mean(-2 * (self.input - self.batch_mean), axis=axes, keepdims=True)

      input_grad = dx_hat / np.sqrt(self.batch_var + self.epsilon) + \
          var_grad * 2 * (self.input - self.batch_mean) / batch_size + \
          mean_grad / batch_size

      optimizer.updateParams(self, gamma_grad, beta_grad)
      return input_grad
    else:

      batch_size = self.input.shape[0]
      gamma_grad = np.sum(grad * self.input_hat, axis=0, keepdims=True)
      beta_grad  = np.sum(grad, axis=0, keepdims=True)
      dx_hat = grad * self.weights
      var_grad = np.sum(dx_hat * (self.input - self.batch_mean) * -0.5 *
        (self.batch_var + self.epsilon) ** -1.5, axis=0, keepdims=True)
      mean_grad = np.sum(dx_hat * -1 / np.sqrt(self.batch_var + self.epsilon),
        axis=0, keepdims=True) + \
        var_grad * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)
      input_grad = dx_hat / np.sqrt(self.batch_var + self.epsilon) + \
        var_grad * 2 * (self.input - self.batch_mean) / batch_size + \
        mean_grad / batch_size

      optimizer.updateParams(self, gamma_grad, beta_grad)
      return input_grad

  def getConfig(self):
    return {
      "layer": "BatchNorm",
      "dim": self.dim,
      "momentum": self.momentum,
      "epsilon": self.epsilon,
      "is_conv": self.is_conv
    }

  def getParams(self):
    return {
      "weights": self.weights,
      "bias": self.bias,
      "running_mean": self.running_mean,
      "running_var": self.running_var
    }

  def setParams(self, weights):
    if self.is_conv:
      self.weights = np.array(weights["weights"], dtype=np.float64).reshape(1, self.dim, 1, 1)
      self.bias = np.array(weights["bias"], dtype=np.float64).reshape(1, self.dim, 1, 1)
      self.running_mean = np.array(weights["running_mean"], dtype=np.float64).reshape(1, self.dim, 1, 1)
      self.running_var = np.array(weights["running_var"], dtype=np.float64).reshape(1, self.dim, 1, 1)
    else:
      self.weights = np.array(weights["weights"], dtype=np.float64).reshape(1, self.dim)
      self.bias = np.array(weights["bias"], dtype=np.float64).reshape(1, self.dim)
      self.running_mean = np.array(weights["running_mean"], dtype=np.float64).reshape(1, self.dim)
      self.running_var = np.array(weights["running_var"], dtype=np.float64).reshape(1, self.dim)


class SpatialAttribution(Layer):
  def __init__(self, dims, rows, columns):
    if (dims < 2):
      raise ValueError("Dimensions of the input must be at least 2 for Spatial Atribution layer")
    self.dims = dims
    self.rows = rows
    self.columns = columns 

    self.W_rows = np.random.randn(rows, rows) * 0.01
    self.b_rows = np.ones((rows, 1))

    self.W_cols = np.random.randn(columns, columns) * 0.01
    self.b_cols = np.ones((columns, 1))

    self.wrap_Wr = AdamWrapper(self.W_rows, self.b_rows)     
    self.wrap_Wc = AdamWrapper(self.W_cols, self.b_cols)

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

  def forwardProp(self, input, b_training=True):

    row_means = np.mean(input, axis=-1, keepdims=True)
    col_means = np.mean(input, axis=-2, keepdims=True)

    row_attributions = np.matmul(self.W_rows, row_means) + self.b_rows
    col_attributions = np.matmul(col_means, self.W_cols.T) + self.b_cols.T

    row_attributions_sig = self.sigmoid(row_attributions)
    col_attributions_sig = self.sigmoid(col_attributions)

    attention_grid = (row_attributions_sig * col_attributions_sig)
    output = input + (input * attention_grid)

    if b_training:
      self.input_cache = input
      self.row_means_cache = row_means
      self.col_means_cache = col_means
      self.row_attributions_cache = row_attributions_sig
      self.col_attributions_cache = col_attributions_sig
      self.attention_grid_cache = attention_grid

    return output
  
  def backwardProp(self, gradient, optimizer):
    dx = gradient * (1 + self.attention_grid_cache)
    dAg = gradient * self.input_cache

    d_row_sig = np.sum(dAg * self.col_attributions_cache, axis=-1, keepdims=True)
    d_col_sig = np.sum(dAg * self.row_attributions_cache, axis=-2, keepdims=True)

    d_row_z = d_row_sig * self.sigmoid_deriv(self.row_attributions_cache)
    d_col_z = d_col_sig * self.sigmoid_deriv(self.col_attributions_cache)

    batch_axes = tuple(range(gradient.ndim - 2))

    dW_rows = np.sum(np.matmul(d_row_z, self.row_means_cache.swapaxes(-1, -2)), axis=batch_axes)
    db_rows = np.sum(d_row_z, axis=batch_axes + (-1,)).reshape(self.rows, 1)

    dW_cols = np.sum(np.matmul(self.col_means_cache.swapaxes(-1, -2), d_col_z), axis=batch_axes)
    db_cols = np.sum(d_col_z, axis=batch_axes + (-2,)).reshape(self.columns, 1)

    d_row_means = np.matmul(self.W_rows.T, d_row_z)
    d_col_means = np.matmul(d_col_z, self.W_cols)

    dx += d_row_means / self.columns
    dx += d_col_means / self.rows

    optimizer.updateParams(self.wrap_Wr, dW_rows, db_rows)
    optimizer.updateParams(self.wrap_Wc, dW_cols, db_cols)

    return dx  
  
  def getConfig(self):
    return {
      "layer": "SpatialAttribution",
      "dims": self.dims,
      "rows": self.rows,
      "columns": self.columns
    }

  def getParams(self):
    return {
      "W_rows": self.W_rows,
      "b_rows": self.b_rows,
      "W_cols": self.W_cols,
      "b_cols": self.b_cols
    }

  def setParams(self, params):
    self.W_rows = np.array(params["W_rows"], dtype=np.float64).reshape(self.rows, self.rows)
    self.b_rows = np.array(params["b_rows"], dtype=np.float64).reshape(self.rows, 1)
    
    self.W_cols = np.array(params["W_cols"], dtype=np.float64).reshape(self.columns, self.columns)
    self.b_cols = np.array(params["b_cols"], dtype=np.float64).reshape(self.columns, 1)  


class GlobalAvgPooling(Layer):
  def __init__(self, axis=1):
    if isinstance(axis, (list, tuple)):
      self.axis = tuple(axis)
    else:
      self.axis = (axis,)


  def forwardProp(self, input, b_training=True):
    self.input_shape = input.shape
    return np.mean(input, axis=self.axis)

  def backwardProp(self, gradient, optimizer):
    pool_size = 1
    for ax in self.axis:
      pool_size *= self.input_shape[ax]
    
    target_shape = list(self.input_shape)
    for ax in self.axis:
      target_shape[ax] = 1
    
    grad_expanded = np.reshape(gradient, target_shape)
    grad_expanded = np.broadcast_to(grad_expanded, self.input_shape)

    return grad_expanded / pool_size

  def getConfig(self):

    return {"layer": "GlobalAvgPooling", "axis": self.axis}

  def getParams(self):
    return {}

  def setParams(self, params):
    pass

class GlobalMaxPooling(Layer):
  def __init__(self, axis=1):
    if isinstance(axis, (list, tuple)):
      self.axis = axis[0] 
    else:
      self.axis = axis


  def forwardProp(self, input, b_training=True):
    self.input_shape = input.shape
    self.max_indices = np.expand_dims(np.argmax(input, axis=self.axis), self.axis)
    return np.max(input, axis=self.axis)

  def backwardProp(self, gradient, optimizer):
    grad_expanded = np.zeros(self.input_shape)
    grad_reshaped = np.expand_dims(gradient, self.axis)
    np.put_along_axis(grad_expanded, self.max_indices, grad_reshaped, axis=self.axis)
    return grad_expanded

  def getConfig(self):

    return {"layer": "GlobalMaxPooling", "axis": self.axis}

  def getParams(self):
    return {}

  def setParams(self, params):
    pass

class CoupledPooling(Layer):
  def __init__(self, axis=1):
    if not isinstance(axis, int):
      raise ValueError(f"Coupled Pooling Layer only works with one 1 axis, got {axis}")
    
    self.axis = axis
    self.global_avg_pool = GlobalAvgPooling(axis)
    self.global_max_pool = GlobalMaxPooling(axis)
  
  def forwardProp(self, input, b_training=True):
    return np.concatenate((self.global_avg_pool.forwardProp(input, b_training),
                           self.global_max_pool.forwardProp(input, b_training)), axis=-1)
  
  def backwardProp(self, gradient, optimizer):
    grad_avg, grad_max = np.split(gradient, 2, axis=-1)
    res_grad = (self.global_avg_pool.backwardProp(gradient=grad_avg, optimizer=optimizer) +
                self.global_max_pool.backwardProp(gradient=grad_max, optimizer=optimizer))
    return res_grad
  
  def getConfig(self):
    return {"layer": "CoupledPooling", "axis": self.axis}
  
  def getParams(self):
    return {}
  
  def setParams(self, params):
    pass


class Dropout(Layer):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate
    self.mask = None

  def forwardProp(self, input, b_training=False):
    self.b_training = b_training
    self.input = input
    if self.b_training:
      self.mask = (np.random.rand(*input.shape) >= self.rate).astype(input.dtype)
      return (input * self.mask) / (1.0 - self.rate)
    else:
      return input

  def backwardProp(self, gradient, optimizer):
    if self.b_training:
      return (gradient * self.mask) / (1.0 - self.rate)
    else:
      return gradient

  def getConfig(self):
    return {"layer": "Dropout", "rate": self.rate}

  def getParams(self):
    return {}

  def setParams(self, weights):
    pass


class Optimizer:
  def __init__(self):
    pass
  
  def updateParams(self, layer, dW, db):
    pass
  
  def step(self):
    pass

  def getConfig(self):
    pass

  def SaveOptimizer(self, filename, layers):
    pass

  @staticmethod
  def LoadOptimizer(filename, layers, reset=False):
    pass



# Optimizer Adam Class
class Adam(Optimizer):
  def __init__(self, lr, b1, b2, epsilon=1e-7, weight_decay=0.0):
    self.lr = lr
    self.b1 = b1
    self.b2 = b2
    self.epsilon = epsilon
    self.weight_decay = weight_decay
    self.m = {}
    self.v = {}
    self.t = 0

    self.T_max = 1000
    self.eta_min = lr * 0.01
    self.restart = False
    self.lr_t = lr

  def updateParams(self, layer, dW, dB):
    # determine attributes
    if hasattr(layer, "weights"):
      weight_attr = "weights"
    elif hasattr(layer, "kernels"):
      weight_attr = "kernels"
    else:
      raise ValueError("Layer has no weight attribute")

    if hasattr(layer, "bias"):
      bias_attr = "bias"
    elif hasattr(layer, "biases"):
      bias_attr = "biases"
    else:
      raise ValueError("Layer has no bias attribute")

    if layer not in self.m:
      self.m[layer] = {"W": np.zeros_like(dW), "B": np.zeros_like(dB)}
      self.v[layer] = {"W": np.zeros_like(dW), "B": np.zeros_like(dB)}

    # apply L2 weight decay to gradient (dL/dW += lambda * W)
    W = getattr(layer, weight_attr)
    
    #if self.weight_decay:
      #dW = dW + self.weight_decay * W
    t = self.t + 1
    self.m[layer]["W"] = self.b1 * self.m[layer]["W"] + (1 - self.b1) * dW
    self.m[layer]["B"] = self.b1 * self.m[layer]["B"] + (1 - self.b1) * dB
    self.v[layer]["W"] = self.b2 * self.v[layer]["W"] + (1 - self.b2) * (dW ** 2)
    self.v[layer]["B"] = self.b2 * self.v[layer]["B"] + (1 - self.b2) * (dB ** 2)

    mW_hat = self.m[layer]["W"] / (1 - self.b1 ** t)
    mB_hat = self.m[layer]["B"] / (1 - self.b1 ** t)
    vW_hat = self.v[layer]["W"] / (1 - self.b2 ** t)
    vB_hat = self.v[layer]["B"] / (1 - self.b2 ** t)

    B = getattr(layer, bias_attr)
    if self.weight_decay:
      W[...] -= self.lr_t * (mW_hat / (np.sqrt(vW_hat) + self.epsilon) + self.weight_decay * W)
    else:
      W[...] -= self.lr_t * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
    B[...] -= self.lr_t * mB_hat / (np.sqrt(vB_hat) + self.epsilon)

  def step(self):
    if self.restart:
      t_cur = (self.t - 1) % self.T_max
    else:
      t_cur = min(self.t - 1, self.T_max)
    self.lr_t = self.eta_min + 0.5 * (self.lr - self.eta_min) * (1 + np.cos(np.pi * (t_cur / self.T_max))) 

    self.t += 1


  def getConfig(self):
    return {
      "lr": self.lr,
      "b1": self.b1,
      "b2": self.b2,
      "epsilon": self.epsilon,
      "weight_decay": self.weight_decay,
      "t": self.t,
      "T_max": self.T_max,
      "eta_min": self.eta_min,
      "restart": self.restart,
      "lr_t": self.lr_t
    }
  
  def SaveOptimizer(self, filename, layers):
    config = self.getConfig()
    m_to_save, v_to_save = {}, {}

    for i, layer in enumerate(layers):
      if layer in self.m:
        m_to_save[str(i)] = {
          "W": self.m[layer]["W"].tolist(),
          "B": self.m[layer]["B"].tolist()
        }
        v_to_save[str(i)] = {
          "W": self.v[layer]["W"].tolist(),
          "B": self.v[layer]["B"].tolist()
        }
    
    load = {"config": config, "m": m_to_save, "v": v_to_save}

    with open(filename + ".json", "w") as f:
      import json
      json.dump(load, f)
  
  def Sett(self, t):
    self.t = t

  @staticmethod
  def LoadOptimizer(filename, layers, reset=False):
    import json
    with open(filename + ".json", "r") as f:
      load = json.load(f)
    config = load["config"]
    
    optimizer = Adam(config["lr"], config["b1"], config["b2"], config["epsilon"], config.get("weight_decay", 0.0))
    optimizer.T_max = config.get("T_max", 2000)
    optimizer.eta_min = config.get("eta_min", 0.0001)
    optimizer.restart = config.get("restart", False) 
    optimizer.Sett(config["t"])


    for i, layer in enumerate(layers):
      if str(i) in load["m"]:
        optimizer.m[layer] = {
          "W": np.array(load["m"][str(i)]["W"]),
          "B": np.array(load["m"][str(i)]["B"])
        }
        optimizer.v[layer] = {
          "W": np.array(load["v"][str(i)]["W"]),
          "B": np.array(load["v"][str(i)]["B"])
        }

    if reset:
      optimizer.lr_t = optimizer.lr
    else:
      optimizer.lr_t = config.get("lr_t", optimizer.lr)

    return optimizer

class AdamWrapper:
  def __init__(self, weights_reffs, biases_reffs):
    self.weights = weights_reffs
    self.bias = biases_reffs



class Pooling(Layer):
  def __init__(self, pool_size, stride, padding=(0,0), mode='max'):
    self.pool_size = pool_size
    self.stride = stride
    self.padding = padding
    self.mode = mode # 'max' or 'average'
  
  @staticmethod
  def slidingWindowIndices(height, width, pool_size, s):

    if isinstance(s, (tuple, list)):
      stride_height, stride_width = s
    else:
      stride_height = stride_width = s

    pool_height, pool_width = pool_size
    row_starts = np.arange(0, height - pool_height + 1, stride_height)
    col_starts = np.arange(0, width - pool_width + 1, stride_width)
    grid_r, grid_c = np.meshgrid(row_starts, col_starts, indexing="ij")

    row_offsets = np.arange(pool_height).reshape(-1,1)
    col_offsets = np.arange(pool_width).reshape(1,-1)

    rows = grid_r.reshape(-1,1,1) + row_offsets
    cols = grid_c.reshape(-1,1,1) + col_offsets

    return rows, cols

  def forwardProp(self, input, b_training=True):
    pool_height, pool_width = self.pool_size 
    
    #Ability to hadle different strides for dimensions
    if isinstance(self.stride, (tuple, list)):
      stride_height, stride_width = self.stride
    else:
      stride_height = stride_width = self.stride


    self.input = input
    batch_size, input_depth, input_height, input_width = input.shape

    pad_height, pad_width = self.padding
    if pad_height > 0 or pad_width > 0:
      padded = np.zeros((batch_size, input_depth,
        input_height + 2*pad_height,
        input_width + 2*pad_width))
      padded[:, :, pad_height:pad_height + input_height,
        pad_width:pad_width + input_width] = input
    else:
      padded = input

    self.padded_input = padded
    batch_size, input_depth, input_height, input_width = padded.shape

    output_height = (input_height - pool_height) // stride_height + 1
    output_width = (input_width - pool_width) // stride_width + 1
    
    output = np.zeros((batch_size, input_depth, output_height, output_width))

    rows, cols = self.slidingWindowIndices(input_height, input_width, self.pool_size, self.stride)
    self.indicies = (rows, cols)

    windows = padded[:, :, rows, cols]  

    if self.mode == 'max':
      pooled = windows.max(axis=(3,4))
    elif self.mode == 'average':
      pooled = windows.mean(axis=(3,4))
    else:
      raise ValueError("Unsupported pooling mode: {}".format(self.mode))
    output = pooled.reshape(batch_size, input_depth, output_height, output_width)

    self.output = output
    return output

  def backwardProp(self, gradient, optimizer):
    batch_size, depth, input_height, input_width = self.input.shape
    pad_height, pad_width = self.padding
    pool_height, pool_width = self.pool_size

    padded_input = self.padded_input.copy()
    padded_grad = np.zeros_like(self.padded_input)
    rows, cols = self.indicies
    
    windows = padded_input[:, :, rows, cols]

    output_height = gradient.shape[2]
    output_width = gradient.shape[3]

    if self.mode == 'max':
      b, c, n, ph, pw = windows.shape
      flat_windows = windows.reshape(b, c, n, ph * pw)
      max_indices = np.argmax(flat_windows, axis=-1, keepdims=True)
      flat_mask = np.zeros_like(flat_windows)
      np.put_along_axis(flat_mask, max_indices, 1.0, axis=-1)
      mask = flat_mask.reshape(windows.shape)
      grad_expanded = gradient.reshape(batch_size, depth, output_height*output_width, 1, 1)
      input_windows = mask * grad_expanded
    elif self.mode == 'average':
      grad_share = 1.0 / (pool_height * pool_width)
      grad_expanded = gradient.reshape(batch_size, depth, output_height*output_width, 1, 1)
      input_windows = np.ones_like(windows) * grad_share * grad_expanded
    else:
      raise ValueError("Unsupported pooling mode")

    for idx in range(output_height * output_width):
      r = rows[idx]
      c = cols[idx]
      padded_grad[:, :, rows[idx], cols[idx]] += input_windows[:, :, idx]

    if pad_height or pad_width:
      output_grad = padded_grad[:, :, pad_height:pad_height+input_height,
        pad_width:pad_width+input_width]
    else:
      output_grad = padded_grad

    return output_grad
  
  def getConfig(self):
    return {"layer": "Pooling", "pool_size": self.pool_size, "stride": self.stride,
      "padding": self.padding, "mode": self.mode
      }
  
  def getParams(self):
    return None
  
  def setParams(self, weights):
    pass


''' 
Functions to calculate the Error a.k.a Loss Function(and its derivative)
Currently has
- Mean root square
- Cross entropy
'''

class Module:
  def __init__(self):
    pass
  def pass_forward(self):
    return None

  def handle_data(self, *args):
    return None


class CrossEntropyLoss(Module):
  def __init__(self, weights_dictionary):
    self.weights_dictionary= weights_dictionary

    if weights_dictionary is not None:
      num_classes = max(weights_dictionary.keys()) + 1
      self.weights_arr = np.ones(num_classes, dtype=float)
      for c, w in weights_dictionary.items():
        self.weights_arr[c] = w if w > 0 else 1e-6

  def pass_forward(self, y_true, y_pred):
    return self.handle_data(y_true, y_pred)
  

  def handle_data(self, y_true, y_pred):
    if self.weights_dictionary:
      y_weighted = y_true * self.weights_arr

      weights_sum = np.sum(y_weighted, axis=1)
      cross_entr = -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)
      cross_entr =np.sum(cross_entr * weights_sum) / np.sum(weights_sum)
      cross_entr_deriv = (y_pred - y_true) * weights_sum[:, None] / np.sum(weights_sum)
    else:
      cross_entr = -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
      cross_entr_deriv = (y_pred - y_true) / y_true.shape[0]

    return cross_entr, cross_entr_deriv

    
    

def meanSError(expected_output, actual_output):
  return np.mean(np.power(expected_output - actual_output, 2))

def MSEDeriv(expected_output, actual_output):
  return 2 * (actual_output - expected_output) / expected_output.size
  
def cross_entropy(y_true, y_pred):
  return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def cross_entropy_deriv(y_true, y_pred):
  return (y_pred - y_true)