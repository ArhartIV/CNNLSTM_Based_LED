import numpy as np
import json
from .utils import BatchModifier
from .Modules import Module, CNNModule, LSTMModule, BranchModule
from .CNN_Classes import cross_entropy, cross_entropy_deriv, Module

class Model:
  def __init__(self):
    self.modules_list = []
    self.batch_modifiers = []
    self.module_registry = {
            "CNNModule": CNNModule,
            "LSTMModule": LSTMModule,
            "BranchModule": BranchModule
            }

  def train(self, batch_size, epochs, X, Y, filename=" ", X_val=None, Y_val=None, loss_module:Module=None):
    n_samples = X.shape[0]
    best_val_acc = 0.0

    for epoch in range(epochs):
      indices = np.arange(n_samples)
      np.random.shuffle(indices)
      X, Y = X[indices], Y[indices]

      total_loss = 0.0

      for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        if len(self.batch_modifiers) > 0:
          for modifier in self.batch_modifiers:
            X_batch, Y_batch = modifier.modify_batch(X_batch, Y_batch)
        
        out = X_batch
        for module in self.modules_list:
          out = module.forward(out, True)
        
        if loss_module:
          loss, grad = loss_module.pass_forward(Y_batch, out)
        else:        
          loss = cross_entropy(Y_batch, out)        
          grad = cross_entropy_deriv(Y_batch, out)

        total_loss += loss

        for module in reversed(self.modules_list):
          opt  = module.get_optimizer()
          grad = module.backward(grad)
          if opt:
            opt.step()

      avg_loss = total_loss / (n_samples // batch_size)

      if (epoch+1) % 5 == 0 and filename != " ":
        self.save(filename + "_AutoSave")
        for i, module in enumerate(self.modules_list):
          if hasattr(module, 'save_optimizer_state'):
            module.save_optimizer_state(f"{filename}_{i}_{module.__class__.__name__}_opt_autosave")
            
      print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
      if X_val is not None and Y_val is not None:
        val_out = self.inference(X_val)
        val_preds = np.argmax(val_out, axis=1)
        val_true = np.argmax(Y_val, axis=1)

        val_acc = np.mean(val_preds == val_true) * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_val_acc:
          print(f"New Best Accuracy ({best_val_acc:.2f}% -> {val_acc:.2f}%) Saving...")
          best_val_acc = val_acc

          if filename != " ":
            self.save(filename + "_BEST")


  def inference(self, input):
    output = input
    for module in self.modules_list:
      output = module.forward(output, b_training=False)
    return output
  
  def add_module(self, new_module : Module):
    self.modules_list.append(new_module)


  def add_batch_modifier(self, batch_modifier: BatchModifier):
    self.batch_modifiers.append(batch_modifier)

  def save(self, file_name):
    model_data = {"modules": []}

    for module in self.modules_list:
      module_type = module.__class__.__name__
      configs, weights = module.get_config(get_weights=True)
      weights_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in weights.items()}

      module_data = {
        "type": module_type,
        "configs": configs,
        "weights": weights_json
        }
      model_data["modules"].append(module_data)

    with open(file_name, 'w') as f:
      json.dump(model_data, f, indent=4)
    print(f"Model saved to {file_name}")

  @staticmethod
  def load(file_name):
    new_model = Model()
        
    with open(file_name, 'r') as f:
      data = json.load(f)

    for module_data in data["modules"]:
      module_type = module_data["type"]
      configs = module_data["configs"]
      weights_raw = module_data["weights"]

      weights = {k: np.array(v) for k, v in weights_raw.items()}

      if module_type in new_model.module_registry:
        module_class = new_model.module_registry[module_type]
        loaded_module = module_class.load_module(configs, weights, file_name)
                
        new_model.add_module(loaded_module)
      else:
        print(f"Warning: Unknown module type '{module_type}' found in save file.")

    return new_model