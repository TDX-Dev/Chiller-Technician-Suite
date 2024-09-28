import torch.onnx
from pumpFrequencyModel import PumpFrequencyModel
import pandas
import onnx

features = ['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW']
targets = ['CH Load']


device = torch.device('cuda')

model = PumpFrequencyModel(features=len(features), targets=len(targets)).to(device)



model.load_state_dict(torch.load("./pump_freq.pt"))


model.eval()

# After training the model
dummy_input = torch.randn(1, len(features)).to(device)  # Create a dummy input tensor

# Export the model
onnx_file_path = "./pump_frequency_model.onnx"
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_file_path, 
                  export_params=True,
                  opset_version=11,  # Make sure to use an appropriate ONNX opset version
                  do_constant_folding=True,  # Optimize for inference
                  input_names=['input'],  # Name of the input layer
                  output_names=['output'],  # Name of the output layer
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # Dynamic batch size
