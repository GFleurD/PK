import onnx

model = onnx.load("pk.onnx")
for tensor in model.graph.initializer:
    print(tensor.name, tensor.data_location)  # should all be 0
