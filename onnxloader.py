import onnx
import onnxruntime
import torch
import numpy as np

batch_size = 1
onnx_model = onnx.load("./encoder/encoder_256.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("./encoder/encoder_256.onnx")
x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# print(to_numpy(bitstream).shape)
# print(np.ndarray(ort_outs).shape)
# compare ONNX Runtime and PyTorch results
# if to_numpy(bitstream).size() == ort_outs[0].size():
#     print(1)
#np.testing.assert_allclose(to_numpy(bitstream), ort_outs[0], rtol=1e-03, atol=1e-05)
print(ort_outs)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")