import torch
import torchvision.models as models
import time
import numpy as np

mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
resnet152 = models.resnet152(pretrained=True)


tensor = torch.randn((16,3,3,3))



res = [ ]
iters=10
start = time.time()
for i in range(iters):
    output1 = mobilenet_v3_large(tensor)
end = time.time()

res.append(end - start)
print('Timing mobile:', res[0])

start = time.time()
for i in range(iters):
    output2 = resnet152(tensor)
end = time.time()

res.append(end - start)
print('Timing resnet:', res[1])

print(f"Times faster: {np.max(res)/np.min(res)}")

