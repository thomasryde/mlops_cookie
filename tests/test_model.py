import sys
sys.path.append('src/models')
from model import MyAwesomeModel
import torch
import pytest
model = MyAwesomeModel()

output = model(torch.randn(1,1,28,28))
assert output.shape == torch.Size([1,10]), "Wrongly implemented model (input/output size)"
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected input to specific shape'):
        model(torch.randn(1,1,29,28))