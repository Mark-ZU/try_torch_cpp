import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, gate):
        super(MyCell, self).__init__()
        self.gate = gate
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.gate(self.linear(x)) + h)
        # new_h = torch.tanh((self.linear(x)) + h)
        return new_h, new_h

scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
x = torch.rand(3, 4)
h = torch.rand(3, 4)
scripted_cell = torch.jit.trace(my_cell, (x, h))

saved_name = "scripted_cell.pt"

scripted_cell.save(saved_name)
loaded = torch.jit.load(saved_name)

# print(scripted_cell.graph)
# print(scripted_gate.code)
# print(scripted_cell.code)

# print(scripted_cell)
# print(scripted_cell(x, h))

print(loaded)
print(loaded.code)
print(loaded(x,h))