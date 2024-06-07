import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, image_size*image_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, self.image_size, self.image_size)
