import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUTS_SIZE = 28*28
BATCH_SIZE = 600


def plot(samples):
    fig = plt.figure(figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


class Dataset():
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        trainData = datasets.MNIST('./mnist_data', train=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(trainData,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=2)

    def trainLength(self):
        return 60000


class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        output_size = 1
        hidden_size = 32

        self.D = nn.Sequential(
            nn.Linear(INPUTS_SIZE, hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_size*4, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_size*2, output_size),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        out = self.D(inputs)
        return out

    def eval(self, inputs):
        pass


class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.z_dim = 100
        hidden_size = 64

        self.G = nn.Sequential(
            nn.Linear(self.z_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size*2),

            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size*4),

            nn.Linear(hidden_size*4, INPUTS_SIZE),
            nn.Tanh()
        )

    def forward(self, dummy_for_summary=None):
        inputs = self.generate_noise((BATCH_SIZE, self.z_dim))
        out = self.G(inputs)

        return out

    def generate_noise(self, size):
        noise = torch.Tensor(size=size).uniform_()
        return noise

    def sample(self):
        with torch.no_grad():
            inputs = self.generate_noise((9, self.z_dim))
            out = self.G(inputs)
            return out

    def eval(self):
        pass


SAVE_SAMPLES = 0

epoches = 100
lossEvery = 10

D = D_net()
G = G_net()
dataloader = Dataset()

print("Discriminator info:")
summary(D, input_size=(1, 28*28))
print("Generator info:")
summary(G, input_size=())
print()

criterion = nn.BCELoss()
D_optimizer = optim.SGD(D.parameters(), lr=0.02)
G_optimizer = optim.SGD(G.parameters(), lr=0.02)


# Train
for epoch in range(epoches):
    running_loss_D = 0.0
    running_loss_G = 0.0
    for i, data in enumerate(dataloader.trainloader, 0):

        inputs, labels = data
        if (inputs.size()[0] != BATCH_SIZE):
            continue


        # zero the parameter gradients
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()


        # inputs and ground truth data
        inputs = torch.reshape(inputs, (inputs.size()[0], 28*28))
        fake_inputs = G()
        valid = torch.ones(BATCH_SIZE)
        fake = torch.zeros(BATCH_SIZE)

        # train G_net
        G_loss = criterion(D(fake_inputs), valid)
        G_loss.backward()
        G_optimizer.step()

        # train D_net
        D_real = D(inputs).squeeze()
        D_fake = D(fake_inputs.detach()).squeeze()

        D_real_loss = criterion(D_real, valid)
        D_fake_loss = criterion(D_fake, fake)
        D_loss = D_real_loss.add(D_fake_loss)
        D_loss.backward()
        D_optimizer.step()

        running_loss_D += D_loss.item()
        running_loss_G += G_loss.item()

        # Save g output
        if i % 10 == 0 and SAVE_SAMPLES:
            samples = torch.reshape(G.sample(), (9, 28, 28)).detach()
            fig = plot(samples)
            plt.savefig('./out/{}.png'.format(str(epoch).zfill(3)+str(i).zfill(4)), bbox_inches='tight')
            plt.close(fig)

        # logging loss
        if i % lossEvery == 0:
            print("E %2d  s %3d     lossD: %.4f    lossG: %.4f" % (epoch, i, running_loss_D / lossEvery, running_loss_G / lossEvery))
            running_loss_D = 0.
            running_loss_G = 0.

    print("Epoch ended")
