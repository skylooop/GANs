import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,  1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


lr = 3e-4
z_dim = 64
img_dim = 28*28
epochs = 25
disc = Discriminator(img_dim).cuda()
gen = Generator(z_dim, img_dim).cuda()
noise = torch.randn((32, z_dim)).cuda()
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

dataset = datasets.MNIST("data/",transform = transforms, download=True)
loader = DataLoader(dataset,batch_size=32, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)

crit = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/Gan/fake")
writer_real = SummaryWriter(f"runs/Gan/real")
step = 0

for epoch in range(epochs):
    for batch_idx, (imges, _) in enumerate(loader):
        real = imges.view(-1,784).cuda()
        batch_size = real.shape[0]
        #train discr
        noise = torch.randn((batch_size, z_dim)).cuda()
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = crit(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = crit(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake+lossD_real)/2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()

        #train generator
        out = disc(fake).view(-1)
        lossG = crit(out, torch.ones_like(out))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1



