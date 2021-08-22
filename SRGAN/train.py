import torch
import config
from torch import nn
from torch import optim
from utils import plot_examples #load_checkpoint, save_checkpoint
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import ImageDS


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=False)
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        #train discriminator -> max E[log(D(x))] + E[1-log(D(G(z))]
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real) - 0.1*torch.rand_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        #train Generator -> max log(D(G(z)))
        disc_fake = disc(fake)
        adverserial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = adverserial_loss + loss_for_vgg

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        if idx % 200 == 0:
            plot_examples("Data/LR", gen)


def main():
    dataset = ImageDS(root = "Data/")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,

    )
    gen = Generator().to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse,bce,vgg_loss)


if __name__ == "__main__":
    main()




