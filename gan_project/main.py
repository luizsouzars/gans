import torch
import torch.optim as optim
import torch.nn as nn
from data.download_data import get_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from train import train_gan
import argparse

def main(args):
    # Configurações
    batch_size = args.batch_size
    image_size = 28
    nz = 100  # Dimensão do vetor de ruído
    epochs = args.epochs
    lr = 0.0002
    beta1 = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicialização dos modelos
    netG = Generator(nz, image_size).to(device)
    netD = Discriminator(image_size).to(device)

    # Funções de perda e otimizadores
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Carregamento do dataset
    dataloader = get_dataloader(batch_size)

    # Treinamento da GAN
    train_gan(netG, netD, dataloader, criterion, optimizerG, optimizerD, nz, epochs, image_size, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinamento de uma GAN.')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch para treinamento (padrão: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas para treinamento (padrão: 50)')
    
    args = parser.parse_args()
    
    main(args)
