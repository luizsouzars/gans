import os
import torch
from tqdm import tqdm
from utils.visualize import plot_generated_images, plot_gradients, plot_losses

def get_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients.append(param.grad.view(-1).cpu().numpy())
    return gradients

def save_model(model, epoch, path='models', filename='generator_best.pth'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))

def train_gan(netG, netD, dataloader, criterion, optimizerG, optimizerD, nz, epochs, image_size, device, save_interval):
    fixed_noise = torch.randn(64, nz, device=device)
    G_losses = []
    D_losses = []
    best_g_loss = float('inf')  # Inicializando com infinito
    best_epoch = 0

    for epoch in range(epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for i, data in enumerate(dataloader, 0):
                # Atualizar Discriminador
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), 1., dtype=torch.float, device=device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, device=device)
                fake = netG(noise)
                label.fill_(0.)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # Atualizar Gerador
                netG.zero_grad()
                label.fill_(1.)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                # Atualizar barra de progresso
                pbar.set_postfix({
                    'Loss_D': f'{errD.item():.4f}',
                    'Loss_G': f'{errG.item():.4f}',
                })
                pbar.update(1)

            # Geração de imagens para visualização e salvamento em intervalos definidos
            if (epoch + 1) % save_interval == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                plot_generated_images(fake, epoch + 1)

                # Coletar e plotar gradientes
                gradientsD = get_gradients(netD)
                gradientsG = get_gradients(netG)

                plot_gradients(gradientsD, epoch + 1, output_dir='gradients/discriminator')
                plot_gradients(gradientsG, epoch + 1, output_dir='gradients/generator')

            # Armazenar perdas para plotar depois
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Salvar o melhor modelo do gerador
            if errG.item() < best_g_loss:
                best_g_loss = errG.item()
                best_epoch = epoch
                save_model(netG, epoch + 1, filename='generator_best.pth')

    # Plotar perdas após o treinamento
    plot_losses(G_losses, D_losses, best_epoch, best_g_loss)
