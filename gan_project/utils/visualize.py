import matplotlib.pyplot as plt
import numpy as np
import os

# Configurando o estilo e as cores dos plots
plt.style.use('ggplot')
colors = {
    'generator_loss': '#00C853',      # Green
    'discriminator_loss': '#D500F9',  # Purple
    'layer1': '#FFAB00',              # Amber
    'layer2': '#2979FF',              # Blue
    'layer3': '#00E5FF',              # Cyan
    'layer4': '#FF3D00'               # Red
}

def plot_generated_images(images, epoch, output_dir='generated_images'):
    images = (images + 1) / 2  # Desnormalizar as imagens

    # Criar diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar imagem
    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i, 0], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.figtext(0.5, 0.01, f'Epoch: {epoch}', ha='center', fontsize=12, color='red', weight='bold', 
                bbox=dict(facecolor='white'))
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}.png'))
    plt.close(fig)

def plot_gradients(gradients, epoch, output_dir='gradients'):
    # Criar diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots()
    for i, gradient in enumerate(gradients):
        color = colors.get(f'layer{i+1}', None)
        ax.plot(gradient, label=f'Layer {i+1}', color=color)
    ax.legend(loc='lower right')
    ax.set_title(f'Gradients at Epoch {epoch}')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Gradient Value')
    ax.set_ylim([-0.1, 0.1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}.png'))
    plt.close(fig)

def plot_losses(G_losses, D_losses, output_dir='losses'):
    # Criar diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', color=colors['generator_loss'])
    plt.plot(D_losses, label='Discriminator Loss', color=colors['discriminator_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title('Generator and Discriminator Loss During Training')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()
