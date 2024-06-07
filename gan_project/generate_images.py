import torch
from models.generator import Generator
import matplotlib.pyplot as plt
import os

def load_model(path, nz, image_size, device):
    model = Generator(nz, image_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def generate_and_save_images(model, nz, num_images, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    noise = torch.randn(num_images, nz, device=model.main[0].weight.device)
    with torch.no_grad():
        fake_images = model(noise).cpu().detach()
    
    fake_images = (fake_images + 1) / 2  # Desnormalizar as imagens

    # Salvar imagens geradas
    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(fake_images[i, 0], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_synthetic_images.png'))
    plt.close(fig)

if __name__ == '__main__':
    nz = 100  # Dimensão do vetor de ruído
    image_size = 28  # Tamanho da imagem gerada
    num_images = 64  # Número de imagens a serem geradas
    model_path = 'models/generator_best.pth'  # Caminho do modelo salvo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar o modelo
    model = load_model(model_path, nz, image_size, device)

    # Gerar e salvar imagens sintéticas
    generate_and_save_images(model, nz, num_images)
