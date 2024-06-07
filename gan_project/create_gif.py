import os
import argparse
from PIL import Image

def create_gif(image_folder, gif_path, duration=500):
    # Lista de arquivos no diretório
    files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Lista para armazenar as imagens
    images = []

    # Abrir cada imagem e adicioná-la à lista
    for file in files:
        img_path = os.path.join(image_folder, file)
        img = Image.open(img_path)
        images.append(img)

    # Salvar as imagens como um GIF animado
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    print(f'GIF criado com sucesso: {gif_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cria um GIF a partir de imagens PNG em um diretório.')
    parser.add_argument('image_folder', type=str, help='Caminho para o diretório que contém as imagens')
    parser.add_argument('gif_path', type=str, help='Caminho para salvar o GIF')
    parser.add_argument('--duration', type=int, default=500, help='Duração de cada frame no GIF em milissegundos (padrão: 500ms)')

    args = parser.parse_args()

    create_gif(args.image_folder, args.gif_path, args.duration)