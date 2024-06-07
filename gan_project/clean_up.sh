#!/bin/bash

# Diretórios a serem limpos
GENERATED_IMAGES_DIR="generated_images"
GRADIENTS_DIR="gradients"
MODELS_DIR="models"
SYNTHETIC_IMAGES_DIR="synthetic_images"
LOSSES_DIR="losses"
OUTPUTS="outputs"

# Função para excluir arquivos específicos em um diretório
delete_files() {
    if [ -d "$1" ]; then
        echo "Limpando arquivos PNG, GIF e PTH no diretório $1..."
        find "$1" -type f \( -name "*.png" -o -name "*.gif" -o -name "*.pth" \) -exec rm -v {} \;
        echo "Arquivos em $1 limpos com sucesso."
    else
        echo "Diretório $1 não encontrado."
    fi
}

# Limpar arquivos nos diretórios
delete_files $GENERATED_IMAGES_DIR
delete_files $GRADIENTS_DIR
delete_files $MODELS_DIR
delete_files $SYNTHETIC_IMAGES_DIR
delete_files $LOSSES_DIR
delete_files $OUTPUTS

echo "Limpeza concluída."
