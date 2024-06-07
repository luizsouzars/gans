#!/bin/bash

# Limpar o terminal
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    cls
else
    clear
fi

# Parâmetros de treinamento
BATCH_SIZE=64
EPOCHS=50
SAVE_INTERVAL=10

# Verifica argumentos de linha de comando e os passa para o script Python
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --save_interval) SAVE_INTERVAL="$2"; shift ;;
    esac
    shift
done

# Treinamento do modelo
echo "Iniciando o treinamento do modelo com batch_size=${BATCH_SIZE}, epochs=${EPOCHS}, save_interval=${SAVE_INTERVAL}..."
python main.py --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --save_interval ${SAVE_INTERVAL}

# Verificando se o modelo foi treinado com sucesso
if [ $? -eq 0 ]; then
    echo "Treinamento concluído com sucesso."

    # Gerando imagens sintéticas
    echo "Gerando imagens sintéticas..."
    python generate_images.py

    # Criando GIF das imagens geradas durante o treinamento
    echo "Criando GIF das imagens geradas durante o treinamento..."
    python create_gif.py generated_images outputs/generated_images.gif --duration 500

    # Criando GIF dos gradientes do discriminador durante o treinamento
    echo "Criando GIF dos gradientes do discriminador durante o treinamento..."
    python create_gif.py gradients/discriminator outputs/discriminator_gradients.gif --duration 500

    # Criando GIF dos gradientes do gerador durante o treinamento
    echo "Criando GIF dos gradientes do gerador durante o treinamento..."
    python create_gif.py gradients/generator outputs/generator_gradients.gif --duration 500

else
    echo "Erro no treinamento do modelo. Abortando a geração de imagens."
fi
