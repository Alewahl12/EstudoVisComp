import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'

    #imagem original
    img_original = cv.imread(image_path)
    cv.imshow('Imagem Original', img_original)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Converter imagem para escala de cinza
    imagem_cinza = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    cv.imshow('Imagem Cinza', imagem_cinza)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Aplicar thresholding binario
    _, img_threshold = cv.threshold(imagem_cinza, 127, 255, cv.THRESH_BINARY)
    cv.imshow('Imagem com Thresholding', img_threshold)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Aplicar thresholding adaptativo
    img_adaptativo = cv.adaptiveThreshold(imagem_cinza, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow('Imagem com Thresholding Adaptativo', img_adaptativo)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Aplicar region growing
    #Criar seed inicial
    seed = (200, 200)  # Coordenadas (x, y) do ponto inicial

    #Criar máscara para a região e aplicar crescimento de região
    mascara = np.zeros_like(imagem_cinza)
    mascara[seed] = 255

    #Crescimento de regiao
    for i in range(1, 5):
        mascara = cv.dilate(mascara, None, iterations=1)
        mascara[imagem_cinza < 200] = 255  # Ajuste de limite para crescimento

    # Exibir a imagem segmentada
    img_segmentada = cv.bitwise_and(img_original, img_original, mask=mascara)
    cv.imshow('Imagem Segmentada por Crescimento de Regiao', img_segmentada)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Exibir imagens lado a lado
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img_threshold, cmap='gray')
    plt.title('Thresholding Binario')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_adaptativo, cmap='gray')
    plt.title('Thresholding Adaptativo')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(img_segmentada, cv.COLOR_BGR2RGB))
    plt.title('Segmentacao por Crescimento de Regiao')
    plt.axis('off')
    
    plt.suptitle('Segmentacao de Imagem')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()