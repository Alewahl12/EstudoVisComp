import cv2 as cv
import matplotlib.pyplot as plt

def main():
    #Caminho da imagem
    image_path = 'imagemCachorro.jpg'

    #Iagem original
    img_original = cv.imread(image_path)
    cv.imshow('Imagem Original', img_original)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Imagem em escala de cinza
    imagem_cinza = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.imshow('Imagem Cinza', imagem_cinza)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Redimensionar imagem
    img_redimensionada = cv.resize(img_original, (700, 500))
    cv.imshow('Imagem Redimensionada', img_redimensionada)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Rotacionar imagem com RotationMatrix2D
    #Obter dimensões da imagem
    altura, largura = img_original.shape[:2]
    centro = (largura // 2, altura // 2)

    #Criar matriz de rotação
    angulo = 45  # Ângulo de rotação em graus
    matriz_rotacao = cv.getRotationMatrix2D(centro, angulo, 1.0)

    #Aplicar rotação
    img_rotacionada = cv.warpAffine(img_original, matriz_rotacao, (largura, altura))
    cv.imshow('Imagem Rotacionada', img_rotacionada)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Aplicar recorte(cropping) na imagem
    #Definir região de interesse (ROI)
    roi = img_original[100:400, 200:500]  # (y1:y2, x1:x2)
    cv.imshow('Regiao de Interesse (ROI)', roi)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #EXIBiR IMAGENS LADO A LADO E ROI no mesmo gráfico
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(imagem_cinza, cmap='gray')
    plt.title('Imagem Cinza')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(cv.cvtColor(img_redimensionada, cv.COLOR_BGR2RGB))
    plt.title('Imagem Redimensionada')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(cv.cvtColor(img_rotacionada, cv.COLOR_BGR2RGB))
    plt.title('Imagem Rotacionada')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
    plt.title('Região de Interesse (ROI)')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()




if __name__ == "__main__":
    main()