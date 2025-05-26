import cv2 as cv
import matplotlib.pyplot as plt


def main():
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'

    # Carregar imagem original
    img_original = cv.imread(image_path)
    cv.imshow('Imagem Original', img_original)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Converter imagem para escala de cinza
    img_cinza = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.imshow('Imagem Cinza', img_cinza)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Aplicar filtro gaussian blur (perda de informação)
    img_blur = cv.GaussianBlur(img_original, (25, 25), 0)
    cv.imshow("Imagem com blur", img_blur)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Imagem com equalização de histograma
    imagem_equalizada = cv.equalizeHist(img_cinza)
    cv.imshow("Imagem Equalizada", imagem_equalizada)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Exibir imagens lado a lado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(img_cinza, cmap='gray')
    plt.title('Imagem Cinza')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(cv.cvtColor(img_blur, cv.COLOR_BGR2RGB))
    plt.title('Imagem com Blur')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(imagem_equalizada, cmap='gray')
    plt.title('Imagem Equalizada')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()