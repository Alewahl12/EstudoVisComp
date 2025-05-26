import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'

    # Imagem original
    img_original = cv.imread(image_path)

    #Converter imagem para escala de cinza
    imagem_cinza = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    
    #Sobel
    sobel_x = cv.Sobel(imagem_cinza, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(imagem_cinza, cv.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv.magnitude(sobel_x, sobel_y)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(imagem_cinza, cmap='gray')
    plt.title('Imagem Cinza')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Bordas com Sobel')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    #prewitt
    kernel_prewitt_x = np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]])
    
    kernel_prewitt_y = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [-1, -1, -1]])
    
    prewitt_x = cv.filter2D(imagem_cinza, cv.CV_64F, kernel_prewitt_x)
    prewitt_y = cv.filter2D(imagem_cinza, cv.CV_64F, kernel_prewitt_y)
    prewitt_combined = cv.magnitude(prewitt_x, prewitt_y)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(imagem_cinza, cmap='gray')
    plt.title('Imagem Cinza')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(prewitt_combined, cmap='gray')
    plt.title('Bordas com Prewitt')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    #Canny
    img_canny = cv.Canny(imagem_cinza, 100, 200)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(imagem_cinza, cmap='gray')
    plt.title('Imagem Cinza')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_canny, cmap='gray')
    plt.title('Bordas com Canny')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()