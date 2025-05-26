import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():

    #Aplicar Transformacoes geometricas como translação, rotação e escala em uma imagem e mostrar os resultados lado a lado e matriz de transformação, perspectiva e homografia
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'
    img_original = cv.imread(image_path)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    # Translação
    M_translacao = np.float32([[1, 0, 100], [0, 1, 50]])
    img_translacao = cv.warpAffine(img_original, M_translacao, (img_original.shape[1], img_original.shape[0]))

    # Rotação
    M_rotacao = cv.getRotationMatrix2D((img_original.shape[1] / 2, img_original.shape[0] / 2), 45, 1)
    img_rotacao = cv.warpAffine(img_original, M_rotacao, (img_original.shape[1], img_original.shape[0]))

    # Escala
    img_escala = cv.resize(img_original, None, fx=1.5, fy=1.5)

    #Perspectiva e homografia
    # Definir pontos de origem e destino para a transformação de perspectiva
    pts_orig = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    pts_dest = np.float32([[10, 10], [220, 30], [30, 220], [210, 210]])
    M_perspectiva = cv.getPerspectiveTransform(pts_orig, pts_dest)
    img_perspectiva = cv.warpPerspective(img_original, M_perspectiva, (img_original.shape[1], img_original.shape[0]))

    # Exibir resultados
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(img_translacao, cv.COLOR_BGR2RGB))
    plt.title('Translação')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv.cvtColor(img_rotacao, cv.COLOR_BGR2RGB))
    plt.title('Rotação')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv.cvtColor(img_escala, cv.COLOR_BGR2RGB))
    plt.title('Escala')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv.cvtColor(img_perspectiva, cv.COLOR_BGR2RGB))
    plt.title('Perspectiva')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Cinza')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


#Perspectiva e homografia sao a mesma coisa
if __name__ == "__main__":
    main()