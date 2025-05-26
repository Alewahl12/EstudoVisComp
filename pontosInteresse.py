import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Aplicar detecção de pontos de interesse usando o método SIFT, SURF e ORB respectivamente e mostrar os resultados lado a lado
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'
    img_original = cv.imread(image_path)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    #SIFT
    sift = cv.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(img_gray, None)
    img_sift = cv.drawKeypoints(img_original, keypoints_sift, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #Rich Keypoints para mostrar tamanho e orientação dos keypoints
    
    # SURF (Não incluido no openCV por padrao)

    # ORB
    orb = cv.ORB_create()
    keypoints_orb, descriptors_orb = orb.detectAndCompute(img_gray, None)
    img_orb = cv.drawKeypoints(img_original, keypoints_orb, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   
    # Exibir imagens lado a lado
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    plt.title('Detecção de Pontos de Interesse - SIFT')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_orb, cv.COLOR_BGR2RGB))
    plt.title('Detecção de Pontos de Interesse - ORB')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()