import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def main():
    #Aplicar ddetecção de cantos usando o método harris-corner e Shi-Tomasi se utilizar cv.imshow, apenas utilizar plt no final
    # Caminho da imagem
    image_path = 'imagemCachorro.jpg'
    img_original = cv.imread(image_path)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    # Harris Corner Detection
    harris_corners = cv.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv.dilate(harris_corners, None)
    img_harris = img_original.copy()
    img_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]  # Marcar cantos em vermelho

    # Shi-Tomasi Corner Detection
    shi_tomasi_corners = cv.goodFeaturesToTrack(img_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    img_shi_tomasi = img_original.copy()
    if shi_tomasi_corners is not None:
        for corner in shi_tomasi_corners:
            x, y = corner.ravel()
            cv.circle(img_shi_tomasi, (int(x), int(y)), 3, (0, 255, 0), -1)  # Marcar cantos em verde
    else:
        print("Nenhum canto detectado com Shi-Tomasi.")

    # Exibir imagens lado a lado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_harris, cv.COLOR_BGR2RGB))
    plt.title('Detecção de Cantos - Harris')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_shi_tomasi, cv.COLOR_BGR2RGB))
    plt.title('Detecção de Cantos - Shi-Tomasi')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    


if __name__ == "__main__":
    main()