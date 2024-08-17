import cv2
import numpy as np

# Caminho relativo correto
imagem = cv2.imread('faces/pexels-fox-1595391.jpg')


# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Não foi possível abrir ou ler o arquivo: faces/pexels-fox-1595391.jpg")
else:
    # CONVERTER PARA ESCALA DE CINZA
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imagem_cinza',imagem_cinza)  # Exibir a imagem em uma janela cinza
    
    # vamos diminuir a imagem
    imagem = cv2.resize(imagem, (0,0), fx = 0.5, fy = 0.5)
    imagem_cinza = cv2.resize(imagem_cinza, (0,0), fx = 0.5, fy = 0.5)
    
    # podemos usar outra opcao
    #imagem = cv2.
    
    # criar a deteccao da face
    detecao_facial = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #melhorar a deteccao da face
    deteccoes = detecao_facial.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    # vamos criar o retangulo na face
    for (x, y, w, h) in deteccoes:
        cv2.rectangle(imagem, (x,y), (x + w, y + h), (0,255,255),2)
        cv2.imshow('imagem', imagem) # sempre mostrando na imagem original
        
        print(w,h)
        
    cv2.waitKey(0)  # Esperar até que uma tecla seja pressionada
    cv2.destroyAllWindows()  # Fechar todas as janelas
