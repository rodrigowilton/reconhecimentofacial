import cv2
import numpy as np

arquivo_modelo = 'weights/res10_300x300_ssd_iter_140000.caffemodel'
arquivo_prontotxt = 'weights/deploy.prototxt.txt'
network = cv2.dnn.readNetFromCaffe(arquivo_prontotxt, arquivo_modelo)

def detecta_face_ssd(net, path_imagem, tamanho = 900, confi_min = 0.5):
	imagem = cv2.imread(path_imagem)
	(h, w) = imagem.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(imagem,(tamanho, tamanho)), 1.0, (tamanho, tamanho), (104.0, 117.0, 123.0))
	net.setInput(blob)
	deteccoes = net.forward()
	
	for i in range(0, deteccoes.shape[2]):
		confianca = deteccoes[0, 0, i, 2]
		
		if confianca > confi_min:
			box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			
			text_confi = "{:.2f}%".format(confianca * 100)
			cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
			cv2.putText(imagem, text_confi, (start_x, start_y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
	cv2.imshow('imagem',imagem)

path_imagem = '../recface/faces/people4.jpg'
detecta_face_ssd(network, path_imagem)

cv2.waitKey(0)
		
		