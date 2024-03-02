import cv2
import numpy as np
from PIL import Image
import pytesseract

cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture("rtsp://admin:Debiot221501@192.168.1.180:554/Streaming/Channels/101")

ctexto = ''

while True:
    ret, frame = cap.read()
    if  ret == False:
        break

    cv2.rectangle(frame, (870, 750), (1070, 850), (0,0,0), cv2.FILLED)
    cv2.putText(frame,ctexto[0:7], (900,810), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255, 0), 2)

    al, an, c = frame.shape
    
    #en x
    x1 = int(an/11)
    x2 = int(x1 * 5)
    y1 = int(al/ 5)
    y2 = int(y1 * 2)

    #en y
    y1 = int(al/3)
    y2 = int(y1 * 2)

    #texto
    cv2.rectangle(frame, (x1 + 160, y1 + 500), (1120, 940), (0,0,0), cv2.FILLED)
    cv2.putText(frame, 'Procesando Placa', (x1 + 180,y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    recorte = frame[y1:y2, x1:x2]

    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    Color = cv2.absdiff(mG, mB)


    #umbral = None
    _, umbral = cv2.threshold(Color, 200, 255, cv2.THRESH_BINARY)
    #_, umbral = cv2.threshold(Color, 200, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area <5000:
            x, y, ancho, alto = cv2.boundingRect(contorno)
            xpi = x + x1
            ypi = y + y1

            xpf = x + ancho + x1
            ypf = y + alto + y1

            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            placa = frame[ypi:ypf, xpi:xpf]
            alp, anp, cp = placa.shape
            #print(apl,anp)

            Mva = np.zeros((alp, anp))

            mBp = np.matrix(placa[:, :, 0])
            mGp = np.matrix(placa[:, :, 1])
            mRp = np.matrix(placa[:, :, 2])

            #ESTOY CREANDO LA MASCARA 
            for col in range(0, alp):
                for fil in range(0, anp):
                    Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                    Mva[col, fil] = 255 - Max
            
            #aca se binariza la imagen
            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            #convertimor la matriz en la imagen
            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")

            #evaluacion del tamaño de la placa
            if alp >= 36 and anp >= 82:

                pytesseract.pytesseract.tesseract_cmd = r'C:\Users\analista.tic\AppData\Local\Programs\Tesseract-OCR\tesseract'

                #extraccion de texto
                config = "--psm 1"
                texto = pytesseract.image_to_string(bin, config=config)

                if len(texto) >= 7:

                    ctexto = texto

                    #mostramos los valores en la pantalla

                    #cv2.putText(frame, ctexto[0:7], (910,810), cv2.FONT_HERSHEY_SINPLEX, 1, (0, 255, 0),2)

            break
            #aca se va a mostrar el recorte de la placa
            #cv2.imshow("Recorte", bin)
    
    # recorte en gris
    cv2.imshow("Vehiculo", frame)

    #leemos una tecla
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
