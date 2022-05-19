import cv2 as cv
import mediapipe as mp
import os
import sys
import time
def get_training_data(DIR=str, labels=list, size=tuple):
    '''
    Parametros:
    - DIR:    str | Pasta principal com subpastas com videos, 
    - labels: list | Nome das subpastas,
    - size:  tuple | Dimensao destino do arquivo de video.


    HlmsGetter é uma biblioteca auxiliar de IA para fazer a leitura de arquivos de 
    video e obter a posição x, y das marcações na superficie de mão para poder 
    treinar uma rede neural a reconhecer gestos com a mão.
    '''
    #============================Get Metadata================================#
    paths = []
    training_data = []
    class_num = 0

    for l in labels:
        paths.append(os.path.join(DIR, l))
    
        for path in paths:
            for p in os.listdir(path):
                if p[-4:] == '.mp4':
                    training_data.append([os.path.join(path,p), class_num])
            class_num += 1
    #========================================================================#

    #===========================Set the landmarks things=====================#
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    video =  []
    frame =  []
    videos = []
    #========================================================================#

    #========================Configure loading bar===========================#
    videos_readed = 0
    n_videos = len(training_data)/100
    sys.stdout.write('\n')
    time.sleep(2)
    sys.stdout.write('reading videos...\n')
    sys.stdout.write('[{}]'.format(' '*50))
    sys.stdout.write(('\b'*(50)))
    #========================================================================#

    #==========Reading videos and generate 1 label per frame=================#
    labels = []
    for t in training_data:
        
        cap = cv.VideoCapture(t[0])

        while True:
            success, img = cap.read()
            try:
                img = cv.resize(img, size)
            except:
                
                break
            
            if success:
                
                results =  hands.process(img)
                
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:

                        for id, lm in enumerate(handLms.landmark):
                            
                            h,w,c = img.shape
                            cx, cy = int(lm.x*w), int(lm.y*h)
                            lmsid = [id, cx, cy]
                            frame.append(lmsid)
                            labels.append(t[1])
                            #print(f'{lmsid}, label={t[1]}')

    
                
            else:
                break
        videos_readed += 1
            
        
        percentage = int(videos_readed / n_videos)
        sys.stdout.write('\b'*((percentage//2)))
        sys.stdout.write('█'*((percentage//2)))
    videos.append(frame)
        
    return videos, labels





