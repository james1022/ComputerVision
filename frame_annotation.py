import sys
import numpy as np
import cv2
import cv2.cv as cv
import os

os.chdir('/home/james/Desktop/JamesResearch/EndoscopicVideos')
# os.chdir('/media/james_external_drive/Research_backup/video_files')
video_name = 'video01.mp4'
outfile_name = 'video01_numpy_rough.npy'
outfile_text_name = 'video01_text_output_rough.txt'
video = cv2.VideoCapture(video_name)
length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
coordinates = []
curr = None
count = 1

n_attributes = 10
s = (n_attributes, length)
annotation = np.zeros(s)
nextIndices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def on_mouse_rectangle(event, x, y, flags, params):
    global img, coordinates
    t = time()

    if event == cv.CV_EVENT_LBUTTONDOWN:
        coordinates.append(x)
        coordinates.append(y)

    elif event == cv.CV_EVENT_LBUTTONUP:
        coordinates.append(x)
        coordinates.append(y)

        coordinatesLen = len(coordinates);
        print coordinates
        for i in range(0, coordinatesLen, 4):
            cv2.rectangle(img, (coordinates[i], coordinates[i + 1]), (coordinates[i+2], coordinates[i+3]), (0,255,0), 3)
            cv2.imshow('rectangled', img)
            cv2.imwrite('Rect'+str(t)+'.jpg', img)
            print "Written to file"
        coordinates = []

def onChange(trackbarValue):
    curr = cv2.getTrackbarPos('play', 'mywindow')
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, trackbarValue)

def onPause(trackbarValue): 
    if trackbarValue == 1:
        curr = cv2.getTrackbarPos('play', 'mywindow')
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, curr)
        cv.SetMouseCallback('mywindow', on_mouse_rectangle, 0)
        cv2.imshow("mywindow", img)

def onBlood(trackbarValue):
    annotation[0, nextIndices[0]] = curr
    nextIndices[0] += 1

def onClips(trackbarValue):
    annotation[1, nextIndices[1]] = curr
    nextIndices[1] += 1

def onGallbladder(trackbarValue):
    annotation[2, nextIndices[2]] = curr
    nextIndices[2] += 1

def onLiver(trackbarValue):
    annotation[3, nextIndices[3]] = curr
    nextIndices[3] += 1

def onPancreas(trackbarValue):
    annotation[4, nextIndices[4]] = curr
    nextIndices[4] += 1

def onNeedle(trackbarValue):
    annotation[5, nextIndices[5]] = curr
    nextIndices[5] += 1

def onInjectionTool(trackbarValue):
    annotation[6, nextIndices[6]] = curr
    nextIndices[6] += 1

def onSmoke(trackbarValue):
    annotation[7, nextIndices[7]] = curr
    nextIndices[7] += 1

def onWater(trackbarValue):
    annotation[8, nextIndices[8]] = curr
    nextIndices[8] += 1

def onBag(trackbarValue):
    annotation[9, nextIndices[9]] = curr
    nextIndices[9] += 1


def main(argv):
    
    cv2.namedWindow('mywindow')
    cv2.createTrackbar('play', 'mywindow', 0, length, onChange)
    #cv2.createTrackbar('pause', 'mywindow', 0, 1, onPause)

    cv2.namedWindow('toolbar')
    cv2.createTrackbar('Blood      ', 'toolbar', 0, 1, onBlood) #0
    cv2.createTrackbar('Clips      ', 'toolbar', 0, 1, onClips) #1
    cv2.createTrackbar('Gallbladder', 'toolbar', 0, 1, onGallbladder) #2
    cv2.createTrackbar('Liver      ', 'toolbar', 0, 1, onLiver) #3
    cv2.createTrackbar('Pancreas   ', 'toolbar', 0, 1, onPancreas) #4
    cv2.createTrackbar('Needle     ', 'toolbar', 0, 1, onNeedle) #5
    cv2.createTrackbar('Injection Tool', 'toolbar', 0, 1, onInjectionTool) #6
    cv2.createTrackbar('Smoke      ', 'toolbar', 0, 1, onSmoke) #7
    cv2.createTrackbar('Water      ', 'toolbar', 0, 1, onWater) #8 
    cv2.createTrackbar('Retrieval Bag', 'toolbar', 0, 1, onBag) #9

    onChange(0)
    #onPause(0)
    cv2.waitKey()

    framerate = 7
    play = cv2.getTrackbarPos('play','mywindow')  
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, play)

    print length

    # global img, res_blue, res_gray, curr, annotation
    global curr

    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

    while video.isOpened():

        curr = int(video.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        #video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, curr + framerate)
        
        #cv2.setTrackbarPos('play', 'mywindow', curr)
        #pause = cv2.getTrackbarPos('pause', 'mywindow')

        #if pause == 0:
        
        #for _ in range(framerate - 1):
            #video.read()
        err,img = video.read()
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.resize(img, None, fx = 0.6, fy = 0.6)
        cv2.imshow('mywindow', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#    np.save(outfile_name, annotation)
#    np.savetxt(outfile_text_name, annotation)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
