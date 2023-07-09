import cv2
import os

top, right, bottom, left = 25, 25, 225, 225

exit_con='**'

a=''

dir0='test_data'

# get the reference to the webcam
camera = cv2.VideoCapture(0)
while(True):

    a=input('exit: ** or enter the label name : ')

    if a==exit_con:
        break

    dir1=str(dir0)+'/'+str(a)
    print(dir1)

    try:
        os.mkdir(dir1)
    except:
        print(dir1)

    i=0
    pre_presskey = False
    # phase = 200
    while(True):
        (t, frame) = camera.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # get the crop
        crop = frame[top:bottom, right:left]

        # write img file to directory
        print(i)
        if i>250:
            # phase = 100
            break
        # if(i >= phase):
        #     pre_presskey = False
        #     phase += 100
        #     continue
        # draw the segmented hand
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.imshow("Video Feed 1", crop)

        cv2.imshow("Video Feed", frame)

        # observe the keypress by the user, press "Q" to take a photo one image from small frame
        pressKey = cv2.waitKey(1)
        if pressKey == ord('q') or pre_presskey:
            cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),crop)
            i+=1
            pre_presskey = True
            continue

# free up memory
camera.release()
cv2.destroyAllWindows()