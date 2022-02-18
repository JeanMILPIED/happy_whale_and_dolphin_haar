# happy_whale_and_dolphin_haar
haar cascade file for whale and dolphin kaggle competition  
date: 18/02/2022  

to all competitors, if you are interested in images pre treatment, and particularly to detect fins and crop around fins,  
you'll find attached a haar cascade classifier (.xml) trained on the competition train images + images with absolutely no fins.  

#### Source
I followed this excellent article to use the proper tools to train a haar cascade: https://medium.com/@vipulgote4/guide-to-make-custom-haar-cascade-xml-file-for-object-detection-with-opencv-6932e22c3f0e  

#### Code  
to use the haar cascade filter, you can follow this script:  

```
#here is code to check the output
face_cascade=cv2.CascadeClassifier(r"<your_folder_here>\happy-whale-and-dolphin\haar_cascade\cascade_fins2.xml")###path of cascade file
## following is an test image u can take any image from the p folder in the temp folder and paste address of it on below line 
img= cv2.imread(r"<your_folder_ere>\happy-whale-and-dolphin\train_images\0a100ed55f1c7e.jpg")###path of image file which we want to detect

if img.shape[1] > 1000:
    # resize image
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
else:
    resized=img

#rotation
my_angle=-30
my_center=None
resized_rot= rotate(resized, my_angle, my_center, scale = 1.0)

gray=cv2.cvtColor(resized_rot,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=10, minSize=(10, 10))#try to tune this 6.5 and 17 parameter to get good result 
print(faces)
##if not getting good result try to train new cascade.xml file again deleting other file expect p and n in temp folder

if len(faces)!=0:
    big_w=[face[2]*face[3] for face in faces]
    selected_face=faces[big_w.index(max(big_w))]
    for face in faces:
        x,y,w,h = face
        resized_rot_f=cv2.rectangle(resized_rot,(x,y),(x+w,y+h),(0,255,0),2)
    x,y,w,h = selected_face
    resized_rot_f=cv2.rectangle(resized_rot_f,(x,y),(x+w,y+h),(255,255,0),2)
    plt.imshow(cv2.cvtColor(resized_rot_f, cv2.COLOR_BGR2RGB))
    plt.show()
    print('fin_size = {}%'.format(np.round(w*h/(resized_rot_f.shape[0]*resized_rot_f.shape[1])*100,1)))
else:
    print('no fin detected')
```  

#### Results examples  
![example1](https://myoctocat.com/assets/images/base-octocat.svg)
![example2](https://myoctocat.com/assets/images/base-octocat.svg)

