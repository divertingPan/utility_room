"""
reference: https://blog.csdn.net/wangjian1204/article/details/80295335
"""

import dlib
import cv2


img_path = './face_aligned.png'
img = cv2.imread(img_path)
print('img_shape:', img.shape)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# get face
hog_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
 
rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
faces = dlib.full_object_detections()
for rect in rects:
    faces.append(shape_predictor(img_rgb, rect))
 
for landmark in faces:
    landmark_list_x = []
    landmark_list_y = []
    for idx, point in enumerate(landmark.parts()):
        img_output = cv2.circle(img, (point.x, point.y), 2, (255, 0, 0), 2)
        img_output = cv2.putText(img_output, str(idx), (point.x, point.y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        landmark_list_x.append(point.x)
        landmark_list_y.append(point.y)
        
cv2.imshow('imshow', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
