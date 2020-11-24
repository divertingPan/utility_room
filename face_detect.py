"""
reference: https://blog.csdn.net/Lee_01/article/details/89472174
"""

from imutils.face_utils import rect_to_bb, FaceAligner
import dlib
import cv2
import os


dataset_path = 'cohn-kanade-images'
output_path = 'image_cropped'
crop_size = 256

def get_face(fa, image):
    detector = dlib.get_frontal_face_detector()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = gray.shape[0] // 4
    
    rects = detector(gray, 2)
    face_aligned = []
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        # 调用 align 函数对图像中的指定人脸进行处理
        if w > thresh:
            face_aligned = fa.align(image, gray, rect)
       
    return face_aligned


def align_dlib():
    
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=crop_size)
    
    subject_dir = os.listdir(dataset_path)
    for i, sdir in enumerate(subject_dir):
        video_dir = os.listdir(os.path.join(dataset_path, sdir))
        for vdir in video_dir:
            save_path = os.path.join(output_path, sdir, vdir)
            print(save_path)
            os.makedirs(save_path, exist_ok=True)
            img_list = os.listdir(os.path.join(dataset_path, sdir, vdir))
            for img in img_list:
                image = cv2.imread(os.path.join(dataset_path, sdir, vdir, img))
                face_aligned = get_face(fa, image)
                try:
                    cv2.imwrite(os.path.join(save_path, img), face_aligned)
                except:
                    height, width, _ = image.shape
                    a = int(height/2 - crop_size/2)
                    b = int(height/2 + crop_size/2)
                    c = int(width/2 - crop_size/2)
                    d = int(width/2 + crop_size/2)
                    image = image[a:b, c:d]
                    cv2.imwrite(os.path.join(save_path, img), image)

def align_crop():
    subject_dir = os.listdir(dataset_path)
    for i, sdir in enumerate(subject_dir):
        video_dir = os.listdir(os.path.join(dataset_path, sdir))
        for vdir in video_dir:
            save_path = os.path.join(output_path, sdir, vdir)
            print(save_path)
            os.makedirs(save_path, exist_ok=True)
            img_list = os.listdir(os.path.join(dataset_path, sdir, vdir))
            for img in img_list:
                image = cv2.imread(os.path.join(dataset_path, sdir, vdir, img))
                height, width, _ = image.shape
                a = int(height/2 - crop_size/2)
                b = int(height/2 + crop_size/2)
                c = int(width/2 - crop_size/2)
                d = int(width/2 + crop_size/2)
                image = image[a:b, c:d]
                cv2.imwrite(os.path.join(save_path, img), image)

if __name__ == '__main__':
    align_dlib()
