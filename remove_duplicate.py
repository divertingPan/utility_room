'''
https://www.jianshu.com/p/5a7daa2123a8
'''

import cv2
import numpy as np
import os


if __name__ == '__main__':
    path = './anim'
    
    img_list = os.listdir(path)
    hash_list = []
    for ind, img_name in enumerate(img_list):
        print('checking:', ind+1, 'from total:', len(img_list))
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            os.remove(os.path.join(path, img_name))
            print('wrong file')
            continue

        img = cv2.resize(img,(8,8))

        avg_np = np.mean(img)
        img = np.where(img>avg_np,1,0)
        if len(hash_list)<1:
            hash_list.append(img)
        else:
            for i in hash_list:
                flag = True
                dis = np.bitwise_xor(i,img)

                if np.sum(dis) < 5:
                    flag = False
                    os.remove(os.path.join(path, img_name))
                    print('remove')
                    break
            if flag:
                hash_list.append(img)
