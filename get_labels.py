# -*- coding: utf-8 -*-
import os
import pandas as pd


"""
    expected form: 
        
    file_path   label
    aligned_images_DB\Markus_Beyer\5   Markus_Beyer
"""


dataset_path = 'cohn-kanade-images'
label_path = 'Emotion'
video_folder = []
labels = []

subject_dir = os.listdir(dataset_path)
for i, sdir in enumerate(subject_dir):
    video_dir = os.listdir(os.path.join(dataset_path, sdir))
    for vdir in video_dir:
        try:
            label_dir = os.path.join(label_path, sdir, vdir)
            os.listdir(label_dir)
        except:
            continue
        if os.listdir(label_dir):
            video_folder.append(os.path.join(sdir, vdir))
            label_file = os.path.join(label_dir, os.listdir(label_dir)[0])
            with open(label_file, "r") as f:
                label_value = f.read()
                label_value = int(float(label_value.split(' ')[3])) - 1
            labels.append(label_value)
        
df = pd.DataFrame({'path':video_folder, 'label':labels})
df = df.sample(frac=1.0)
df.to_csv('./label.csv', index=None)

