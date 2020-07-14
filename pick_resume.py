# -*- coding: utf-8 -*-
import pandas as pd
import os
from shutil import copyfile


resume_path = './resume'
menu_path = './新建XLSX工作表.xlsx'
destination_path = './sorted'


if not os.path.exists(destination_path):
    os.makedirs(destination_path)

df = pd.read_excel(menu_path, usecols=[1]) # 以人名列作为检索key

for maindir, subdir, filename in os.walk(resume_path):
    for i, file in enumerate(filename):
        file_no_space = ''.join(file.split())
        for i, key in enumerate(df['姓名']):
            if key in file_no_space:
                copyfile(os.path.join(maindir, file), os.path.join(destination_path, str(i+1)+'_'+file))
                print(file)
