<kbd>老潘的杂物间</kbd>  
---
这里是老潘的杂物间，主要存放一些不成体系的但是偶尔要用的一旦要用却找不到又会挺麻烦的东西。如果顺便能方便到你，那可真是不胜荣幸！  


**下面是各个文件的简要说明**  

* [landmark_extract](https://github.com/divertingPan/utility_room/tree/master/landmark_extract)：导出图片中的人脸关键点，里面附带了检测模型`shape_predictor_68_face_landmarks.dat`
  
* [Hurst_calculate.py](https://github.com/divertingPan/utility_room/blob/master/Hurst_calculate.py)：计算时间序列Hurst指数的核心代码。毕设用的，不知道后面什么时候会不会又用到。包括一部分画图的代码。
  
* [dataloader.py](https://github.com/divertingPan/utility_room/blob/master/dataloader.py)：用pytorch从csv文件中读取数据，例子的数据是对应图片的路径，顺便还带了一个图片预处理和增广，但是少了一步最终的加载，先放在下面
```python
dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True)
```

* [face_detect.py](https://github.com/divertingPan/utility_room/blob/master/face_detect.py)：从一系列视频帧图像中裁剪出人脸。图像应该是已经从视频导出的帧。该程序想要直接运行的话，图片存放的路径格式应该是dataset_path/video_001/00001.jpg、dataset_path/video_001/00002.jpg，另须一个模型文件`shape_predictor_68_face_landmarks.dat`，在[landmark_extract](https://github.com/divertingPan/utility_room/tree/master/landmark_extract)里有。

* [get_labels.py](https://github.com/divertingPan/utility_room/blob/master/get_labels.py)：CK+数据集，每个数据都有一个txt记录这个数据的标签，为了整合label到一个csv里面，使用这个脚本整合，可以简单修改后做其他类似任务。


* [pick_resume.py](https://github.com/divertingPan/utility_room/blob/master/pick_resume.py)：一个根据表格中的信息筛选文件的小工具。表格中是一些人名，从海量简历里面找出这些人的简历并复制出来。

* [plot_lr_rate.py](https://github.com/divertingPan/utility_room/blob/master/plot_lr_rate.py)：里面有几个动态学习率的方法，同时有一个画出学习率的方法。其中也可以通过下面方法获取到目前的学习率值
```python
optimize_g.param_groups[0]['lr']
```

* [remove_duplicate.py](https://github.com/divertingPan/utility_room/blob/master/remove_duplicate.py)：去除数据集中的重复图片和无法读取的图片

* [sum_by_province.py](https://github.com/divertingPan/utility_room/blob/master/sum_by_province.py)：某天的一个临时工程，也许哪天会再用上。可以按照列的信息筛选符合条件的人。不过因为该项目过于简单，所以连正则也没用上。

* [task_2022.ipynb](https://github.com/divertingPan/utility_room/blob/master/task_2022.ipynb)：对应解释的链接为https://divertingpan.github.io/post/quiz_two_cv_tasks/

* [train_cat.py](https://github.com/divertingPan/utility_room/blob/master/train_cat.py)：本身是catGAN里面的一个文件，但是由于结构简单、逻辑清晰、要素充分，非常适合用作pytorch建立网络、训练网络用的整体流程模板。  
内含：超参数的定义调用方法；CUDA加速一行指令；构建网络的Sequential和forwad；数据预处理
