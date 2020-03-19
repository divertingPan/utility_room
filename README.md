<kbd>老潘的杂物间</kbd>  
---
这里是老潘的杂物间，主要存放一些不成体系的但是偶尔要用的一旦要用却找不到又会挺麻烦的东西。如果顺便能方便到你，那可真是不胜荣幸！  


**下面是各个文件的简要说明**  
  
* Hurst_calculate.py：计算时间序列Hurst指数的核心代码。毕设用的，不知道后面什么时候会不会又用到。
  
* dataloader.py：用pytorch从csv文件中读取数据，例子的数据是对应图片的路径，顺便还带了一个图片预处理和增广，但是少了一步最终的加载，先放在下面
```python
dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True)
```

* plot_lr_rate.py：里面有几个动态学习率的方法，同时有一个画出学习率的方法。其中也可以通过下面方法获取到目前的学习率值
```python
optimize_g.param_groups[0]['lr']
```

* sum_by_province.py：某天的一个临时工程，也许哪天会再用上。可以按照列的信息筛选符合条件的人。不过因为该项目过于简单，所以连正则也没用上。

* train_cat.py：本身是catGAN里面的一个文件，但是由于结构简单、逻辑清晰、要素充分，非常适合用作pytorch建立网络、训练网络用的整体流程模板。  
内含：超参数的定义调用方法；CUDA加速一行指令；构建网络的Sequential和forwad；数据加载和预处理；设置动态学习率；训练；结果输出；模型保存。
