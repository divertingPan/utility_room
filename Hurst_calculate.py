# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt


def Hurst(ts, ftitle): 
    
    n_min, n_max = 0, int(np.log2(len(ts)))
    RSlist = []
    L_temp = []
    for cut in range(n_max-2, n_min, -1):
        children = len(ts) // 2**cut
        children_list = [ts[i*children:(i+1)*children] for i in range(2**cut)]
        L = []
        for a_children in children_list:
            Ma = np.mean(a_children)  #计算平均值
            Xta = Series(map(lambda x: x-Ma, a_children)).cumsum()  #构造均值差累积序列
            Ra = max(Xta) - min(Xta) + 0.00000015  #计算极差
            Sa = np.std(a_children) + 0.0000001  #计算标准差
            rs = Ra / Sa  #计算R/S
            L.append(rs)
        RS = np.mean(L)
        RSlist.append(RS)
        L_temp.append(L)
        
    x = [i for i in range(1, len(RSlist)+1)]
    y = np.log2(RSlist)
    poly = np.polyfit(x, y, 1)
    z = np.polyval(poly, x)

    ''' 输出每段子区间Hurst值 '''
    e = 1
    Logi = []
    Lx = []
    for i in L_temp:
        Logi.extend(np.log2(i))
        Lx.extend([e]*len(i))
        e += 1

    print(poly[0])

    ''' 标准化输出图像 '''
    plt.rcParams['font.sans-serif']=['SimHei']  #如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus']=False  #显示负号
     
    #label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    #color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    #线型：-  --   -.  :    , 
    #marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10,5))
    plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
         
    plt.plot(Lx, Logi, '.', color='orange', label='子区间R/S值')
    plt.plot(x, y, 's', color='blue', label='R/S均值')
    plt.plot(x, z, color='green', label='拟合曲线', linewidth=1.5)

    plt.xticks(x, fontsize=12, fontweight='bold') #默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(ftitle, fontsize=12, fontweight='bold')    #默认字体大小为12
    plt.xlabel("log(n)", fontsize=13, fontweight='bold')
    plt.ylabel("log(R/S)", fontsize=13, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细
    plt.savefig('output/'+ftitle+'.svg', format='svg')  #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()
    
    
def Draw(file_name, figure_title, length):

    df = pd.read_csv(file_name)
    test_series = df['pkg']
    test_series = np.array(test_series)
    series1 = test_series[-length::1]
    series2 = [i+1 for i in series1]
    Hurst(series2, figure_title)
