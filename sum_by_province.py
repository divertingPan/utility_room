# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:00:24 2020

@author: panpy
"""

import xlrd
import xlwt

xlsx = xlrd.open_workbook('研究生基础名单（计算分布用）.xlsx')

table1 = xlsx.sheet_by_index(0) # 全日制学生
table2 = xlsx.sheet_by_index(1) # 非全日制学生

all_data = []

for n in range(1, table1.nrows):
    apartment = table1.cell(n, 1).value
    number = table1.cell(n, 4).value
    position = table1.cell(n, 6).value
    data = {'学院': apartment, '学号': str(number), '位置': position}
    all_data.append(data)
for n in range(1, table2.nrows):
    apartment = table2.cell(n, 1).value
    number = table2.cell(n, 4).value
    position = table2.cell(n, 6).value
    data = {'学院': apartment, '学号': str(number), '位置': position}
    all_data.append(data)

all_apartment = []
for data in all_data:
    all_apartment.append(data['学院'])
all_apartment = list(set(all_apartment))

all_grade = []
for data in all_data:
    all_grade.append(data['学号'][0:3])
all_grade = list(set(all_grade))

all_province = ['北京市', '天津市', '上海市', '重庆市', '河北省', '山西省',
                '辽宁省', '吉林省', '黑龙江省', '江苏省', '浙江省', '安徽省',
                '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省',
                '广东省', '海南省', '四川省', '贵州省', '云南省', '陕西省', 
                '甘肃省', '青海省', '台湾省', '内蒙古自治区', '广西壮族自治区', 
                '西藏自治区', '宁夏回族自治区', '新疆维吾尔自治区', 
                '香港特别行政区', '澳门特别行政区', '其他']


writebook = xlwt.Workbook()
sheet = writebook.add_sheet('sheet1')

item_id = 1
sheet.write(0, 0, '院系')
sheet.write(0, 1, '年级')
sheet.write(0, 2, '省市')
sheet.write(0, 3, '人数')

for apartment in all_apartment:
    print('正在计算', apartment)
    for grade in all_grade:
        j = 0
        print('查询', grade, '信息')    
        for province in all_province:
            print('查询', province, '信息')
            temp = 0
            for data in all_data:
                if data['学院'] == apartment \
                            and data['位置'][0:2] == province[0:2] \
                            and data['学号'][0:3] == grade[0:3]:
                    temp = temp + 1
                    j = j + 1
                    
            if province == '其他':
                apartment_temp = [data for data in all_data \
                                   if data['学院'] == apartment \
                                   and data['学号'][0:3] == grade[0:3]]
                temp = len(apartment_temp) - j
            sheet.write(item_id, 0, apartment)
            sheet.write(item_id, 1, grade)
            sheet.write(item_id, 2, province)
            sheet.write(item_id, 3, temp)
            item_id = item_id + 1
            
# writebook.save('全日制.xls')
# writebook.save('非全日制.xls')
writebook.save('全体.xls')