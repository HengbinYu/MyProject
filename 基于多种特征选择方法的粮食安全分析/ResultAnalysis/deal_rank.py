import xlwt
# f = open('ReducedFeatures_FRAR2.txt', 'r', encoding='utf-8-sig')
f = open('ReducedFeatures_FRAR2.txt', 'r')
dic = {}
for line in f:
    line = line.strip().split()
    for i in range(1, len(line)):
        if line[i] not in dic:
            dic[line[i]] = {}
        dic[line[i]][line[0]] = i
f.close()
list_data = ['河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '江苏省', '安徽省', '山东省', '河南省', '湖北省', '重庆市', '四川省', '贵州省', '云南省', '陕西省', '甘肃省', '宁夏回族自治区', '新疆维吾尔自治区']
list_pro = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '台湾省', '香港特别行政区', '澳门特别行政区', 'None']
workbook = xlwt.Workbook(encoding='utf-8')    # 创建一个workbook 设置编码
worksheet = workbook.add_sheet('Sheet', cell_overwrite_ok=True)    # 创建一个worksheet
worksheet.write(0, 0, label='省份')

for i in range(len(list_pro)):
    worksheet.write(i+1, 0, label=list_pro[i])
m = 0
for i in dic:
    print(len(dic[i]))
    m += 1
    worksheet.write(0, m, label=i)
    n = 0
    for j in range(len(list_pro)):
        if list_pro[j] in dic[i]:
            worksheet.write(j+1, m, label=11 - dic[i][list_pro[j]])  # 11 -
        elif list_pro[j] in list_data:
            worksheet.write(j + 1, m, label=0)
        else:
            worksheet.write(j+1, m, label='null')
workbook.save(f"Features_Color.xls")  # 保存
