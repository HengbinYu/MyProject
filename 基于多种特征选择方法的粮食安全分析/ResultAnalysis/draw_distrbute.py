import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gp
import matplotlib.colors as colors
import matplotlib.ticker as ticker
Color=['Accent', 'Accent_r', 'Blues', 'Blues_r','BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
       'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r','Greys', 'Greys_r', 'OrRd', 'OrRd_r', \
       'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
       'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples',
       'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
       'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia',
       'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
       'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
       'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
       'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
       'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
       'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot',
       'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral',
       'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r','rainbow',
       'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r','tab20',
       'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
       'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

plt.rcParams['font.family'] = ['SimHei']
FeatureFile='H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\TotalFeatures.xlsx'
fm=pd.read_excel(FeatureFile,sheet_name='FRAR')
Features=fm.iloc[:,0].tolist()
matplotlib.use('TKAgg')
china_map = gp.GeoDataFrame.from_file("Map/China.shp", encoding='utf-8-sig')# 读取中国地图的shp文件,画轮廓
plot_feature=Features[4]
province_color_level = pd.read_excel(r'Features_Color.xls')[plot_feature]  # 读取各省特征分布,按照特征重要性给各省画颜色

# for index in Color:
# index='cubehelix_r'
# index='afmhot_r'
# index='magma_r'
index='YlGnBu'

print(index)
geo_ploy = china_map  # 画各省轮廓图
fig, ax = plt.subplots()
ax.set_aspect('equal')
geo_ploy.plot(ax=ax, color='white', edgecolor='grey', linewidth=0.8)

# 定义ScalarMappable对象
cmap = plt.get_cmap(index)
norm = colors.Normalize(vmin=province_color_level.min(), vmax=province_color_level.max())
scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

geo_ploy.plot(column=province_color_level, ax=ax, cmap=index, edgecolor='grey', linewidth=0.8, vmin=-2, vmax=10)
ax.set_axis_off()  # 轴设置不可见
plt.title(plot_feature,fontdict={'weight': 'bold', 'size': 15})
# 添加颜色条

formatter = lambda x, pos: '>10' if x == 0 else f"{11-int(x):.0f}"
mycolorbar = plt.colorbar(mappable=scalar_map, ax=ax, format=ticker.FuncFormatter(formatter))

# mycolorbar=plt.colorbar(mappable=scalar_map, ax=ax,format=ticker.FuncFormatter(lambda x, pos: f"{11-int(x):.0f}"))
# plt.colorbar(scalar_map)
mycolorbar.set_label('全\n国\n各\n省\n重\n要\n性\n排\n名',rotation=0,va='center', ha='left')
plt.tight_layout()
plt.show()
