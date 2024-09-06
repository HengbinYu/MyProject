import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

OriData = pd.read_excel("H:/OneDrive - mail.bnu.edu.cn/智慧托育/数据/InitialData_before_woman.xlsx",engine='openpyxl')  # 请将
data=OriData.iloc[:,:14]

n_clusters = 4
km = KModes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=1,random_state=42)
clusters = km.fit_predict(data)

# 将聚类结果及与聚类中心的距离存到原数据的最后两列
data['Cluster'] = clusters
OriData['Cluster'] = clusters
# 使用numpy的方法计算每个样本到最近的簇中心的距离
cluster_centers = km.cluster_centroids_
distances = np.linalg.norm(data.drop(['Cluster'], axis=1).values - cluster_centers[clusters], axis=1)
data['Distance_to_Center'] = distances
OriData['Distance_to_Center'] = distances
# 5. 对聚类效果采用轮廓系数进行检验
silhouette_avg = silhouette_score(data.drop(['Cluster', 'Distance_to_Center'], axis=1), clusters)
print(f"Silhouette Score: {silhouette_avg}")

# 保存结果到新的Excel文件
data.to_excel("clustered4_woman_Cao.xlsx", index=False)
OriData.to_excel("clustered4_woman_all_Cao.xlsx", index=False)

cluster_counts = pd.Series(clusters).value_counts()
print("Cluster Counts:")
print(cluster_counts)
