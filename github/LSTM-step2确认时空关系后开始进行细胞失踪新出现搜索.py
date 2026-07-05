import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import networkx as nx

'''
#############################################识别每个时刻目标细胞最近细胞的距离，并绘制折线图
'''
#############################################识别每个时刻目标细胞最近细胞的距离，并绘制折线图
"""
计算每个ID在每个时刻最近的另一个ID的距离。
参数:
    df (pd.DataFrame): 包含ID、t、Position X、Position Y、Position Z的DataFrame。

返回:
    pd.DataFrame: 包含原始数据以及最近ID的距离和ID。
"""
def calculate_nearest_id_distances(df):
    # 初始化结果列表
    results = []
    # 按时间分组
    for t, group in df.groupby('t'):
        # 提取当前时刻的所有坐标
        coords = group[['Position X', 'Position Y', 'Position Z']].values
        ids = group['ID'].values
        # 计算所有点之间的距离矩阵
        distance_matrix = cdist(coords, coords, 'euclidean')
        # 遍历每个ID
        for i, (ID, coord) in enumerate(zip(ids, coords)):
            # 找到最近的其他ID的距离和索引
            distances = distance_matrix[i]
            distances[i] = np.inf  # 将自身距离设置为无穷大，避免选到自己
            nearest_index = np.argmin(distances)
            nearest_distance = distances[nearest_index]
            nearest_id = ids[nearest_index]
            # 保存结果
            results.append({
                'ID': ID,
                't': t,
                'Position X': coord[0],
                'Position Y': coord[1],
                'Position Z': coord[2],
                'Nearest ID': nearest_id,
                'Nearest Distance': nearest_distance
            })
    # 将结果转换为DataFrame
    result_df = pd.DataFrame(results)
    return result_df



'''*****************************************************
#############################################形态参数对碰撞距离进行线性拟合---新方法
'''
#############################################形态参数对碰撞距离进行线性拟合---新方法
def detect_collisions(df, volume_col='Volume', ellip_oblate_col='Ellipticity (oblate)',
                      ellip_prolate_col='Ellipticity (prolate)',
                      x_col='Position X', y_col='Position Y', z_col='Position Z',
                      time_col='t', id_col='ID', report=True, **kwargs):
    """
    输入：
        df : DataFrame，必须包含列：id_col、time_col、x_col、y_col、z_col、
             volume_col、ellip_oblate_col、ellip_prolate_col
    返回：
        df_out : 原 df 增加 'Collision' 布尔列
        collision_dict : {time : [碰撞细胞ID列表]}
        cell_collision_count : {细胞ID : 碰撞次数}
    """
    df = df.copy()

    # 1. 计算预测碰撞距离
    radius = (3 * df[volume_col] / (4 * np.pi)) ** (1 / 3)
    df['touchdistance'] = radius * (
        1 + df[ellip_oblate_col] + df[ellip_prolate_col]
    )

    # 2. 初始化结果列
    df['Collision'] = False
    collision_dict = {}
    cell_collision_count = {}

    # 3. 按时间分组检测碰撞
    for t, group in df.groupby(time_col):
        coords = group[[x_col, y_col, z_col]].values
        touch = group['touchdistance'].values
        ids = group[id_col].values

        dist_mat = cdist(coords, coords, 'euclidean')

        flag = np.zeros(len(group), dtype=bool)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if dist_mat[i, j] <= touch[i] + touch[j]:
                    flag[i] = flag[j] = True

        df.loc[group.index, 'Collision'] = flag

        # 记录碰撞事件
        collision_cells = ids[flag]
        if collision_cells.size > 0:
            collision_dict[t] = collision_cells.tolist()

    # 4. 统计细胞碰撞次数
    for cell_list in collision_dict.values():
        for cell in cell_list:
            cell_collision_count[cell] = cell_collision_count.get(cell, 0) + 1

    if report:
        total = len(collision_dict)
        print(f"总碰撞次数: {total}")
        for t, lst in collision_dict.items():
            print(f"时间 {t} 的碰撞事件: {lst}")
        print("\n每个细胞参与的碰撞次数:")
        for cell, cnt in cell_collision_count.items():
            print(f"细胞 {cell}: {cnt} 次碰撞")
    return df, collision_dict, cell_collision_count




'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化、体积变化
'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化、体积变化
'''
1.找出每个ID的最后时刻且在之前五个时刻内发生过碰撞的
2. 找出每个ID新出现的时刻
'''
def extract_last_and_first_positions(data, time_col='t', id_col='ID',
                                     collision_col='Collision',
                                     x_col='Position X', y_col='Position Y', z_col='Position Z',
                                     report: bool = True):
    """
    返回两个 DataFrame：
      last_collision_df : 每个 ID 最后一次出现且在之前 5 个时刻内发生过碰撞
      first_appearance_df : 每个 ID 第一次出现的时刻和位置

    参数
    ----
    report : bool, 默认 True
        是否打印简要报告
    """
    # 1) 最后一次出现 + 近 5 时刻碰撞
    last_moment = data.groupby(id_col)[time_col].transform(max)
    data['Last_Moment'] = last_moment

    data['Collision_in_Last_5'] = (
        data.groupby(id_col, group_keys=False)
            .apply(lambda g: g[collision_col].shift()
                   .rolling(window=5, min_periods=1).sum()
                   .fillna(0) > 0)
    )

    last_collision_df = (
        data[(data[time_col] == data['Last_Moment']) & (data['Collision_in_Last_5'])]
        .loc[:, [id_col, time_col, x_col, y_col, z_col]]
        .rename(columns={
            time_col: 'Last_Moment',
            x_col: 'Last_Position_X',
            y_col: 'Last_Position_Y',
            z_col: 'Last_Position_Z'
        })
    )

    # 2) 第一次出现
    first_moment = data.groupby(id_col)[time_col].transform(min)
    first_appearance_df = (
        data[data[time_col] == first_moment]
        .loc[:, [id_col, time_col, x_col, y_col, z_col]]
        .rename(columns={
            time_col: 'First_Moment',
            x_col: 'First_Position_X',
            y_col: 'First_Position_Y',
            z_col: 'First_Position_Z'
        })
    )

    # 3) 报告（可选）
    if report:
        print("每个ID的最后时刻且在之前五个时刻内发生过碰撞的：")
        print(last_collision_df)
        print("\n每个ID新出现的时刻和位置：")
        print(first_appearance_df)

    return last_collision_df, first_appearance_df




'''************************************************************************
##########################################计算每个细胞消失事件的时间邻域和空间邻域内存在的细胞出现事件
'''
##########################################计算每个细胞消失事件的时间邻域和空间邻域内存在的细胞出现事件
# 2. 计算每个细胞的平均速度
def calculate_average_speed(cell_df):
    # 计算每个时间步的位移
    cell_df['dx'] = cell_df['Position X'].diff()
    cell_df['dy'] = cell_df['Position Y'].diff()
    cell_df['dz'] = cell_df['Position Z'].diff()
    cell_df['displacement'] = np.sqrt(cell_df['dx']**2 + cell_df['dy']**2 + cell_df['dz']**2)
    # 计算平均速度
    average_speed = cell_df['displacement'].mean()
    return average_speed

# 3.1 定义时间邻域
def define_temporal_domain(row, df):
    if row['Event_Type']==1.0:
        return None
    cell_id = row['ID']
    last_moment = row['Last_Moment']
    
    # 定义时间范围：消失时刻后 0-10 个时刻
    start_time = last_moment
    end_time = last_moment + 10
    
    # 筛选出在时间范围内的出现细胞
    new_cells = df[(df['t'] > start_time) & (df['t'] <= end_time) & (df['Event_Type'] == 1)]
    
    if new_cells.empty:
        return None
    else:
        return new_cells['ID'].tolist()

# 3.2 定义时空邻域
def define_spatial_domain(row, df):
    cell_id = row['ID']
    last_moment = row['Last_Moment']
    touchdistance = row['touchdistance']
    average_speed = row['Average_Speed']
    timecell = row['timecell']
    
    if timecell is None:
        return None
    
    close_cells = []
    for new_cell_id in timecell:
        new_cell = df[(df['ID'] == new_cell_id) & (df['Event_Type'] == 1)]
        if new_cell.empty:
            continue
        
        new_cell = new_cell.iloc[0]  # 获取第一个匹配的行
        time_diff = new_cell['t'] - last_moment
        spatial_radius = max(1.5 * average_speed * time_diff, 15 * time_diff)
        
        distance = np.sqrt(
            (new_cell['Position X'] - row['Position X'])**2 +
            (new_cell['Position Y'] - row['Position Y'])**2 +
            (new_cell['Position Z'] - row['Position Z'])**2
        )
        
        if distance < spatial_radius:
            close_cells.append(new_cell_id)
    
    if close_cells:
        return close_cells
    else:
        return None

'''************************************************************************
##########################################在此基础上，只保留消失事件的细胞，如果某个细胞和其他细胞的Close_New_Cells存在重合部分，那么把它们定义为一个事件，否则单独定义为一个事件，用数字来表示
'''
##########################################在此基础上，只保留消失事件的细胞，如果某个细胞和其他细胞的Close_New_Cells存在重合部分，那么把它们定义为一个事件，否则单独定义为一个事件，用数字来表示
def unique(nums):
    N=[]
    for num in nums:
        N=N+num[:]
    return N

def build_disappearance_network(disappearance_df,
                                close_cells_col='Close_New_Cells',
                                id_col='ID',
                                report=True,
                                plot=False,
                                figsize=(8, 6)):
    """
    根据消失细胞及其“Close_New_Cells”构建网络，并生成汇总表。

    参数
    ----
    disappearance_df : DataFrame
        必须包含 id_col 和 close_cells_col（列表形式）
    close_cells_col : str, 默认 'Close_New_Cells'
    id_col : str, 默认 'ID'
    report : bool, 默认 True, 是否打印汇总表
    plot : bool, 默认 False, 是否绘制网络图
    figsize : tuple, 绘图尺寸

    返回
    ----
    new_df : DataFrame
        汇总表，含 Event_ID, Disappearance_Cells, Appearance_Cells
    """
    # 1. 建图
    G = nx.Graph()
    for _, row in disappearance_df.iterrows():
        G.add_node(row[id_col])

    for i in range(len(disappearance_df)):
        for j in range(i + 1, len(disappearance_df)):
            id1 = disappearance_df[id_col].iloc[i]
            id2 = disappearance_df[id_col].iloc[j]
            cells1 = set(disappearance_df[close_cells_col].iloc[i])
            cells2 = set(disappearance_df[close_cells_col].iloc[j])
            if cells1.intersection(cells2):
                G.add_edge(id1, id2)

    # 2. 汇总表
    components = list(nx.connected_components(G))
    new_data = {
        'Event_ID': [],
        'Disappearance_Cells': [],
        'Appearance_Cells': []
    }
    for idx, comp in enumerate(components, start=1):
        ids = list(comp)
        all_appear = set()
        for cell_id in ids:
            all_appear.update(
                disappearance_df[close_cells_col][disappearance_df[id_col] == cell_id].iloc[0]
            )
        new_data['Event_ID'].append(idx)
        new_data['Disappearance_Cells'].append(ids)
        new_data['Appearance_Cells'].append(list(all_appear))

    new_df = pd.DataFrame(new_data)
    new_df['Consistent'] = new_df.apply(lambda r: len(r['Disappearance_Cells']) == len(r['Appearance_Cells']), axis=1)
    new_df['Disless'] = new_df.apply(lambda r: len(r['Disappearance_Cells']) < len(r['Appearance_Cells']), axis=1)

    # 3. 可选报告与绘图
    if report:
        print(new_df)
    if plot:
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(list(G.nodes()))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=100, edge_color='gray', font_size=6)
        plt.title("Network of IDs")
        plt.show()

    return new_df




workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy4  = os.path.join(workpy, '碰撞检测与预测')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy4, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
events_path = os.path.join(save_dir, 'unique_events.csv')
events=pd.read_csv(events_path, index_col=None)
point=df.rename(columns={'Time': 't'})
point['ID']=point.index
'''
#########################################################运行
'''
#############################################识别每个时刻目标细胞最近细胞的距离，并绘制折线图
u=calculate_nearest_id_distances(point)
# 创建新的索引列
u['Index'] = u['ID'].astype(str) + '-' + u['t'].astype(str)
point['Index'] = point['ID'].astype(str) + '-' + point['t'].astype(str)
# 设置索引
u.set_index('Index', inplace=True)
point.set_index('Index', inplace=True)
U=pd.concat([u, point.loc[:,('Volume', 'Ellipticity (oblate)', 'Ellipticity (prolate)')]],axis=1)
U.to_csv(workpath+"\数据\红色细胞位置形态和最近距离.csv")
#############################################形态参数对碰撞距离进行线性拟合---新方法
data= pd.DataFrame(U)
data, collision_dict, cell_collision_count=detect_collisions(data, report=True)
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化、体积变化
last_df, first_df = extract_last_and_first_positions(data, report=True) 
##########################################计算每个细胞消失事件的时间邻域和空间邻域内存在的细胞出现事件
#data = pd.DataFrame(U)
# 1. 提取每个细胞 ID 的最后出现时刻和最初出现时刻
last_moment = data.groupby('ID')['t'].transform(max)
data['Last_Moment'] = last_moment
first_moment = data.groupby('ID')['t'].transform(min)
data['First_Moment'] = first_moment
# 添加一个标记列，用于区分细胞是消失还是出现
data['Event_Type'] = np.where(data['t'] == data['First_Moment'], 1,  # 出现
                            np.where(data['t'] == data['Last_Moment'], 0,  # 消失
                                     np.nan))  # 其他时刻标记为 NaN
# 2. 只保留细胞消失和出现时刻对应的行
filtered_df = data.dropna(subset=['Event_Type']).copy()
# 计算每个细胞的平均速度并存储在新列中
average_speeds = data.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
# 如果 data 里已存在同名列，则重命名新列
if 'Average_Speed' in data.columns:
    average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
data = data.merge(average_speeds, on='ID')
# 4.1 计算每个消失细胞的时间邻域内的出现细胞
filtered_df = data.dropna(subset=['Event_Type']).copy()
filtered_df['timecell'] = filtered_df.apply(lambda row: define_temporal_domain(row, data), axis=1)
filtered_df[filtered_df['timecell'].notna()]
filtered_df[filtered_df['timecell'].notna()].shape
# 4.2 计算每个消失细胞的时空邻域内的出现细胞
filtered_df['Close_New_Cells'] = filtered_df.apply(lambda row: define_spatial_domain(row, data), axis=1)
filtered_df[filtered_df['Close_New_Cells'].notna()]
# 1. 只保留消失事件的细胞
disappearance_df = filtered_df[filtered_df['Event_Type'] == 0].copy()
disappearance_df[disappearance_df['Close_New_Cells'].notna()]
disappearance_df_notna=disappearance_df[disappearance_df['Close_New_Cells'].notna()]
result_df = build_disappearance_network(disappearance_df_notna,
                                        plot=True,
                                        report=True)
N=unique(result_df['Appearance_Cells'])
print(len(N))
print(len(set(N)))
M=unique(result_df['Disappearance_Cells'])
print(len(M))
print(len(set(M)))

###有参数的df：
disappearance_df_notna
###保存碰撞事件的df：
result_df



#################################可视化
'''
#############################################识别每个时刻目标细胞最近细胞的距离，并绘制折线图
'''
#############################################识别每个时刻目标细胞最近细胞的距离，并绘制折线图
def plot_nearest_distance_over_time(result_df, target_id):
    """
    绘制目标ID在每个时刻的最近距离折线图。

    参数:
        result_df (pd.DataFrame): 包含每个ID在每个时刻的最近距离信息的DataFrame。
        target_id (int or str): 目标ID。

    返回:
        None，但会显示折线图。
    """
    # 筛选出目标ID的数据
    target_data = result_df[result_df['ID'] == target_id].copy()
    # 检查是否有数据
    if target_data.empty:
        print(f"No data found for ID {target_id}.")
        return
    # 按时间排序
    target_data.sort_values(by='t', inplace=True)
    # 提取时间和最近距离
    t = target_data['t'].values
    nearest_distances = target_data['Nearest Distance'].values
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(t, nearest_distances, marker='o', linestyle='-', color='blue', label=f"ID {target_id}")
    # 添加图例和标签
    plt.xlabel('Time (t)')
    plt.ylabel('Nearest Distance')
    plt.title(f'Nearest Distance Over Time for ID {target_id}')
    plt.grid(True)
    plt.legend()
    # 显示图表
    plt.show()
plot_nearest_distance_over_time(data,'green_1000000013')



'''*****************************************************
#############################################形态参数对碰撞距离进行线性拟合---新方法
'''
#############################################形态参数对碰撞距离进行线性拟合---新方法
def plot_last_collision_gap(df, save_path=None, show=True):
    """
    绘制细胞最后一次碰撞与最后一次出现的时间差提琴图。

    参数
    ----
    df        : DataFrame，必须包含 'ID', 't', 'Collision' 列
    save_path : str，可选，保存路径
    show      : bool，是否立即显示

    返回
    ----
    filtered_result : 过滤后的 DataFrame
    """
    # 1. 最后一次出现
    last_appearance = df.groupby('ID')['t'].max().reset_index()
    last_appearance.columns = ['ID', 'Last_Appearance']

    # 2. 最后一次碰撞
    collision_events = df[df['Collision']].groupby('ID')['t'].max().reset_index()
    collision_events.columns = ['ID', 'Last_Collision']

    # 3. 合并
    result = pd.merge(last_appearance, collision_events, on='ID', how='left')
    result['Last_Collision'].fillna(-np.inf, inplace=True)

    # 4. 差值 & 过滤
    result['Difference'] = result['Last_Appearance'] - result['Last_Collision']
    filtered_result = result[result['Difference'] != np.inf].copy()

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=filtered_result['Difference'], color='skyblue')
    plt.title('Violin Plot of Difference (Non-Infinite Values)', fontsize=14)
    plt.xlabel('Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()          # 立即释放图形内存

    return filtered_result
plot_last_collision_gap(data, save_path=None, show=True)
#save_path=workpath+'\图片\最后一次细胞碰撞与细胞失踪的时间差.pdf'

'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化
'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化
#######################绘制细胞出现、碰撞时刻图
def plot_ID_time_collision(df, time_range=None, save_path=None):
    """
    绘制ID和时间的标记图。
    参数： 
    - df: 包含ID、T、Collision三列的DataFrame。
    - time_range: 横坐标时间范围，格式为[min, max]，默认为None（自动计算）。
    - save_path: 图片保存路径，如果指定则保存为PDF文件。
    """
    # 获取唯一的ID列表和时间范围
    unique_ids = sorted(df['ID'].unique())
    if time_range is None:
        time_range = [df['t'].min(), df['t'].max()]
    # 创建一个新图
    plt.figure(figsize=(12, len(unique_ids)*0.3))
    # 遍历DataFrame，绘制每个点
    for _, row in df.iterrows():
        id_index = unique_ids.index(row['ID'])  # 获取ID对应的纵坐标
        time_index = row['t']  # 获取时间对应的横坐标
        color = 'purple' if row['Collision'] else 'skyblue'  # 根据Collision设置颜色
        plt.scatter(time_index, id_index, color=color, s=100, edgecolor='black', zorder=2)
    # 设置坐标轴范围
    plt.xlim(time_range[0], time_range[1])
    plt.ylim(-0.5, len(unique_ids) - 0.5)
    # 设置坐标轴标签和标题
    plt.title("ID-Time Collision Visualization", fontsize=14)
    plt.xlabel("Time (T)", fontsize=12)
    plt.ylabel("ID", fontsize=12)
    # 设置纵坐标刻度
    plt.yticks(ticks=range(len(unique_ids)), labels=unique_ids)
    # 添加网格
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Exists, Collision=False', 
               markerfacecolor='skyblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Exists, Collision=True', 
               markerfacecolor='purple', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Not Exists', 
               markerfacecolor='white', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    # 如果指定了保存路径，保存为PDF文件
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    # 显示图像
    plt.show()
def plot_ID_time_Nearest(df, time_range=None, save_path=None):
    """
    绘制ID和时间的标记图，根据Nearest Distance的数值使用plasma颜色映射。
    参数：
    - df: 包含ID、T、Nearest Distance三列的DataFrame。
    - time_range: 横坐标时间范围，格式为[min, max]，默认为None（自动计算）。
    - save_path: 图片保存路径，如果指定则保存为PDF文件。
    """
    # 获取唯一的ID列表和时间范围
    unique_ids = sorted(df['ID'].unique())
    if time_range is None:
        time_range = [df['t'].min(), df['t'].max()]
    # 创建一个新图
    plt.figure(figsize=(12, len(unique_ids)*0.3))
    # 使用scatter绘制每个点，根据Nearest Distance的值设置颜色
    scatter = plt.scatter(
        df['t'],  # 横坐标为时间T
        [unique_ids.index(id) for id in df['ID']],  # 纵坐标为ID的索引
        c=df['Nearest Distance'],  # 根据Nearest Distance设置颜色
        cmap='plasma',  # 使用plasma颜色映射
        s=100,  # 点的大小
        edgecolor='black',  # 点的边缘颜色
        zorder=2  # 确保点在网格上方
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Nearest Distance', fontsize=12)
    # 设置坐标轴范围
    plt.xlim(time_range[0], time_range[1])
    plt.ylim(-0.5, len(unique_ids) - 0.5)
    # 设置坐标轴标签和标题
    plt.title("ID-Time Visualization with Nearest Distance", fontsize=14)
    plt.xlabel("Time (T)", fontsize=12)
    plt.ylabel("ID", fontsize=12)
    # 设置纵坐标刻度
    plt.yticks(ticks=range(len(unique_ids)), labels=unique_ids)
    # 添加网格
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)
    # 如果指定了保存路径，保存为PDF文件
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    # 显示图像
    plt.show()
# 调用函数
plot_ID_time_collision(data,save_path=workpath+'\图片\ID时刻汇总_标记碰撞.pdf')
plot_ID_time_Nearest(data,save_path=workpath+'\图片\ID时刻汇总_标记最近细胞距离.pdf')



'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化、体积变化
'''
#############################################根据拟合的碰撞距离绘制碰撞可能性图和最近距离变化、体积变化
'''
    绘制ID和时间的标记图，根据Nearest Distance的数值使用plasma颜色映射。
    参数：
    - df: 包含ID、T、Nearest Distance三列的DataFrame。
    - time_range: 横坐标时间范围，格式为[min, max]，默认为None（自动计算）。
    - save_path: 图片保存路径，如果指定则保存为PDF文件。
'''
def plot_ID_time_Nearest(df, time_range=None, save_path=None):
    # 获取唯一的ID列表和时间范围
    unique_ids = sorted(df['ID'].unique())
    if time_range is None:
        time_range = [df['t'].min(), df['t'].max()]
    # 创建一个新图
    plt.figure(figsize=(12, len(unique_ids)*0.3))
    # 使用scatter绘制每个点，根据Volume的值设置颜色
    scatter = plt.scatter(
        df['t'],  # 横坐标为时间T
        [unique_ids.index(id) for id in df['ID']],  # 纵坐标为ID的索引
        c=df['Volume'],  # 根据Volume设置颜色
        cmap='plasma',  # 使用plasma颜色映射
        s=100,  # 点的大小
        edgecolor='black',  # 点的边缘颜色
        zorder=2  # 确保点在网格上方
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Volume', fontsize=12)
    # 设置坐标轴范围
    plt.xlim(time_range[0], time_range[1])
    plt.ylim(-0.5, len(unique_ids) - 0.5)
    # 设置坐标轴标签和标题
    plt.title("ID-Time Visualization with Volume", fontsize=14)
    plt.xlabel("Time (T)", fontsize=12)
    plt.ylabel("ID", fontsize=12)
    # 设置纵坐标刻度
    plt.yticks(ticks=range(len(unique_ids)), labels=unique_ids)
    # 添加网格
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)
    # 如果指定了保存路径，保存为PDF文件
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    # 显示图像
    plt.show()
# 调用函数
plot_ID_time_Nearest(data, save_path=workpath+'\图片\ID时刻汇总_标记体积变化.pdf')


























