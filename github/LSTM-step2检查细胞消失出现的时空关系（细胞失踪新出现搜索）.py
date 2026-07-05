from ctypes import pointer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import pandas as pd
from PyEMD import EMD, EEMD, visualisation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import LSTM, Dense, Input
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
'''
#############################################绘图展示细胞首次、最后的出现位置，颜色根据不同需要上色
'''
#############################################绘图展示细胞首次、最后的出现位置，颜色根据不同需要上色
'''    绘制指定ID出现的时间范围的横线图，并可保存为PDF文件。
    
    参数:
        df: DataFrame，包含两列：'ID' 和 't'。
        ID: list，需要绘制的ID列表。
        time: list，时间范围，默认为 [1, 120]。
        save_path: str，保存PDF文件的路径。如果为None，则不保存。
'''
def IDtime(df, ID, time=[1, 120], save_path=None):####绘制指定ID出现的时间范围的横线图，并可保存为PDF文件。
    # 将ID转换为数值型
    ID = [int(item) for item in ID]
    # 筛选出指定ID的数据
    data = df.loc[df['ID'].isin(ID), ("ID", "t")]
    # 创建绘图
    plt.figure(figsize=(10, len(ID)*0.3))
    # 将ID映射到数值刻度
    id_mapping = {id: i for i, id in enumerate(ID)}
    # 遍历每个ID
    for id in ID:
        # 筛选出当前ID的所有时间点
        id_times = data[data['ID'] == id]['t']
        # 如果该ID没有时间点，跳过
        if id_times.empty:
            continue
        # 获取时间点的不连续区间
        time_intervals = []
        id_times = id_times.sort_values().unique()  # 去重并排序
        start_time = None
        for t in id_times:
            if start_time is None:
                start_time = t
                end_time = t
            elif t == end_time + 1:  # 如果时间连续
                end_time = t
            else:  # 如果时间不连续
                time_intervals.append((start_time, end_time - start_time + 1))
                start_time = t
                end_time = t
        # 添加最后一个区间
        if start_time is not None:
            time_intervals.append((start_time, end_time - start_time + 1))
        # 绘制不连续的横线
        plt.broken_barh(time_intervals, (id_mapping[id] - 0.4, 0.8), facecolors='skyblue', edgecolor='white')
    # 设置图表标题和坐标轴标签
    plt.title("ID Time Ranges", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("ID", fontsize=12)
    # 设置横坐标范围
    plt.xlim(time[0], time[1])
    # 设置纵坐标范围（使用映射后的数值刻度，但显示原始ID字符串）
    plt.yticks(range(len(ID)), labels=ID)
    # 显示网格
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    # 如果指定了保存路径，保存为PDF文件
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")
    # 显示图表
    plt.show()

###############################################显示细胞首次/最后出现位置（二维、三维、交互三个函数）：
"""
绘制所有ID首次出现的位置（基于Position X和Position Y）。
参数:
    df: DataFrame，包含列：'Position X', 'Position Y', 'ID', 't'。
    save_path: str，保存图像的路径。如果为None，则不保存图像。
    switch：控制筛选首次first还是最后last
"""
def plot_initial_positions(df, save_path=None,switch='first'):
    # 确保数据按时间排序
    df = df.sort_values(by='t')
    # 找到每个ID首次出现的记录
    first_occurrences = df.drop_duplicates(subset='ID', keep=switch)
    # 提取首次出现时的X和Y位置
    x_positions = first_occurrences['Position X']
    y_positions = first_occurrences['Position Y']
    ids = first_occurrences['ID']
    # 创建绘图
    plt.figure(figsize=(10, 6))
    # 绘制散点图
    plt.scatter(x_positions, y_positions, s=50, color='blue', edgecolor='black', alpha=0.7)
    # 添加标注（可选：显示每个点的ID）
    #for x, y, id in zip(x_positions, y_positions, ids):
    #    plt.text(x, y, str(id), fontsize=9, ha='right', va='bottom', color='red')
    # 设置图表标题和坐标轴标签
    plt.title("Initial Positions of IDs", fontsize=14)
    plt.xlabel("Position X", fontsize=12)
    plt.ylabel("Position Y", fontsize=12)
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.5)
    # 如果指定了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图像已保存到：{save_path}")
    # 显示图表
    plt.show()
"""
绘制所有ID首次出现时的三维空间位置。
参数:
    df: DataFrame，包含列：'Position X', 'Position Y', 'Position Z', 'ID', 't'。
    save_path: str，保存图片的路径。如果为None，则不保存。
"""
def plot_initial_positions_3D(df, save_path=None,
                              Distance_to_Image_Border_XY='Distance to Image Border XY',
                              switch='first'):
    # 确保输入数据包含所需列
    required_columns = ['Position X', 'Position Y', 'Position Z', 'ID', 't', Distance_to_Image_Border_XY]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入数据必须包含以下列：{required_columns}")
    # 找到每个ID首次出现的时间
    first_occurrence = df.drop_duplicates(subset='ID', keep=switch)
    
    # 提取首次出现时的三维位置和Distance_to_Image_Border_XY_adjust
    positions = first_occurrence[['Position X', 'Position Y', 'Position Z']]
    distances = first_occurrence[Distance_to_Image_Border_XY]
    # 创建三维绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 根据Distance_to_Image_Border_XY_adjust调整颜色深浅
    scatter = ax.scatter(
        positions['Position X'], positions['Position Y'], positions['Position Z'],
        c=distances,  # 颜色根据距离调整
        cmap='viridis',  # 使用viridis颜色映射
        marker='o', s=50, label='Initial Positions'
    )
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(Distance_to_Image_Border_XY, labelpad=10)
    # 添加标签和标题
    ax.set_xlabel('Position X', labelpad=10)
    ax.set_ylabel('Position Y', labelpad=10)
    ax.set_zlabel('Position Z', labelpad=10)
    ax.set_title('Initial Positions of IDs in 3D Space', fontsize=14)
    # 添加图例
    ax.legend()
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")
    # 显示图表
    plt.show()
"""
使用Plotly绘制所有ID首次出现时的三维空间位置，并根据Distance_to_Image_Border_XY_adjust调整颜色深浅。
图表可以交互式旋转和观察，并可保存为HTML文件。

参数:
    df: DataFrame，包含列：'Position X', 'Position Y', 'Position Z', 'ID', 't', 'Distance_to_Image_Border_XY_adjust'。
    save_path: str，保存HTML文件的路径。如果为None，则不保存。
"""
#########用颜色标记到XY四个面的距离
def plot_initial_positions_interactive_plotly(df, save_path=None,
                                              Distance_to_Image_Border_XY='Distance to Image Border XY',
                                              switch='first'):
    # 确保输入数据包含所需列
    required_columns = ['Position X', 'Position Y', 'Position Z', 'ID', 't', Distance_to_Image_Border_XY]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入数据必须包含以下列：{required_columns}")
    # 找到每个ID首次出现的时间
    first_occurrence = df.drop_duplicates(subset='ID', keep=switch)
    # 绘制交互式三维散点图
    fig = px.scatter_3d(
        first_occurrence,
        x='Position X', y='Position Y', z='Position Z',
        color=Distance_to_Image_Border_XY,  # 根据距离调整颜色
        color_continuous_scale='viridis',  # 使用viridis颜色映射
        title='Initial Positions of IDs in 3D Space',
        labels={'Position X': 'Position X', 'Position Y': 'Position Y', 'Position Z': 'Position Z'}
    )
    # 如果指定了保存路径，保存为HTML文件
    if save_path:
        fig.write_html(save_path)
        print(f"图表已保存为HTML文件：{save_path}")
    # 显示图表
    fig.show()
#########用颜色标记首次、最后出现的时间
def plot_initial_positions_interactive_plotly_T(df, save_path=None,switch='first'):
    # 确保输入数据包含所需列
    required_columns = ['Position X', 'Position Y', 'Position Z', 'ID', 't']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入数据必须包含以下列：{required_columns}")
    # 找到每个ID首次出现的时间
    first_occurrence = df.drop_duplicates(subset='ID', keep=switch)
    # 绘制交互式三维散点图idxmax
    fig = px.scatter_3d(
        first_occurrence,
        x='Position X', y='Position Y', z='Position Z',
        color='t',  # 根据距离调整颜色
        color_continuous_scale='plasma',  # 使用viridis颜色映射
        title='Initial Positions of IDs in 3D Space',
        labels={'Position X': 'Position X', 'Position Y': 'Position Y', 'Position Z': 'Position Z'}
    )
    # 如果指定了保存路径，保存为HTML文件
    if save_path:
        fig.write_html(save_path)
        print(f"图表已保存为HTML文件：{save_path}")
    # 显示图表
    fig.show()


######################美化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import adjustText
def plot_initial_positions_ad(df,
                           save_path=None,
                           switch='first',
                           ## ① 外观开关 --------------------------------------------------
                           palette='viridis',      # 散点颜色映射
                           marker_size=60,
                           marker_alpha=0.85,
                           edge_width=0.8,
                           dark_bg=False,
                           show_labels=False,
                           label_font=9,
                           label_color='auto',
                           ## ② 密度填充开关 ---------------------------------------------
                           density_fill=True,      # 是否画密度等高
                           density_alpha=0.35,     # 填充透明度
                           n_levels=15,            # 等高线层数
                           density_colors='auto',  # 'auto'=与散点同色系 / 给任意 colormap
                           ## ③ 版式开关 --------------------------------------------------
                           figsize=(10, 6),
                           dpi=300,
                           title='Initial 2D Positions with Density',
                           xlabel='Position X',
                           ylabel='Position Y'):
    # 0. 数据准备
    required = ['Position X', 'Position Y', 'ID', 't']
    if not all(c in df.columns for c in required):
        raise ValueError(f'缺少必要列：{required}')
    df = df.sort_values('t')
    first = df.drop_duplicates(subset='ID', keep=switch)
    x = first['Position X'].to_numpy()
    y = first['Position Y'].to_numpy()
    ids = first['ID'].to_numpy()

    # 1. 风格骨架
    if dark_bg:
        plt.style.use('dark_background')
        bg_color = '#111111'
        grid_color = 'white'
        default_label = 'white'
    else:
        sns.set_theme(style='whitegrid', font='Arial')
        bg_color = 'white'
        grid_color = 'gray'
        default_label = 'black'

    # 2. 颜色映射（散点）
    norm = plt.Normalize(first['t'].min(), first['t'].max())
    cmap = sns.color_palette(palette, as_cmap=True)
    face_colors = cmap(norm(first['t']))

    # 3. 画布
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 4. 密度填充（关键）
    if density_fill:
        # KDE 核密度估计
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        # 建立网格
        x_min, x_max = x.min() - (x.max() - x.min()) * 0.15, x.max() + (x.max() - x.min()) * 0.15
        y_min, y_max = y.min() - (y.max() - y.min()) * 0.15, y.max() + (y.max() - y.min()) * 0.15
        xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        # 颜色：默认用与散点同色系的淡色版
        if density_colors == 'auto':
            density_cmap = cmap
        else:
            density_cmap = plt.get_cmap(density_colors)
        # 画填充等高线
        ax.contourf(xx, yy, density, levels=n_levels, cmap=density_cmap,
                    alpha=density_alpha, antialiased=True, zorder=1)

    # 5. 散点（描边 + 填充）
    ax.scatter(x, y,
               s=marker_size,
               c=face_colors,
               alpha=marker_alpha,
               edgecolors='black',
               linewidths=edge_width,
               zorder=3)

    # 6. 自动标注（可选）
    if show_labels:
        texts = []
        for xi, yi, idi in zip(x, y, ids):
            if label_color == 'auto':
                luminance = (0.299 * face_colors[0][0] +
                             0.587 * face_colors[0][1] +
                             0.114 * face_colors[0][2])
                color = 'white' if luminance < 0.5 else 'black'
            else:
                color = label_color
            texts.append(ax.text(xi, yi, str(idi),
                                 fontsize=label_font,
                                 color=color,
                                 ha='center', va='center',
                                 weight='bold'))
        adjustText.adjust_text(texts, ax=ax,
                               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 7. 坐标轴
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='both', labelsize=11)

    # 8. 网格
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4, color=grid_color)
    
    # 8.5 颜色条（散点映射 -> t）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])          # 空数组即可
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8, aspect=30)
    cbar.set_label('Time t', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    # 9. 假外框
    if not dark_bg:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', lw=1,
                               transform=ax.transAxes, zorder=5))

    # 10. 保存 & 展示
    if save_path:
        #fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=dpi)
        fig.savefig(save_path, bbox_inches='tight',
            transparent=False,   # 透明关
            facecolor='white')
        #print(f'图像已保存 → {save_path.with_suffix(".png")} & .pdf')
    plt.show()
    return fig





'''
#############################################提琴图、均值折线图展示两类距离之间三种计算的差异
'''
#############################################提琴图、均值折线图展示两类距离之间三种计算的差异
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
def calculate_initial_distances(df,border=None,Distance_to_Image_Border_XY='Distance to Image Border XY'):
    # 找到每个ID最初时刻的坐标
    initial_coords = df.drop_duplicates(subset='ID', keep='first')[['Position X', 'Position Y', 'Position Z']]
    Bdata=df.drop_duplicates(subset='ID', keep='first')[[Distance_to_Image_Border_XY]]
    # 筛选Distance_to_Image_Border_XY_adjust大于或等于border的ID
    if border:
        initial_coords = initial_coords[Bdata[Distance_to_Image_Border_XY] >= border]
    # 计算每个坐标到其他所有坐标的距离
    distances = cdist(initial_coords, initial_coords, 'euclidean')
    # 将对角线元素（自身距离）设置为无穷大，以排除自身
    np.fill_diagonal(distances, np.inf)
    # 找到每个坐标到最近坐标的距离（排除自身）
    min_distances = np.min(distances, axis=1)
    min_distances = min_distances[min_distances > 0]
    return min_distances
def calculate_final_distances(df,border=None,Distance_to_Image_Border_XY='Distance to Image Border XY'):
    # 找到每个ID最后时刻的坐标
    final_data = df.drop_duplicates(subset='ID', keep='last')
    final_coords = final_data[['Position X', 'Position Y', 'Position Z']].values
    Bdata = final_data[Distance_to_Image_Border_XY].values
    # 筛选Distance_to_Image_Border_XY_adjust大于或等于border的ID
    if border:
        mask_final = Bdata >= border
        final_coords = final_coords[mask_final]
    # 计算每个坐标到其他所有坐标的距离
    distances = cdist(final_coords, final_coords, 'euclidean')
    # 将对角线元素（自身距离）设置为无穷大，以排除自身
    np.fill_diagonal(distances, np.inf)
    # 找到每个坐标到最近坐标的距离（排除自身）
    min_distances = np.min(distances, axis=1)
    min_distances = min_distances[min_distances > 0]
    return min_distances
def calculate_initial_to_final_distances(df, border=None,Distance_to_Image_Border_XY='Distance to Image Border XY'):
    # 找到每个ID最初和最后时刻的坐标
    initial_data = df.drop_duplicates(subset='ID', keep='first')
    final_data = df.drop_duplicates(subset='ID', keep='last')
    # 提取最初和最后时刻的坐标
    initial_coords = initial_data[['Position X', 'Position Y', 'Position Z']].values
    final_coords = final_data[['Position X', 'Position Y', 'Position Z']].values
    # 提取最初和最后时刻的Distance_to_Image_Border_XY_adjust
    BdataI = initial_data[Distance_to_Image_Border_XY].values
    BdataF = final_data[Distance_to_Image_Border_XY].values
    # 筛选Distance_to_Image_Border_XY_adjust大于或等于border的ID
    if border:
        mask_initial = BdataI >= border
        mask_final = BdataF >= border
        initial_coords = initial_coords[mask_initial]
        final_coords = final_coords[mask_final]
    # 计算每个initial_coords到所有final_coords的距离
    distances = cdist(initial_coords, final_coords, 'euclidean')
    # 找到每个initial_coords到最近的final_coords的距离
    min_distances = np.min(distances, axis=1)
    return min_distances

"""
绘制提琴图，展示三种距离的分布：
1. 最初时刻的细胞坐标到最近其他坐标的距离。
2. 最后时刻的细胞坐标到最近其他坐标的距离。
3. 最初时刻到最末时刻的细胞坐标的距离。

参数:
可以通过参数border控制只选择Distance_to_Image_Border_XY_adjust大于或等于该值的ID。
    initial_distances: 最初时刻的细胞坐标到最近其他坐标的距离数组。
    final_distances: 最后时刻的细胞坐标到最近其他坐标的距离数组。
    initial_to_final_distances: 最初时刻到最末时刻的细胞坐标的距离数组。
"""
def plot_violin_plots(initial_distances, final_distances, initial_to_final_distances,save_path=None):
    # 创建一个DataFrame来存储所有距离数据
    data = pd.DataFrame({
        'Distance': np.concatenate([initial_distances, final_distances, initial_to_final_distances]),
        'Type': ['Initial'] * len(initial_distances) + 
                ['Final'] * len(final_distances) + 
                ['Initial to Final'] * len(initial_to_final_distances)
    })
    
    # 绘制提琴图
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Type', y='Distance', data=data, palette='viridis')
    plt.title('Violin Plots of Distances', fontsize=16)
    plt.xlabel('Distance Type', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")
    plt.show()

"""
绘制不同border值下，三种距离函数的均值折线图。

参数:
    otherpoint: 输入数据
    border_values: 一个由数字组成的列表，表示不同的border参数值
"""
def plot_mean_distances_vs_border(otherpoint, border_values,save_path=None):
    # 初始化存储均值的列表
    initial_means = []
    final_means = []
    initial_to_final_means = []
    # 遍历不同的border值，计算每种情况下的均值
    for border in border_values:
        initial_distances = calculate_initial_distances(otherpoint, border)
        final_distances = calculate_final_distances(otherpoint, border)
        initial_to_final_distances = calculate_initial_to_final_distances(otherpoint, border)  
        # 计算均值
        initial_means.append(np.mean(initial_distances))
        final_means.append(np.mean(final_distances))
        initial_to_final_means.append(np.mean(initial_to_final_distances))
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(border_values, initial_means, label='Initial Distances Mean', marker='o')
    plt.plot(border_values, final_means, label='Final Distances Mean', marker='s')
    plt.plot(border_values, initial_to_final_means, label='Initial to Final Distances Mean', marker='^')
    # 添加图例和标签
    plt.xlabel('Border Value')
    plt.ylabel('Mean Distance')
    plt.title('Mean Distances vs. Border Value')
    plt.legend()
    plt.grid(True)
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")
    plt.show()
    
###########################美化
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

def _sig_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def plot_violin_plots_ad(initial_distances, final_distances, initial_to_final_distances,
                      palette=None,
                      save_path=None):
    # 0. 数据框
    data = pd.DataFrame({
        'Distance': np.concatenate([initial_distances, final_distances, initial_to_final_distances]),
        'Type': ['Initial'] * len(initial_distances) +
                ['Final'] * len(final_distances) +
                ['Initial→Final'] * len(initial_to_final_distances)
    })

    # 1. 风格
    sns.set_theme(style="whitegrid", font='Arial', font_scale=1.1)
    if palette is None:
        palette = sns.color_palette('Spectral_r', 3)

    # 2. 画布
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor('white')

    # 3. 提琴图
    vp = sns.violinplot(x='Type', y='Distance', data=data,
                        palette=palette, width=0.8, linewidth=1.2,
                        inner='box', alpha=0.75, ax=ax)

    # 4. 两两 Mann-Whitney U 检验（仅 i < j）
    groups = [initial_distances, final_distances, initial_to_final_distances]
    labels = ['Initial', 'Final', 'Initial→Final']
    y_max = data.Distance.max() + 0.08 * (data.Distance.max() - data.Distance.min())

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            star = _sig_star(p)
            x1, x2 = i, j
            ax.plot([x1, x2], [y_max, y_max], lw=1.2, color='black')
            ax.text((x1+x2)/2, y_max, star, ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
            y_max += 0.1 * (data.Distance.max() - data.Distance.min())  # 抬高下一层

    # 5. 标签与标题
    ax.set_title('Distance Distribution ( pairwise MW U test )', fontsize=16, pad=20)
    ax.set_xlabel(None)
    ax.set_ylabel('Distance (μm)', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    # 6. 保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
        print(f"✅ 图表已保存 → {save_path}")
    plt.show()


from scipy.stats import sem, ttest_rel
def plot_mean_distances_vs_border_ad(otherpoint, border_values,
                                  error_bar='sem',        # 'sem' 或 'std'
                                  mark_diff=True,         # 是否标注最大差异点
                                  palette=None,
                                  save_path=None):
    """
    三折线 + 误差阴影（SEM/STD） + 显著性星号
    """
    # 0. 风格骨架
    sns.set_theme(style="whitegrid", font='Arial', font_scale=1.1)
    if palette is None:
        palette = sns.color_palette('Spectral_r', 3)

    # 1. 预先计算均值 & 误差
    results = {
        'Initial':  [],
        'Final':    [],
        'Init→Final': []
    }
    errors = {k: [] for k in results}

    for border in border_values:
        init   = calculate_initial_distances(otherpoint, border)
        final  = calculate_final_distances(otherpoint, border)
        init2f = calculate_initial_to_final_distances(otherpoint, border)

        for dist_list, key in zip([init, final, init2f],
                                  ['Initial', 'Final', 'Init→Final']):
            results[key].append(np.mean(dist_list))
            if error_bar == 'sem':
                errors[key].append(sem(dist_list))
            else:
                errors[key].append(np.std(dist_list))

    df_plot = pd.DataFrame({**results, **{f'{k}_err': v for k, v in errors.items()}})
    df_plot['Border'] = border_values

    # 2. 画布
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor('white')

    # 3. 绘制三折线 + 误差阴影
    labels = ['Initial', 'Final', 'Init→Final']
    markers = ['o', 's', '^']
    for i, (lab, mk) in enumerate(zip(labels, markers)):
        y = df_plot[lab]
        yerr = df_plot[f'{lab}_err']
        ax.plot(df_plot['Border'], y, marker=mk, markersize=6,
                linewidth=2.5, label=lab, color=palette[i])
        ax.fill_between(df_plot['Border'],
                        y - yerr, y + yerr,
                        color=palette[i], alpha=0.2)

    # 5. 轴与标题
    ax.set_xlabel('Border Value', fontsize=13)
    ax.set_ylabel('Mean Distance (μm)', fontsize=13)
    ax.set_title('Mean Distances vs. Border Value', fontsize=16, pad=15)
    ax.legend(title='Distance Type', frameon=False, loc='best')
    ax.tick_params(axis='both', labelsize=11)
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    # 6. 保存
    if save_path:
        #Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
        print(f"✅ 图表已保存 → {save_path}")
    plt.show()



'''
#############################################计算每个时刻细胞的移动距离及其统计参数
'''
#############################################计算每个时刻细胞的移动距离及其统计参数
import numpy as np
import pandas as pd
"""
计算每个ID在每两个连续时刻之间的位置距离。  
参数:
    df: 包含ID、t（时间）、Position X、Position Y、Position Z的DataFrame。

返回:
    一个DataFrame，包含ID、t（时间）和连续时刻之间的距离。
"""
def calculate_continuous_distances(df):
    # 初始化结果列表
    results = []
    # 按ID分组
    for ID, group in df.groupby('ID'):
        # 按时间排序（确保时间是有序的）
        group = group.sort_values(by='t')
        # 提取时间和坐标
        times = group['t'].values
        coords = group[['Position X', 'Position Y', 'Position Z']].values
        # 初始化距离列表
        distances = []
        previous_time = times[0]
        previous_coord = coords[0]
        # 遍历每个时刻，计算连续时刻之间的距离
        for t, coord in zip(times[1:], coords[1:]):
            # 检查时间是否连续（时间差为1，假设时间单位是整数）
            if t - previous_time == 1:
                # 计算欧几里得距离
                distance = np.linalg.norm(coord - previous_coord)
                distances.append([ID, t, distance])
            # 更新前一个时刻的时间和坐标
            previous_time = t
            previous_coord = coord
        # 如果有计算结果，将其转换为DataFrame并添加到结果列表中
        if distances:
            result = pd.DataFrame(distances, columns=['ID', 't', 'Distance'])
            results.append(result)
    # 合并所有结果
    return pd.concat(results, ignore_index=True)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
"""
绘制 Distance 列的提琴图，并计算均值、极大值、极小值、众数等统计数值。

参数:
    distances_df: 包含 Distance 列的 DataFrame
"""
def plot_violin_and_statistics(distances_df,save_path=None):
    # 绘制提琴图
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=distances_df['Distance'], color='skyblue')
    plt.title('Violin Plot of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")
    plt.show()
    # 计算统计数值
    mean_distance = distances_df['Distance'].mean()
    max_distance = distances_df['Distance'].max()
    min_distance = distances_df['Distance'].min()
    mode_distance = stats.mode(distances_df['Distance'], keepdims=True).mode[0]
    median_distance = distances_df['Distance'].median()
    # 打印统计数值
    print(f"Mean Distance: {mean_distance:.4f}")
    print(f"Maximum Distance: {max_distance:.4f}")
    print(f"Minimum Distance: {min_distance:.4f}")
    print(f"Mode Distance: {mode_distance:.4f}")
    print(f"Median Distance: {median_distance:.4f}")



'''
#############################################排序目标细胞的出现时刻，还原轨迹
'''
#############################################排序目标细胞的出现时刻，还原轨迹
import numpy as np
"""
获取指定ID的所有出现时刻，并按时间排序。
参数:
    df (pd.DataFrame): 包含ID和时间戳t的DataFrame。
    target_id (int or str): 指定的ID。

返回:
    list: 指定ID的所有出现时刻，按时间排序。
"""
def get_sorted_timestamps(df, target_id):
    # 筛选出指定ID的所有记录
    target_data = df[df['ID'] == target_id]
    # 提取时间戳并排序
    sorted_timestamps = target_data['t'].sort_values().tolist()
    return sorted_timestamps
"""
绘制指定ID的三维轨迹，颜色从开始到结束渐变，并标记时间断开的位置。

参数:
    df (pd.DataFrame): 包含ID、t、Position X、Position Y、Position Z的DataFrame。
    target_id (int or str): 目标ID。
"""
def plot_3d_trajectory(df, target_id,save_path=None):
    # 筛选出目标ID的数据
    target_data = df[df['ID'] == target_id].sort_values(by='t')
    # 检查是否有匹配的ID
    if target_data.empty:
        print(f"No data found for ID {target_id}.")
        return
    # 提取坐标和时间
    x = target_data['Position X'].values
    y = target_data['Position Y'].values
    z = target_data['Position Z'].values
    t = target_data['t'].values
    # 检查数据是否有效
    if len(x) < 2:
        print(f"Not enough data points for ID {target_id}. Need at least 2 points.")
        return
    # 检测时间断开的位置
    time_gaps = np.where(np.diff(t) > 1)[0]  # 时间差大于1的位置
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制轨迹，颜色从开始到结束渐变
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(t.min(), t.max())
    lc = Line3DCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(t[:-1])  # 设置颜色映射数组
    lc.set_linewidth(2)
    ax.add_collection3d(lc)
    # 标记时间断开的位置
    for gap in time_gaps:
        ax.scatter(x[gap + 1], y[gap + 1], z[gap + 1], color='red', s=100, marker='o', zorder=5)
        ax.text(x[gap + 1], y[gap + 1], z[gap + 1], 'Gap', color='black', fontsize=12, zorder=6)
    # 设置图形属性
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title(f'3D Trajectory of ID {target_id}')
    fig.colorbar(lc, ax=ax, label='Time')
    # 设置合适的坐标轴范围
    ax.set_xlim([x.min() - 1, x.max() + 1])
    ax.set_ylim([y.min() - 1, y.max() + 1])
    ax.set_zlim([z.min() - 1, z.max() + 1])
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',facecolor='white')
        print(f"图表已保存到：{save_path}")    
    plt.show()


def inspect_nan(df: pd.DataFrame,
                fill_strategy: str = 'median',
                return_report: bool = True) -> pd.DataFrame:
    """
    1. 统计各列 NaN 数量与占比；
    2. 按指定策略填充 NaN（median/mean/ffill/bfill/zero）；
    3. 可选返回详细报告。

    参数
    ----
    df : pd.DataFrame
        输入表（行索引 TrackID，列含 Time）。
    fill_strategy : str, default 'median'
        填充方式：
        • 'median'  – 用该列中位数
        • 'mean'    – 用该列均值
        • 'ffill'   – 前向填充（用上一行同列值）
        • 'bfill'   – 后向填充（用下一行同列值）
        • 'zero'    – 用 0
    return_report : bool, default True
        是否返回 NaN 统计报告 DataFrame。

    返回
    ----
    df_filled : pd.DataFrame
        已填充 NaN 的新表。
    report : pd.DataFrame, optional
        NaN 统计报告（仅当 return_report=True）。
    """
    # 1. 统计
    nan_cnt = df.isna().sum()
    nan_pct = (nan_cnt / len(df) * 100).round(2)
    report = pd.DataFrame({'NaN_Count': nan_cnt, 'NaN_%': nan_pct})

    # 2. 填充
    df_filled = df.copy()
    numeric = df_filled.select_dtypes(include=[np.number])
    if fill_strategy == 'median':
        df_filled[numeric.columns] = numeric.fillna(numeric.median())
    elif fill_strategy == 'mean':
        df_filled[numeric.columns] = numeric.fillna(numeric.mean())
    elif fill_strategy == 'ffill':
        df_filled[numeric.columns] = numeric.ffill()
    elif fill_strategy == 'bfill':
        df_filled[numeric.columns] = numeric.bfill()
    elif fill_strategy == 'zero':
        df_filled[numeric.columns] = numeric.fillna(0)
    else:
        raise ValueError('fill_strategy 仅支持 median|mean|ffill|bfill|zero')

    if return_report:
        return df_filled, report
    else:
        return df_filled


'''
#############################################检查细胞出现消失位置与距离边缘距离的关系
'''
#############################################检查细胞出现消失位置与距离边缘距离的关系
def stacked_cumul_distance_edge3(df,
                                 switch='first',
                                 disp_col='Displacement^2',
                                 edge_dist_col='Distance to Image Border XYZ',
                                 time_col='t',
                                 track_col='TrackID',quantile=0.75,meanquantile=3,
                                 time_bin=3,
                                 palette=['#a1c9f4','#3c78d8','#e06666'],  # Near / Mid / Far
                                 save_path=None):
    # 0. 索引转列
    if df.index.name == track_col:
        df = df.reset_index()

    # 1. 全局统计量（开方后）
    true_disp = np.sqrt(df[disp_col])          # 先转真实位移
    D_max  = true_disp.max()*quantile
    D_mean = true_disp.mean()*meanquantile


    # 2. 取出现/消失事件
    df = df.sort_values(time_col)
    event = df.drop_duplicates(subset=track_col, keep=switch)

    # 3. 三分类
    dist_vals = event[edge_dist_col]
    event['zone'] = pd.cut(dist_vals,
                           bins=[0, D_mean, D_max, np.inf],
                           labels=['Near','Mid','Far'],
                           include_lowest=True)

    # 4. 时间分箱 + 计数
    event['time_bin'] = (event[time_col] // time_bin) * time_bin
    cnt = (event.groupby(['time_bin', 'zone'])
                 .size()
                 .unstack(fill_value=0)
                 .reindex(columns=['Near','Mid','Far'], fill_value=0))

    # 5. 累积
    cnt_cum = cnt.cumsum()

    # 6. 堆叠面积图（美化版）
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    fig.patch.set_facecolor('white')

    # 面积 + 描边 + 渐变透明
    ax.stackplot(cnt_cum.index,
                 cnt_cum['Near'], cnt_cum['Mid'], cnt_cum['Far'],
                 labels=['Near', 'Mid', 'Far'],
                 colors=palette, alpha=0.85, edgecolor='white', lw=0.8)

    # 样本量标注
    ax.text(0.02, 0.98,
            f"n = {len(event)}  events\n"
            f"tracks = {event[track_col].nunique()}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black')

    # 轴
    ax.set_xlabel('Time (frame)', fontsize=13)
    ax.set_ylabel('Cumulative events', fontsize=13)
    ax.set_title(f'Cumulative {"appearance" if switch=="first" else "disappearance"} events\n'
                 f'Near < {D_mean:.1f} μm,  Far > {D_max:.1f} μm',
                 fontsize=14, pad=15)

    # 整数时间刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Y 轴千位分隔符
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # 图例
    ax.legend(title='Distance zone', frameon=False, loc='lower right')

    # 无框线 + 淡化网格
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
        print(f"✅ 美化堆叠图已保存 → {save_path}")
    plt.show()
    return cnt_cum

def density_curve_by_distance(df,
                              switch='first',
                              border_xyz=('min', 'min', 'min'),
                              bin_width=10,
                              max_range=None):
    """
    统计“首次/末次"细胞 -> 到图像边界距离 -> 数量分布
    优先使用数据里已算好的距离列：
        1) 'Distance to Image Border XYZ'
        2) 'Distance to Image Border XY'
    都没有再现场计算
    """
    required = ['Position X', 'Position Y', 'Position Z', 'ID', 't']
    if not all(c in df.columns for c in required):
        raise ValueError(f'缺少必要列：{required}')

    # 1. 取首次/末次
    df = df.sort_values('t')
    sub = df.drop_duplicates(subset='ID', keep=switch).reset_index(drop=True)

    # 2. 优先用现成距离列
    if 'Distance to Image Border XYZ' in df.columns:
        dist = sub['Distance to Image Border XYZ'].values
    elif 'Distance to Image Border XY' in df.columns:
        dist = sub['Distance to Image Border XY'].values
    else:
        # ---- 原“自己算”逻辑 ----
        x_min, x_max = df['Position X'].min(), df['Position X'].max()
        y_min, y_max = df['Position Y'].min(), df['Position Y'].max()
        z_min, z_max = df['Position Z'].min(), df['Position Z'].max()

        border_vals = np.array([
            x_min if border_xyz[0] == 'min' else x_max,
            y_min if border_xyz[1] == 'min' else y_max,
            z_min if border_xyz[2] == 'min' else z_max
        ])
        pos = sub[['Position X', 'Position Y', 'Position Z']].values
        dist = np.sqrt(
            (pos[:, 0] - border_vals[0]) ** 2 +
            (pos[:, 1] - border_vals[1]) ** 2 +
            (pos[:, 2] - border_vals[2]) ** 2
        )

    # 3. 直方图统计
    if max_range is None:
        max_range = np.percentile(dist, 95)
    bins = np.arange(0, max_range + bin_width, bin_width)
    counts, edges = np.histogram(dist, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # 4. 返回
    df_count = pd.DataFrame({
        'Distance_to_Border': centers,
        'Count': counts
    })
    return df_count

'''
###############################################细胞轨迹过程中断点分布展示
'''
###############################################细胞轨迹过程中断点分布展示# %%

def detect_time_gaps(point, time_col='Time', track_col='TrackID',
                     save_dir=None, style='dark'):
    if point.index.name == track_col:
        point = point.reset_index()

    # 风格
    if style == 'dark':
        plt.style.use('dark_background')
        bg, text_c = '#111111', 'white'
    else:
        sns.set_theme(style="whitegrid", font='Arial')
        bg, text_c = 'white', 'black'

    #save_dir = Path(save_dir) if save_dir else Path('./gap_plots')
    #save_dir.mkdir(parents=True, exist_ok=True)


    # 1. 断点检测（整数时间）
    gap_records = []
    for tid, sub in point.groupby(track_col):
        t = sub[time_col].sort_values().astype(int).values   # 强制整数
        dt = np.diff(t)
        gap_mask = dt > 1
        if gap_mask.any():
            for gp in np.where(gap_mask)[0]:
                gap_records.append({
                    'TrackID': tid,
                    'GapStart': t[gp],
                    'GapEnd': t[gp+1],
                    'GapLen': int(t[gp+1] - t[gp])   # 整数
                })
    gap_df = pd.DataFrame(gap_records)

    total_tracks = point[track_col].nunique()
    tracks_with_gap = gap_df['TrackID'].nunique()
    prop_with_gap = tracks_with_gap / total_tracks

    # 2. 图1：环形饼图（已美化）
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie([prop_with_gap, 1 - prop_with_gap],
                                      labels=['With Gap', 'Continuous'],
                                      colors=['#e06666', '#57bb8a'],
                                      autopct='%1.1f%%', startangle=90,
                                      wedgeprops=dict(width=0.5))
    for t in texts + autotexts:
        t.set_color(text_c); t.set_fontsize=11
    ax.set_title('Tracks with Time Gaps', color=text_c, fontsize=13, pad=15)
    fig.patch.set_facecolor(bg)
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'gap_proportion.pdf'), bbox_inches='tight',
            transparent=False,   # 透明关
            facecolor='white')
    plt.show()

    # 3. 图2：每 Track 断点计数（现代条形）
    gap_count = gap_df.groupby('TrackID').size().rename('GapCount')
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    sns.countplot(x=gap_count, color='#3c78d8', edgecolor=bg, ax=ax)
    ax.set_xlabel('Gaps per Track', color=text_c)
    ax.set_ylabel('Number of Tracks', color=text_c)
    ax.set_title('Gap Count Distribution', color=text_c)
    # 标注样本量
    ax.text(0.95, 0.95, f'n = {len(gap_count)}', transform=ax.transAxes,
            ha='right', va='top', color=text_c, fontsize=10)
    fig.patch.set_facecolor(bg)
    if save_dir:
        fig.savefig(os.path.join(save_dir , 'gap_count.pdf'), bbox_inches='tight',
            transparent=False,   # 透明关
            facecolor='white')
    plt.show()

    # 4. 图3：断开时长（整数梯度柱状图）
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    # 用 value_counts 保证整数横轴
    vc = gap_df['GapLen'].value_counts().sort_index()
    sns.barplot(x=vc.index, y=vc.values, palette='Blues_r', edgecolor=bg, ax=ax)
    ax.set_xlabel('Gap Length (time units)', color=text_c)
    ax.set_ylabel('Count', color=text_c)
    ax.set_title('Gap Length Distribution', color=text_c)
    # 强制整数刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.patch.set_facecolor(bg)
    if save_dir:
        fig.savefig(os.path.join(save_dir,'gap_length.pdf'), bbox_inches='tight',
            transparent=False,   # 透明关
            facecolor='white')
    plt.show()

    print(f"✅ 完成！{len(gap_df)} 个断点，图已保存 → {save_dir}")
    return gap_df


'''
#########################################################细胞最近距离消失出现时间差
'''
#########################################################细胞最近距离消失出现时间差
def nearest_appear_after_disappear(df,
                                   id_col='ID',
                                   time_col='t',
                                   pos_cols=['Position X', 'Position Y', 'Position Z'],
                                   border=None,
                                   border_col='Distance to Image Border XY'):
    """
    对每条消失轨迹，找“空间最近”的出现轨迹，并记录时间信息
    返回 DataFrame：
        disappear_ID, appear_ID, disappear_t, appear_t, delta_t, distance
    border: 若给定，仅考虑消失/出现时刻 border_col ≥ border 的轨迹
    """
    # 0. 确保数值 & 去空
    df = df.copy()
    df[pos_cols] = df[pos_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=pos_cols + [time_col, id_col, border_col])

    # 1. 取事件
    df = df.sort_values(time_col)
    disappear = df.drop_duplicates(subset=id_col, keep='last')   # 消失
    appear    = df.drop_duplicates(subset=id_col, keep='first')  # 出现

    # 2. border 过滤（可选）
    if border is not None:
        disappear = disappear[disappear[border_col] >= border]
        appear    = appear[appear[border_col] >= border]

    # 3. 空间距离矩阵（消失 → 出现）
    dis_pos = disappear[pos_cols].astype(float).values
    app_pos = appear[pos_cols].astype(float).values
    D = cdist(dis_pos, app_pos, metric='euclidean')

    # 4. 把“同一 ID”位置设为 inf，排除自身
    same_id_mask = disappear[id_col].values.reshape(-1, 1) == appear[id_col].values
    D[same_id_mask] = np.inf

    # 5. 逐行找最近 & 记录时间
    res = []
    for i, (_, van) in enumerate(disappear.iterrows()):
        j_nearest = int(D[i].argmin())
        app = appear.iloc[j_nearest]
        res.append({
            'disappear_ID': van[id_col],
            'appear_ID': app[id_col],
            'disappear_t': van[time_col],
            'appear_t': app[time_col],
            'delta_t': app[time_col] - van[time_col],
            'distance': D[i, j_nearest]
        })
    return pd.DataFrame(res)
def plot_delta_t_dist(match_df,
                      save_path=None,
                      palette='Blues_r',
                      bin_width=1):
    """
    美化 Δt 分布图
    """
    # 1. 风格骨架
    sns.set_theme(style="whitegrid", font='Arial', font_scale=1.1)
    bg, text_c = 'white', 'black'

    # 2. 画布
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor(bg)

    # 3. 渐变柱状图（整数横轴）
    bins = np.arange(match_df['delta_t'].min(),
                     match_df['delta_t'].max() + bin_width + 1,
                     bin_width)
    sns.histplot(match_df['delta_t'], bins=bins, kde=True,
                 color='#3c78d8', edgecolor=bg, ax=ax)

    # 4. 统计标注
    n = len(match_df)
    med = match_df['delta_t'].median()
    neg_ratio = (match_df['delta_t'] < 0).mean() * 100
    ax.text(0.02, 0.98,
            f'n = {n}\nmedian = {med:.1f}\nnegative = {neg_ratio:.1f} %',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color=text_c)

    # 5. 轴
    ax.set_xlabel('Δt (appear − disappear, frames)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Time gap to nearest appearance', fontsize=14, pad=15)

    # 整数刻度 + 千位分隔
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # 无框线 + 淡化网格
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
        print(f"✅ Δt 分布图已保存 → {save_path}")
    plt.show()

'''
#############################################轨迹碎片化展示
'''
#############################################轨迹碎片化展示
def plot_track_length_violin(point,
                             track_col='TrackID',
                             time_col='Time',
                             save_path=None):
    """
    计算每个 Track 的出现时刻数（帧数），绘制美化提琴图
    """
    # 0. 索引转列（若需要）
    if point.index.name == track_col:
        point = point.reset_index()

    # 1. 统计：每个 Track 的时刻数
    track_len = (point.groupby(track_col)[time_col]
                      .nunique()          # 去重帧数
                      .rename('track_frames'))

    # 2. 画布
    sns.set_theme(style="whitegrid", font='Arial', font_scale=1.1)
    fig, ax = plt.subplots(figsize=(4, 4.5), dpi=300)
    fig.patch.set_facecolor('white')

    # 3. 提琴图
    sns.violinplot(y=track_len, color='#3c78d8', inner='box', alpha=0.75, linewidth=1.2, ax=ax)

    # 4. 统计标注
    n = len(track_len)
    med = track_len.median()
    ax.text(0.05, 0.98, f'n = {n}\nmedian = {med:.0f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='black')

    # 5. 轴
    ax.set_ylabel('Track length (frames)', fontsize=13)
    ax.set_xlabel(None)
    ax.set_title('Distribution of Track Lengths', fontsize=14, pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    if save_path:
        #Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
        #print(f"✅ 提琴图已保存 → {save_path}")
    plt.show()
    return track_len




##############################修正位移计算错误
def fix_displacement(df: pd.DataFrame,col_t='Time') -> pd.DataFrame:
    """
    修正位移计算：
    1. 把 t 转成数值
    2. 每个 ID 内部按 t 排序
    3. 计算 Displacement X/Y/Z 和 Displacement^2
    4. 第一行（t 最小）位移设为 0
    
    参数
    ----
    df : DataFrame
        必须含列 ['Position X','Position Y','Position Z','t']，且索引为 ID（重复）
    
    返回
    ----
    df_new : 同形状 DataFrame，仅更新位移列
    """
    # 深拷贝，避免改原表
    df = df.copy()
    
    # 1. 确保 t 是数值
    df[col_t] = pd.to_numeric(df[col_t], errors='coerce')
    
    # 2. 先按索引(ID) + t 排序，保证同 ID 内部时间升序
    df = df.sort_values([col_t]).sort_index(kind='stable')
    
    # 3. 计算位移
    pos_cols = ['Position X', 'Position Y', 'Position Z']
    delta = df.groupby(level=0)[pos_cols].diff().fillna(0)  # 第一行补 0
    
    df['Displacement X'] = delta['Position X']
    df['Displacement Y'] = delta['Position Y']
    df['Displacement Z'] = delta['Position Z']
    df['Displacement^2'] = (delta ** 2).sum(axis=1)
    
    return df


'''*******************************************************'''
#######################################整理-检查细胞消失出现的时空关系
'''*******************************************************'''
workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy4  = os.path.join(workpy, '碰撞检测与预测')
workpy5  = os.path.join(workpy, '细胞失踪重定位')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

N=1
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy5, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
#############################################绘图展示细胞首次、最后的出现位置，颜色根据不同需要上色
#IDtime(point,list(dict.fromkeys(ID)))##list(dict.fromkeys(ID))给ID去重
#IDtime(point,list(set(point.loc[:,'ID'])),save_path=workpath+'\图片\ID时刻汇总.pdf')
point=df.rename(columns={'Time': 't'})
point['ID']=point.index
point=fix_displacement(point,col_t='t')##col_t
os.makedirs(save_dir+'\图片', exist_ok=True) 
###############################################显示细胞首次出现位置（二维、三维、交互三个函数）：
plot_initial_positions(point,save_path=save_dir+'\图片\细胞出现位置XY.pdf')
plot_initial_positions_3D(point,save_path=save_dir+'\图片\细胞出现位置3D.pdf')
plot_initial_positions_interactive_plotly(point,save_path=save_dir+'\图片\细胞出现位置3D.html')
plot_initial_positions_interactive_plotly_T(point,save_path=save_dir+'\图片\细胞出现位置3D初始时间标记.html')

plot_initial_positions(point,save_path=save_dir+'\图片\细胞消失位置XY.pdf',switch='last')
plot_initial_positions_3D(point,save_path=save_dir+'\图片\细胞消失位置3D.pdf',switch='last')
plot_initial_positions_interactive_plotly(point,save_path=save_dir+'\图片\细胞消失位置3D.html',switch='last')
plot_initial_positions_interactive_plotly_T(point,save_path=save_dir+'\图片\细胞消失位置3D初始时间标记.html',switch='last')

#####美化细胞出现位置XY
plot_initial_positions_ad(point,switch='first',
                           palette='plasma',
                           dark_bg=False,
                           density_fill=True,
                           density_alpha=0.4,
                           n_levels=20,
                           show_labels=False,
                           marker_size=50,
                           save_path=save_dir+'\图片\细胞出现位置XY_美化.pdf')
plot_initial_positions_ad(point,switch='last',
                           palette='plasma',
                           dark_bg=False,
                           density_fill=True,
                           density_alpha=0.4,
                           n_levels=20,
                           show_labels=False,
                           marker_size=50, title='Final 2D Positions with Density',
                           save_path=save_dir+'\图片\细胞消失位置XY_美化.pdf')

# 2. 计算--折线图
dc = density_curve_by_distance(point,
                              switch='first',
                              bin_width=3)
# 3. 画图
sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=dc, x='Distance_to_Border', y='Count',
            marker='o', ax=ax, color='darkviolet')
ax.fill_between(dc['Distance_to_Border'], dc['Count'],
               alpha=0.3, color='darkviolet')
ax.set_title('Cell Count vs Distance to Image Border (First Appearance)')
ax.set_xlabel('Distance to Border (μm)')
ax.set_ylabel('Number of Cells')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '图片', '细胞出现位置边缘距离折线图.pdf'), bbox_inches='tight',facecolor='white')
plt.show()
# 2. 计算--折线图
dc = density_curve_by_distance(point,
                              switch='last',
                              bin_width=3)
# 3. 画图
sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(7,4))
sns.lineplot(data=dc, x='Distance_to_Border', y='Count',
            marker='o', ax=ax, color='darkviolet')
ax.fill_between(dc['Distance_to_Border'], dc['Count'],
               alpha=0.3, color='darkviolet')
ax.set_title('Cell Count vs Distance to Image Border (Disappearance)')
ax.set_xlabel('Distance to Border (μm)')
ax.set_ylabel('Number of Cells')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '图片', '细胞消失位置边缘距离折线图.pdf'), bbox_inches='tight',facecolor='white')
plt.show()

###堆沙图
cumul=stacked_cumul_distance_edge3(point,
                                    switch='first',      # 出现事件
                                    edge_dist_col='Distance to Image Border XYZ',
                                    time_bin=1,save_path=os.path.join(save_dir, '图片','细胞出现位置与边缘距离随时间堆沙图.pdf'))
cumul.to_csv(os.path.join(save_dir, '图片','细胞出现位置与边缘距离随时间堆沙图_默认阈值.csv'))
cumul=stacked_cumul_distance_edge3(point,
                                    switch='last',      # 出现事件
                                    edge_dist_col='Distance to Image Border XYZ',
                                    time_bin=1,save_path=os.path.join(save_dir, '图片','细胞消失位置与边缘距离随时间堆沙图.pdf'))
cumul.to_csv(os.path.join(save_dir, '图片','细胞消失位置与边缘距离随时间堆沙图.csv'))

#############################################提琴图、均值折线图展示两类距离之间三种计算的差异
# 计算距离
initial_distances = calculate_initial_distances(point)
final_distances = calculate_final_distances(point)
initial_to_final_distances = calculate_initial_to_final_distances(point)
distance_df = pd.DataFrame({
    'initial_distances': pd.Series(initial_distances),
    'final_distances': pd.Series(final_distances),
    'initial_to_final_distances': pd.Series(initial_to_final_distances)
})
distance_df.to_csv(os.path.join(save_dir, '图片','细胞出现消失位置间最小距离.csv'))
# 绘制提琴图
plot_violin_plots(initial_distances, final_distances, initial_to_final_distances,save_path=save_dir+'\图片\细胞出现位置比较_提琴图.pdf')
## 绘制提琴图--美化
plot_violin_plots_ad(initial_distances, final_distances, initial_to_final_distances,save_path=save_dir+'\图片\细胞出现位置比较_提琴图_美化.pdf')

border_values = [10,15, 20,25,30,35,40 ] 
plot_mean_distances_vs_border(point, border_values,save_path=save_dir+'\图片\不同border距离下细胞出现位置均值比较.pdf')
###美化
plot_mean_distances_vs_border_ad(point, border_values,save_path=save_dir+'\图片\不同border距离下细胞出现位置均值比较_美化.pdf')
#############################################计算每个时刻细胞的移动距离及其统计参数
u=calculate_continuous_distances(point)
plot_violin_and_statistics(u,save_path=save_dir+'\图片\细胞步长.pdf')

gap_df = detect_time_gaps(point,
                          time_col='t',
                          track_col='TrackID',
                          style='light',save_dir=str(os.path.join(save_dir, '图片')))
gap_df.to_csv(os.path.join(save_dir, '图片','存在GAP的细胞轨迹信息统计.csv'))

#############################################排序目标细胞的出现时刻，还原轨迹
gapIDs=gap_df['TrackID'].unique().tolist()
ID=gapIDs[0]
get_sorted_timestamps(point,ID )
plot_3d_trajectory(point, ID,save_path=save_dir+'\图片\ '+ID+'轨迹还原.pdf')



##########################################检查细胞最近的消失出现时间差
match_df = nearest_appear_after_disappear(point,
                                          id_col='ID',
                                          time_col='t',                   # 可选
                                          border_col='Distance to Image Border XY')
match_df.to_csv(os.path.join(save_dir, '图片','检查细胞最近的消失出现时间差.csv'))
# 查看时间差分布
import seaborn as sns, matplotlib.pyplot as plt
sns.histplot(match_df['delta_t'], kde=True, color='#3c78d8')
plt.xlabel('Δt (appear - disappear)')
plt.title('Time gap to nearest appearance')
plt.savefig(os.path.join(save_dir, '图片','检查细胞最近的消失出现时间差.pdf'), bbox_inches='tight',facecolor='white')
plt.show()
plot_delta_t_dist(match_df,save_path=os.path.join(save_dir, '图片','检查细胞最近的消失出现时间差_细分.pdf'))

#####轨迹碎片化展示
plot_track_length_violin(point,time_col='t',save_path=os.path.join(save_dir, '图片','轨迹时长分布.pdf'))