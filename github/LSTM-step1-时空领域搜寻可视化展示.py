import pandas as pd
import random
import networkx as nx
import ast
import numpy as np
from typing import Union
import joblib   
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from PyEMD import EMD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import gc
from scipy.spatial.distance import cdist




'''*******************************'''
###############################################################
# ######运行获得细胞消失模拟数据
#workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy0 = r'D:\LJY\CellPara\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy6  = os.path.join(workpy, '细胞失踪重定位')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy6, filename)
save_dir_test=os.path.join(save_dir, 'test')
os.makedirs(save_dir, exist_ok=True) 
os.makedirs(save_dir_test, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
#events_path = os.path.join(save_dir, 'unique_events.csv')
#events=pd.read_csv(events_path, index_col=None)
point=df.rename(columns={'Time': 't'})
point['ID']=point.index
df['ID']=df.index

time=20
ID='red_1000000011'

matched_before=pd.read_csv(os.path.join(save_dir_test, 't='+str(time)+'matched_before.csv'),index_col=0)
before_time=pd.read_csv(os.path.join(save_dir_test, 't='+str(time)+'before_time.csv'),index_col=0)
after_time=pd.read_csv(os.path.join(save_dir_test, 't='+str(time)+'after_time.csv'),index_col=0)
before=extract_before_deadline(before_time,df)
after=extract_before_deadline(after_time,df,meth=False)

after.index=range(0,after.shape[0])
target_df=after.loc[after.groupby('ID')['Time'].idxmin()]
target_df.index=target_df['ID']

ori_df=before.loc[ID].loc[before.loc[ID]['Time']==time-1]

sci_disc_full(ori_df,target_df,save_path=os.path.join(save_dir,'平面展示细胞空间距离步长.pdf'),
                        xlim_um=[math.floor(min(df['Position X'])),math.ceil(max(df['Position X']))],
                        ylim_um=[math.floor(min(df['Position Y'])), math.ceil(max(df['Position Y']))])

plot_3d_time_distance_landscape(ori_df, target_df,
                                speed_factor=3,
                                z_scale=3.0,
                                grid_res=80,save_path=os.path.join(save_dir,'三维展示时空距离.pdf'),
                                interp_method='cubic',colid=3)

plot_3d_time_distance_landscape_ad(ori_df, target_df,
                                speed_factor=3,
                                z_scale=3.0,
                                grid_res=80,view_azim = -45,
                                view_elev= 20, save_path=os.path.join(save_dir,'三维展示时空距离-美化.pdf'),
                                interp_method='cubic',colid=4)
plot_3d_time_distance_landscape_ad(ori_df, target_df,
                                speed_factor=3,
                                z_scale=3.0,save_path=os.path.join(save_dir,'俯视图展示时空距离-美化.pdf'),
                                grid_res=80,view_azim = -90,
                                view_elev= 20, view_mode='top',
                                interp_method='cubic',colid=4)






lstm_recommended_features = [
    # 核心运动特征
    'Position X', 'Position Y', 'Position Z',
    'Speed', 'Velocity X', 'Velocity Y', 'Velocity Z',
    # 方向性特征
    'Velocity Angle X', 'Velocity Angle Y', 'Velocity Angle Z',
    'Directional_AC_mean',
    # 行为特征
    'Chemotaxis Index', 'Arrest Coefficient',
    # （可选）
    'MSD', 'Confinement Ratio'
]
ori_df[lstm_recommended_features]
target_df[lstm_recommended_features]





###############

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
    
def extract_before_deadline(df1: pd.DataFrame,      # 索引=ID，列='Time'
                            df2: pd.DataFrame,      # 宽表，列名含 ID
                            time_col: str = 'Time',  # df2 里的时间列名
                            meth=True
                            ) -> pd.DataFrame:
    """
    返回 df2 中每个 ID 满足 df2[time_col] ≤ df1.loc[ID, 'Time'] 的行。
    """
    # 把截止时刻拼到 df2 上
    df2 = df2.merge(df1.rename(columns={'Time': 'deadline'}),
                    left_on='ID',
                    right_index=True,
                    how='left')
    # 过滤
    if meth:
        mask = df2[time_col] <= df2['deadline']
    else:
        mask = df2[time_col] > df2['deadline']
    return df2.loc[mask].drop(columns='deadline')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
def sci_disc_full(
        one_cell,
        Uafter=None,
        x='Position X',
        y='Position Y',
        t_col='Time',
        speed_col='Average Speed',
        vmax=None,
        fade_end=2.0,
        scale_um=50,
        n_contour=5,
        dt_contour=2.0,
        xlim_um=None,      # 手动 [xmin, xmax]，单位 µm
        ylim_um=None,      # 手动 [ymin, ymax]
        figsize=(70, 70),  # mm
        dpi=300,save_path=None):
    """
    返回 figure：轴范围可控，绝不显示不全
    """
    # ---- 数据 ----
    if hasattr(one_cell, 'iloc'):
        one_cell = one_cell.iloc[0]
    pos = one_cell[[x, y]].astype(float).values
    t0  = float(one_cell[t_col])
    v   = float(one_cell[speed_col])
    vmax = vmax or v
    max_dt = 10.0 if (Uafter is None or Uafter.empty) else (Uafter[t_col].max() - t0)
    r_max = max(1.5 * vmax * max_dt, 15 * max_dt)

    # ---- 坐标轴范围 ----
    if xlim_um is None:
        xlim_um = [pos[0] - r_max*fade_end, pos[0] + r_max*fade_end]
    if ylim_um is None:
        ylim_um = [pos[1] - r_max*fade_end, pos[1] + r_max*fade_end]
    x0, y0 = pos

    # ---- 画布 ----
    fig, ax = plt.subplots(figsize=(figsize[0]/25.4, figsize[1]/25.4), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(xlim_um)
    ax.set_ylim(ylim_um)

    # ---- 1. 径向渐变背景 ----
    n = 600
    xx = np.linspace(xlim_um[0], xlim_um[1], n)
    yy = np.linspace(ylim_um[0], ylim_um[1], n)
    X, Y = np.meshgrid(xx, yy)
    D = np.sqrt((X - x0)**2 + (Y - y0)**2)

    d_norm = D / r_max
    alpha = np.clip((fade_end - d_norm) / (fade_end - 1.0), 0, 1)
    alpha[d_norm > fade_end] = 0

    colors = np.ones((256, 4))
    colors[:, 0] = np.linspace(0.031, 1, 256)
    colors[:, 1] = np.linspace(0.251, 1, 256)
    colors[:, 2] = np.linspace(0.506, 1, 256)
    colors[:, 3] = np.linspace(0.70, 0, 256)
    cmap = LinearSegmentedColormap.from_list('sci_blue', colors)

    ax.imshow(D, origin='lower', extent=[xlim_um[0], xlim_um[1], ylim_um[0], ylim_um[1]],
              cmap=cmap, alpha=alpha, zorder=0, aspect='auto')

    # ---- 2. 等高线 ----
    t_levels = np.arange(1, n_contour + 1) * dt_contour
    r_levels = np.maximum(1.5 * vmax * t_levels, 15 * t_levels)
    for r in r_levels:
        circle = plt.Circle((x0, y0), r, fill=False,
                            edgecolor='0.6', linewidth=0.5, zorder=2)
        ax.add_patch(circle)

    # ---- 3. 中心细胞 ----
    ax.scatter(x0, y0, s=36, c='#084081', edgecolors='black', linewidths=0.5, zorder=5)

    # ---- 4. 其余细胞 ----
    if Uafter is not None and not Uafter.empty:
        pts = Uafter[[x, y]].astype(float).values
        # 只保留在轴范围内的点
        mask = (pts[:, 0] >= xlim_um[0]) & (pts[:, 0] <= xlim_um[1]) & \
               (pts[:, 1] >= ylim_um[0]) & (pts[:, 1] <= ylim_um[1])
        pts = pts[mask]
        dist = np.linalg.norm(pts - pos, axis=1)
        gray = np.clip(dist / r_max, 0, 1)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, c=gray, cmap='Greys',
                   edgecolors='black', linewidths=0.5, zorder=3)

    # ---- 5. 比例尺 ----
    sb_x = xlim_um[0] + (xlim_um[1] - xlim_um[0]) * 0.05
    sb_y = ylim_um[0] + (ylim_um[1] - ylim_um[0]) * 0.05
    ax.plot([sb_x, sb_x + scale_um], [sb_y, sb_y], 'k', linewidth=0.5)
    ax.text(sb_x + scale_um/2, sb_y + (ylim_um[1]-ylim_um[0])*0.02, f'{scale_um} µm',
            ha='center', va='bottom', fontsize=8, fontfamily='sans-serif')

    # ---- 6. 科研简洁 ----
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('black')
    ax.set_xlabel('X (µm)', fontsize=8, fontfamily='sans-serif')
    ax.set_ylabel('Y (µm)', fontsize=8, fontfamily='sans-serif')
    ax.tick_params(axis='both', labelsize=8, width=0.5, length=2, colors='black')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

# -------------------------------------------------
# 展示细胞时空邻域
# -------------------------------------------------
def plot_3d_time_distance_landscape(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        speed_factor: float = 1.5,
        z_scale: float = 1,               # 依旧保持“压缩幅度”
        maxminxZ: float=5,
        grid_res: int = 60,
        interp_method: str = 'cubic',
        surf_alpha: float = 0.75,
        figsize: tuple = (10, 7),
        cmap_colors: tuple = ('#4169E1', '#FFD700'),  # ① 原版配色
        return_ax: bool = False, colid: int = 1,save_path=None,
        v='Average Displacement^2'):
    """
    绘制“时间-距离”三维地形网格图（含散点）
    """
    # ---- 1. 起点信息 ----
    start_x = float(df1['Position X'])
    start_y = float(df1['Position Y'])
    start_t = float(df1['Time'])
    start_v = float(df1[v]) * speed_factor

    # ---- 2. 距离 & 所需时间 T ----
    dx = df2['Position X'].to_numpy() - start_x
    dy = df2['Position Y'].to_numpy() - start_y
    dist = np.hypot(dx, dy)
    T = dist / start_v

    # ---- 3. 时间差 & Z 偏移 ----
    delta_t = df2['Time'].to_numpy() - start_t
    z_offset = (delta_t - T) * z_scale
    z_offset = np.clip(z_offset, -maxminxZ, maxminxZ)          # 压缩极端值

    # ---- 4. 插值网格 ----
    x, y = df2['Position X'].to_numpy(), df2['Position Y'].to_numpy()
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z_offset, (XI, YI), method=interp_method)

    # -------------------------------------------------
    # 5. 多方案连续渐变配色（colid=1…5）
    # -------------------------------------------------
    if colid == 1:
        nodes = np.linspace(0, 1, 6)
        custom_colors = ['#08203B', '#1E88E5', '#00E5FF',
                         '#FFEB3B', '#FFA726', '#E53935']
    elif colid == 2:
        nodes = np.array([0, 0.08, 0.35, 0.55, 0.75, 1])
        custom_colors = ['#08203B', '#6A0DAD', '#00E5FF',
                         '#FFEB3B', '#FFA726', '#FF6F61']
    elif colid == 3:
        nodes = np.array([0, 0.3, 0.5, 0.7, 0.85, 1])
        custom_colors = ['#003f5c', '#2DBECD', '#DFFF00',
                         '#FF9F1C', '#FF5E0E', '#A10D26']
    elif colid == 4:
        nodes = np.array([0, 0.25, 0.45, 0.65, 0.8, 1])
        custom_colors = ['#001845', '#0047AB', '#00B4D8',
                         '#FFD60A', '#FF8C00', '#E63946']
    elif colid == 5:
        nodes = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        custom_colors = ['#011627', '#3A506B', '#5BC0BE',
                         '#F9F871', '#FF9A1F', '#8B0000']
    else:
        raise ValueError('colid must be 1-5')

    cmap = colors.LinearSegmentedColormap.from_list(
        'multi_smooth', list(zip(nodes, custom_colors)), N=256)
    norm = colors.TwoSlopeNorm(vmin=-maxminxZ, vcenter=0, vmax=maxminxZ)

    # ---- 6. 绘图 ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 6.1 曲面
    surf = ax.plot_surface(XI, YI, ZI, cmap=cmap, norm=norm,
                           alpha=surf_alpha, rstride=1, cstride=1,
                           linewidth=0.1, antialiased=True)

    # 6.2 ② 把原始散点叠在曲面上
    ax.scatter(x, y, z_offset,
               c=z_offset, cmap=cmap, norm=norm,
               s=25, edgecolors='k', linewidths=0.3, zorder=5)  # zorder 确保在曲面之上

    fig.colorbar(surf, ax=ax, shrink=0.6, label='time-distance')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (Time difference mapping)')
    ax.set_title('Three-dimensional "time-distance" topographic map')
    plt.tight_layout()

    if return_ax:
        return ax
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



# -------------------------------------------------
# 科研风 3D/2D 地形图 · 强化散点 + 突出起点（完整可运行）
# -------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
from scipy.interpolate import griddata

# 白色科研风全局
rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.4,
    'grid.alpha': 0,
    'font.family': 'Arial',
    'font.size': 10,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


def plot_3d_time_distance_landscape_ad(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        speed_factor: float = 1.5,
        z_scale: float = 0.8,
        maxminxZ: float = 5,
        grid_res: int = 80,
        interp_method: str = 'cubic',
        surf_alpha: float = 0.80,
        figsize: tuple = (8, 6),
        return_ax: bool = False,
        v: str = 'Average Displacement^2',
        colid: int = 1,
        view_azim: float = 45,
        view_elev: float = 20,
        pt_size: int = 35,
        pt_glow: int = 3,
        start_mult: float = 2.5,
        start_lw: float = 1.2,save_path=None,
        view_mode: str = '3d'):  # <-- 新增参数
    """
    科研风 3D「时间-距离」地形图  or  2D 俯视图
    view_mode = '3d' | 'top'
    """
    # ---- 1. 起点信息 ----
    start_x = float(df1['Position X'])
    start_y = float(df1['Position Y'])
    start_t = float(df1['Time'])
    start_v = float(df1[v]) * speed_factor

    # ---- 2. 距离 & 所需时间 T ----
    dx = df2['Position X'].to_numpy() - start_x
    dy = df2['Position Y'].to_numpy() - start_y
    dist = np.hypot(dx, dy)
    T = dist / start_v

    # ---- 3. 时间差 & Z 偏移 ----
    delta_t = df2['Time'].to_numpy() - start_t
    z_offset = (delta_t - T) * z_scale
    z_offset = np.clip(z_offset, -maxminxZ, maxminxZ)

    # ---- 4. 插值网格 ----
    x, y = df2['Position X'].to_numpy(), df2['Position Y'].to_numpy()
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z_offset, (XI, YI), method=interp_method)

    # ---- 5. 配色 ----
    if colid == 1:
        nodes, custom_colors = np.linspace(0, 1, 6), ['#08203B', '#1E88E5', '#00E5FF', '#FFEB3B', '#FFA726', '#E53935']
    elif colid == 2:
        nodes = np.array([0, 0.08, 0.35, 0.55, 0.75, 1])
        custom_colors = ['#08203B', '#6A0DAD', '#00E5FF', '#FFEB3B', '#FFA726', '#FF6F61']
    elif colid == 3:
        nodes = np.array([0, 0.3, 0.5, 0.7, 0.85, 1])
        custom_colors = ['#003f5c', '#2DBECD', '#DFFF00', '#FF9F1C', '#FF5E0E', '#A10D26']
    elif colid == 4:
        nodes = np.array([0, 0.25, 0.45, 0.65, 0.8, 1])
        custom_colors = ['#001845', '#0047AB', '#00B4D8', '#FFD60A', '#FF8C00', '#E63946']
    elif colid == 5:
        nodes = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        custom_colors = ['#011627', '#3A506B', '#5BC0BE', '#F9F871', '#FF9A1F', '#8B0000']
    else:
        raise ValueError('colid must be 1-5')

    cmap = colors.LinearSegmentedColormap.from_list('multi_smooth', list(zip(nodes, custom_colors)), N=256)
    norm = colors.TwoSlopeNorm(vmin=-maxminxZ, vcenter=0, vmax=maxminxZ)

    # ---- 6. 绘图 ----
    fig = plt.figure(figsize=figsize)

    if view_mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        # 曲面
        surf = ax.plot_surface(XI, YI, ZI, cmap=cmap, norm=norm,
                               alpha=surf_alpha, rstride=1, cstride=1,
                               linewidth=0, shade=False)
        # 强化散点
        for k in range(pt_glow, 0, -1):
            ax.scatter(x, y, z_offset, c=z_offset, cmap=cmap, norm=norm,
                       s=pt_size + k * 8, alpha=0.18, edgecolors='none', zorder=5)
        ax.scatter(x, y, z_offset, c=z_offset, cmap=cmap, norm=norm,
                   s=pt_size, edgecolors='k', linewidths=0.25, zorder=6)
        # 起点
        for k in range(3, 0, -1):
            ax.scatter(start_x, start_y, 0, s=pt_size * start_mult + k * 15,
                       c='none', edgecolors='white', linewidths=start_lw + k * 0.5,
                       zorder=10, alpha=0.35)
        ax.scatter(start_x, start_y, 0, s=pt_size * start_mult,
                   c='white', edgecolors='black', linewidths=start_lw, zorder=11, alpha=1)

        ax.set_xlabel('X (μm)', labelpad=8)
        ax.set_ylabel('Y (μm)', labelpad=8)
        ax.set_zlabel('Z (time-distance)', labelpad=8)
        ax.set_title('3D time-distance landscape', pad=10, fontsize=12)
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.grid(True, alpha=0.15)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor('gray')
            axis.pane.set_linewidth(0.5)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)

    elif view_mode == 'top':
        # ========= 新增美化参数 =========
        contourf_alpha = 0.55      # 曲面透明度（越低越浅）
        glow_layers = 3            # 白色光晕层数
        base_size = pt_size        # 基准散点大小
        # 按高度分 5 档（可改）
        n_levels = 5
        quantiles = np.linspace(0, 1, n_levels + 1)
        breaks = np.quantile(z_offset, quantiles)
        breaks[-1] += 1e-6          # 保证最大值落入最后一档


        # 1. 离散 norm
        norm_disc = colors.BoundaryNorm(boundaries=breaks, ncolors=n_levels, clip=True)

        # 2. 越高端越亮的颜色列表
        bright_colors = [cmap(int(np.clip(i, 0, 255))) for i in np.linspace(0, 255, n_levels)]
        bright_cmap = colors.ListedColormap(bright_colors)
        # 越高端越亮的颜色列表（从原 cmap 抽 5 色，末端加亮）
        bright_colors = [cmap(int(np.clip(i, 0, 255))) for i in np.linspace(0, 255, n_levels)]
        bright_cmap = colors.ListedColormap(bright_colors)
        # ========= 绘图 =========
        ax = fig.add_subplot(111)
        # 1. 浅底曲面
        cf = ax.contourf(XI, YI, ZI, levels=80, cmap=cmap, norm=norm,
                         alpha=contourf_alpha, antialiased=True)
        # 2. 分档散点 + 白晕
        for idx in range(n_levels):
            mask = (z_offset >= breaks[idx]) & (z_offset < breaks[idx + 1])
            if not mask.any():
                continue
            # 越大、越亮、白边越粗
            scale = 1 + idx * 0.6
            edge_width = 0.3 + idx * 0.25
            current_size = base_size * scale
            # 白色光晕（从外到内）
            for g in range(glow_layers, 0, -1):
                ax.scatter(x[mask], y[mask],
                           s=current_size + g * 10,
                           c='none', edgecolors='white',
                           linewidths=edge_width + g * 0.4,
                           alpha=0.4 / g, zorder=5)
            # 主散点
            ax.scatter(x[mask], y[mask],
                       c=z_offset[mask], cmap=bright_cmap, norm=norm_disc,
                       s=current_size, edgecolors='white',
                       linewidths=edge_width, zorder=6)
        # 3. 起点（保持 3D 版风格）
        for k in range(3, 0, -1):
            ax.scatter(start_x, start_y,
                       s=base_size * start_mult + k * 15,
                       c='none', edgecolors='white',
                       linewidths=start_lw + k * 0.5,
                       alpha=0.35, zorder=10)
        ax.scatter(start_x, start_y,
                   s=base_size * start_mult,
                   c='white', edgecolors='black',
                   linewidths=start_lw, zorder=11)
        # 4. 其余装饰
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('2D top view of time-distance landscape', fontsize=12)
        ax.grid(True, alpha=0.15)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.55, pad=0.08)

    else:
        raise ValueError("view_mode must be '3d' or 'top'")

    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    if return_ax:
        return ax
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



