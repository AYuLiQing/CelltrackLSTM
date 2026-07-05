workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy6  = os.path.join(workpy, '预测结果整理图片')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy6, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0] 


'''****************************'''
##########################################扩展数据：
def line_insert(df: pd.DataFrame,
                       value_col: str,
                       insert_n: int,
                       time_col: str = 'Time') -> pd.DataFrame:
    """
    每【1.0 单位时间】插入 insert_n 个点（不含两端）
    示例：insert_n=3 → 步长 0.25；insert_n=1 → 步长 0.5
    """
    if insert_n <= 0:
        return df[[time_col, value_col]].copy()

    times = df[time_col].to_numpy()
    vals  = df[value_col].to_numpy()
    new_t, new_v = [], []

    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i+1]
        v0, v1 = vals[i], vals[i+1]
        dt = t1 - t0
        # 每单位时间插入 insert_n 个点 → 总份数 = dt * insert_n
        n_seg = int(np.round(dt * insert_n))
        if n_seg <= 0:          # 极小步长保护
            n_seg = 1
        seg_t = np.linspace(t0, t1, n_seg + 1)[1:-1]   # 去掉两端真实值
        seg_v = np.linspace(v0, v1, n_seg + 1)[1:-1]   # 线性插值（可换成 OT）

        # 先写左端真实值
        new_t.append(t0)
        new_v.append(v0)
        # 再写中间插入点
        for t, v in zip(seg_t, seg_v):
            new_t.append(t)
            new_v.append(v)

    # 尾巴
    new_t.append(times[-1])
    new_v.append(vals[-1])

    return pd.DataFrame({time_col: new_t, value_col: new_v})

from ot import wasserstein_1d          # 需要：pip install pot
def conservative_multivariate_interpolation(df, feature_cols, time_col='Time', insert_n=3):
    """
    保守的多变量插值方法--三次样条
    先对每个特征单独插值，然后确保它们之间的一致性
    """
    # 提取时间序列
    times = df[time_col].to_numpy()
    unique_times = np.unique(times)
    
    # 创建需要插值的时间点
    full_time_points = []
    for i in range(len(unique_times) - 1):
        t_start, t_end = unique_times[i], unique_times[i+1]
        n_segments = int(np.round((t_end - t_start) * insert_n))
        if n_segments <= 0:
            n_segments = 1
        
        segment_times = np.linspace(t_start, t_end, n_segments + 2)
        full_time_points.extend(segment_times[:-1])  # 不包括终点
    
    # 添加最后一个时间点
    full_time_points.append(unique_times[-1])
    full_time_points = np.sort(np.unique(full_time_points))
    
    # 对每个特征进行单独插值
    interpolated_data = {}
    for feature in feature_cols:
        # 按时间分组并计算平均值
        feature_vals = []
        feature_times = []
        for t in unique_times:
            mask = times == t
            if np.any(mask):
                feature_vals.append(np.mean(df[feature][mask]))
                feature_times.append(t)
        
        # 使用三次样条插值
        if len(feature_times) > 1:
            interp_func = interpd(feature_times, feature_vals, kind='cubic', 
                                  fill_value='extrapolate')
            interpolated_data[feature] = interp_func(full_time_points)
        else:
            # 如果只有一个时间点，使用常数插值
            interpolated_data[feature] = np.full_like(full_time_points, feature_vals[0])
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        time_col: full_time_points
    })
    
    for feature in feature_cols:
        result_df[feature] = interpolated_data[feature]
    
    return result_df

import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, interp1d, CubicSpline
from scipy.optimize import linprog
from ot import emd
from scipy.spatial.distance import cdist


def insert_meth(data: pd.DataFrame,
               pts_per_unit: int,
               method: str = 'rbf',
               plot: bool = False):
    """
    对原始时间轴做插值扩展，支持 4 种内核 + 可视化开关。

    参数
    ----
    data : pd.DataFrame
        原始数据，index 为时间轴（浮点/整数均可）。
    pts_per_unit : int
        单位长度内插入点数，决定新轴密度。
    method : {'rbf', 'spline', 'linear', 'ot'}, optional
        插值方法，默认 'rbf'。
    plot : bool, optional
        是否弹出可视化，默认 False（不弹图）。

    返回
    ----
    pd.DataFrame
        插值后的数据，index 为新的密集时间轴。
    """
    # ---------- 1. 构造新时间轴 ----------
    t_min, t_max = data.index.min(), data.index.max()
    T = t_max - t_min
    N = int(T * pts_per_unit) + 1
    new_time = np.linspace(t_min, t_max, num=N)

    # ---------- 2. 按方法插值 ----------
    old_time = np.array(data.index)
    interp_df = pd.DataFrame(index=new_time)

    for col in data.columns:
        y = data[col].values

        if method == 'rbf':
            rbf = Rbf(old_time, y, function='gaussian')
            interp_df[col] = rbf(new_time)

        elif method == 'spline':
            cs = CubicSpline(old_time, y)
            interp_df[col] = cs(new_time)

        elif method == 'linear':
            f = interp1d(old_time, y, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            interp_df[col] = f(new_time)

        elif method == 'ot':
            # 1-D 最优传输：将原分布搬到新轴
            cost = cdist(old_time.reshape(-1, 1),
                         new_time.reshape(-1, 1),
                         metric='euclidean')
            a = np.ones(len(old_time)) / len(old_time)  # 原测度
            b = np.ones(len(new_time)) / len(new_time)  # 目标测度
            gamma = emd(a, b, cost)                     # 传输矩阵
            # 加权平均得到插值
            interp_df[col] = (gamma.T @ y) / gamma.sum(axis=0)

        else:
            raise ValueError("method 必须为 'rbf'|'spline'|'linear'|'ot'")

    # ---------- 3. 可视化（可选） ----------
    if plot:
        plt.figure(figsize=(15, 5))
        cols_to_plot = min(15, data.shape[1])
        for i in range(cols_to_plot):
            plt.subplot(3, 5, i + 1)
            plt.plot(new_time, interp_df.iloc[:, i],
                     label=f'Param_{i+1}', color='green')
            # 把原始点也画上，方便对比
            plt.scatter(old_time, data.iloc[:, i],
                        color='black', s=10, zorder=5)
            plt.title(f'Param_{i+1}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    return interp_df

def plot_wasserstein_insert(original_df, extended_df,
                            value_col='Area',
                            time_col='Time',
                            figsize=(14, 5),
                            save_path=None):
    """
    美观绘制：原始 vs Wasserstein 插值后曲线
    仅可视化，无数据修改
    """
    # 统一风格
    orig_c, interp_c = '#1f77b4', '#ff7f0e'

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 原始数据
    ax.plot(original_df[time_col], original_df[value_col],
            color=orig_c, lw=2.5, label='Original', marker='o', markersize=4)

    # 插值后数据
    ax.plot(extended_df[time_col], extended_df[value_col],
            color=interp_c, lw=2, alpha=0.8, label='Wasserstein Insert')

    # 真实点突出
    ax.scatter(original_df[time_col], original_df[value_col],
               color=orig_c, s=30, zorder=5, edgecolor='white', linewidth=1)

    # 美化
    ax.set_title(f'{value_col}  DOT', fontsize=16, weight='bold')
    ax.set_xlabel(time_col, fontsize=13)
    ax.set_ylabel(value_col, fontsize=13)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    plt.show()

IDs=df.index.drop_duplicates()
ID=IDs[0]
touchcellpoint=df.loc[ID].sort_values(by='Time')

#U=line_insert(touchcellpoint, 'Area', insert_n=4)
#u = touchcellpoint[touchcellpoint['Time'] != 5].reset_index(drop=True)
data=touchcellpoint.drop(columns='cluster')
time=data['Time']
data = data.set_index('Time')
U=insert_meth(data, pts_per_unit=4, method='spline',  plot=False)###rbf,spline,linear,ot
U['Time']=U.index
data['Time']=data.index
par='Area'
plot_wasserstein_insert(data,U,par,save_path= os.path.join(save_dir, ID+par+'参数插值spline.pdf'))

u = data.drop(index=5)
U=insert_meth(u, pts_per_unit=4, method='rbf',  plot=False)
U['Time']=U.index
plot_wasserstein_insert(u,U,'Area')

data = data.set_index('Time')
rbf_df=insert_meth(data, pts_per_unit=4, method='rbf',  plot=False)###rbf,spline,linear,ot
rbf_df['Time']=rbf_df.index
data['Time']=data.index
data = data.set_index('Time')
linear_df=insert_meth(data, pts_per_unit=4, method='linear',  plot=False)###rbf,spline,linear,ot
linear_df['Time']=linear_df.index
data['Time']=data.index
data = data.set_index('Time')
ot_df=insert_meth(data, pts_per_unit=4, method='ot',  plot=False)###rbf,spline,linear,ot
ot_df['Time']=ot_df.index
data['Time']=data.index
data = data.set_index('Time')
spline_df=insert_meth(data, pts_per_unit=4, method='spline',  plot=False)###rbf,spline,linear,ot
spline_df['Time']=spline_df.index
data['Time']=data.index
plot_compare_curves_final(
    data,          # 原始
    spline_df,     # 样条
    rbf_df,        # RBF
    linear_df,     # 线性
    value_col=par,
    labels=['Original', 'RBF', 'Spline', 'Linear'],
    save_path=os.path.join(save_dir, ID+par+'参数插值all.pdf'),
    ext='pdf')



import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_compare_curves_final(orig_df, *other_dfs,
                              value_col='Area',
                              time_col='Time',
                              labels=None,
                              figsize=(6.8, 2.6),
                              save_path=None,
                              ext='pdf',
                              dpi=600):
    n_total = 1 + len(other_dfs)
    if n_total > 4:
        raise ValueError('最多 4 条曲线（原始+3）')
    if labels is None:
        labels = ['Original'] + [f'Series-{i+1}' for i in range(len(other_dfs))]

    # 莫兰迪色系：雾蓝 + 低饱和对照
    palette = ['#5B8EA7', '#D62728', '#6A9F58', '#C58038'][:n_total]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 1. 原始：雾蓝粗实线 + 大圆点（底层）
    ax.plot(orig_df[time_col], orig_df[value_col],
            color=palette[0], lw=1.6, label=labels[0], zorder=3)
    ax.scatter(orig_df[time_col], orig_df[value_col],
               color=palette[0], s=35, zorder=4, marker='o',
               edgecolors='white', linewidths=0.6)

    # 2. 插值：细虚线 + 稀疏符号（Linear 置顶）
    dash_styles = [(4, 2), (6,3), (2, 4)]  # 节距不同
    markers     = ['+', 'x', '*']              # 符号
    for i, df in enumerate(other_dfs, 1):
        z = 6 if i == 1 else 3                # Linear 置顶
        ax.plot(df[time_col], df[value_col],
                color=palette[i], lw=0.9,
                dashes=dash_styles[i-1],
                label=labels[i], zorder=z)
        # 稀疏符号（每 20 个点采 1 个）
        step = max(1, len(df) // 20)
        ax.scatter(df[time_col].iloc[::step], df[value_col].iloc[::step],
                   color=palette[i], s=18, zorder=z,
                   marker=markers[i-1], linewidths=0.5,
                   edgecolors='white', alpha=0.9)

    # 3. 期刊级美化
    ax.set_title(f'{value_col} Comparison', fontsize=10, weight='bold', pad=6)
    ax.set_xlabel(time_col, fontsize=9, labelpad=2)
    ax.set_ylabel(value_col, fontsize=9, labelpad=2)
    ax.legend(frameon=False, fontsize=8, loc='best', ncol=min(4, n_total))
    ax.grid(alpha=0.08)          # 极淡网格
    sns.despine(ax=ax)

    # 4. 保存
    if save_path:
        out_file = f'{save_path}.{ext}'
        plt.savefig(out_file, dpi=dpi if ext == 'png' else None,
                    bbox_inches='tight', transparent=True)
        print(f'图表已保存 → {out_file}')
    plt.show()


#############################通过删除一个时刻的真实值并通过还原来判断优劣
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import numpy as np

def calc_metrics(y_true, y_pred):
    y_true, y_pred = y_true.dropna(), y_pred.dropna()
    r2_orig = r2_score(y_true, y_pred)
    pred_shift = y_pred - y_pred.mean() + y_true.mean()
    r2_shift = r2_score(y_true, pred_shift)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nrmse = rmse / y_true.std()
    return {'R2': r2_orig, 'R2_shift': r2_shift, 'r': r,
            'MAPE': mape, 'RMSE': rmse, 'NRMSE': nrmse}
def calc_metrics_df(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise metrics between two identical-shape DataFrames.
    Returns (n_metrics × n_columns) DataFrame indexed by metric name.
    """
    if df_true.shape != df_pred.shape:
        raise ValueError('Shapes do not match')
    if not (df_true.index.equals(df_pred.index) and df_true.columns.equals(df_pred.columns)):
        raise ValueError('Index or columns do not match')
    # 逐列计算并拼接
    res = {col: calc_metrics(df_true[col], df_pred[col])
           for col in df_true.columns}
    return pd.DataFrame(res).T   # 行=column, 列=metric

# ---------- 主循环 ----------
res_df = pd.DataFrame()
time_list = data['Time'].unique()
data=touchcellpoint
data=data.drop(columns='Time')
for t in time_list[1:-1]:                  # 跳过首尾，防止边界效应
    # 抹除
    masked = data.drop(index=t)
    # 插值
    interp = insert_meth(masked, pts_per_unit=4, method='rbf',  plot=False)
    # 只保留被抹除时刻的那一行真值 & 插值结果
    true_val = data.loc[t, 'Area']
    interp_val = interp.loc[t, 'Area']
    # 记录指标
    m = calc_metrics(true_val,interp_val)
    m['Time'] = t
    res_df = pd.concat([res_df, pd.Series(m).to_frame().T], ignore_index=True)

# ---------- 可视化 ----------
import seaborn as sns
import matplotlib.pyplot as plt

metrics_long = res_df.melt(id_vars='Time', var_name='Metric', value_name='Value')
plt.figure(figsize=(8, 5))
sns.boxplot(data=metrics_long, x='Metric', y='Value', palette='Set2')
sns.stripplot(data=metrics_long, x='Metric', y='Value', color='black', size=4, alpha=0.7)
plt.title('Wasserstein 单点抹除-插值误差分布', fontsize=16, weight='bold')
plt.ylabel('Score', fontsize=13)
plt.xlabel('')
plt.grid(axis='y', alpha=0.3)
sns.despine(trim=True)
plt.tight_layout() 
plt.show()


#############################通过删除全部时刻的真实值并通过最有传输还原来判断优劣
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import numpy as np
# 出现次数≥5 的 TrackID
IDs = df.index.value_counts().loc[lambda x: x >= 10].index.to_numpy()
# 从原 IDs 中无放回随机抽取 100 个
IDs = np.random.default_rng().choice(IDs, size=300, replace=False)
param=df.columns
param=param.drop('cluster')
#metrics_df_IDall=pd.DataFrame()
metrics_df_IDall = pd.DataFrame(
    columns=pd.MultiIndex.from_product([df.columns,
                                        ['R2','R2_shift','r','MAPE','RMSE','NRMSE']]),
    dtype=float
)
for ID in IDs:
    touchcellpoint=df.loc[ID].sort_values(by='Time')
    touchcellpoint=touchcellpoint.drop(columns='cluster')
    touchcellpoint_pre=touchcellpoint.iloc[:-5,:]

    data=touchcellpoint
    data=data.set_index('Time')
    #data=data.drop(columns='cluster')
    
    # ---------- 1. 高密度插值 ----------
    U = insert_meth(data, pts_per_unit=4, method='ot', plot=False) # 已扩展
    U['Time']=U.index
    # ---------- 2. 删除所有原始时刻 ----------
    orig_times = data.index
    no_true = U.drop(index=U.index[U.index.isin(orig_times)])

    # ---------- 3. 仅在原始时刻重新插值 ----------
    # 用 no_true 作为新“基线”，再次 wasserstein_insert
    pred_full = insert_meth(no_true, pts_per_unit=4, method='ot',  plot=False)   # 高密度2
    pred_full['Time']=pred_full.index
    # 提取与原始时刻最邻近的插值点（线性查找即可）
    pred_at_orig = (pred_full[pred_full['Time'].isin(orig_times)]
                    .sort_values('Time'))

    # 对齐顺序
    true_sorted = (data
                   .sort_values('Time'))
    true_sorted = true_sorted.iloc[1:-1].copy()        # 去掉头尾各一行 
    pred_at_orig=pred_at_orig.loc[true_sorted.index]
    pred_at_orig=pred_at_orig.drop(columns="Time")
    ## ---------- 4. 评估 ----------处理单个参数
    #metrics = calc_metrics(true_sorted['Area'], pred_at_orig['Area'])
    #metrics_df = pd.Series(metrics).to_frame().T
    #metrics_df.index = ['Wasserstein Re-predict']
    #metrics_df.index=[ID]
    #metrics_df_IDall=pd.concat([metrics_df_IDall,metrics_df],axis=0)
    # ---------- 4. 评估 ----------同时分开处理所有参数
    # 1. 逐列评估 → 小表 (columns=metric, index=column名)
    mini = calc_metrics_df(true_sorted, pred_at_orig)   # shape: n_cols × n_metrics

    # 2. 压平成单行 MultiIndex 列
    ser = mini.stack()          # (column, metric) -> value
    ser.name = ID               # 外层行索引

    # 3. 挂到总表
    metrics_df_IDall = pd.concat([metrics_df_IDall, ser.to_frame().T], axis=0)
metrics_df_IDall=metrics_df_IDall.drop(columns='Time')
metrics_df_IDall.to_csv(os.path.join(save_dir, 'ot插值效果评估_全部真实数据缺失重建_allParam.csv'))


# ---------- 5. 可视化 ----------
import seaborn as sns
import matplotlib.pyplot as plt
##################################################具体参数点线图
fig, ax = plt.subplots(figsize=(10, 5))
# 高密度基线（无真值）
ax.plot(no_true.index, no_true['Area'],
        color='gray', lw=1.5, alpha=0.6, label='Grid w/o true points')
# 原始真值
ax.scatter(true_sorted.index, true_sorted['Area'],
           color='#1f77b4', s=40, zorder=5, label='True values')
# 仅在原始时刻的推测值
ax.scatter(pred_at_orig.index, pred_at_orig['Area'],
           color='#d62728', s=40, marker='x', zorder=5, label='W-asserstein predict')
ax.set_title('Original Points Reconstructed by Wasserstein', fontsize=16, weight='bold')
ax.set_xlabel('Time', fontsize=13)
ax.set_ylabel('Area', fontsize=13)
ax.legend(frameon=False)
ax.grid(alpha=0.3)
sns.despine(ax=ax)
plt.tight_layout()
plt.show()

metrics_df

##################################################点评估指标箱线图
# 1. 长表化
metrics_long = metrics_df_IDall.reset_index() \
                               .melt(id_vars='index',
                                     var_name='Metric',
                                     value_name='Value')
# 2. 画图
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid', {'axes.edgecolor':'0.2'})
order = ['R2','R2_shift','r','MAPE','RMSE','NRMSE']
plt.figure(figsize=(10,4))
box = sns.boxplot(data=metrics_long, x='Metric', y='Value',
                  order=order, palette='Set2', width=0.55)
sns.stripplot(data=metrics_long, x='Metric', y='Value',
              order=order, color='black', size=3, alpha=0.7, jitter=True)
# 3. 标均值
for i, metric in enumerate(order):
    avg = metrics_long.query("Metric==@metric")['Value'].mean()
    plt.text(i, avg*1.06 if metric!='MAPE' else avg*1.18,
             f'{avg:.3f}', ha='center', va='bottom',
             fontsize=9, weight='bold', color='darkred')
# 4. 美化
plt.title('all-gap reconstruction test:'+'ot', fontsize=15, weight='bold', pad=15)
plt.ylabel('Score', fontsize=12)
plt.xlabel('')
sns.despine(trim=True)
plt.tight_layout()
# 5. 保存
fig_path = os.path.join(save_dir, 'rbf全缺失重建_指标箱线图.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()


##########################################交叉验证
# ---------- 留一法（保留端点） ----------
#data=data.set_index('Time')
data=data.drop(columns='cluster')
res_df = pd.DataFrame()
time_list=data.index
# 只循环中间时刻
for t in time_list[1:-1]:
    # 抹掉单点
    masked = data.drop(index=t)
    # 插值
    interp = insert_meth(masked, pts_per_unit=4, method='rbf',  plot=False)   # 高密度2
    # 取出被抹掉的那一行
    true_val  = data.loc[t, 'Area']
    interp_val = interp.loc[t, 'Area']
    # 计算指标（单点退化为单样本指标，仍用原函数即可）
    m = calc_metrics(pd.Series(true_val),pd.Series(interp_val))
    m['Time'] = t
    res_df = pd.concat([res_df, pd.Series(m).to_frame().T], ignore_index=True)
# ---------- 可视化 ----------
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

metrics_long = res_df.melt(id_vars='Time', var_name='Metric', value_name='Value')
order = ['R2','R2_shift','r','MAPE','RMSE','NRMSE']

plt.figure(figsize=(10,5))
sns.boxplot(data=metrics_long, x='Metric', y='Value', order=order,
            palette='Set2', width=0.6)
sns.stripplot(data=metrics_long, x='Metric', y='Value', order=order,
              color='black', size=4, alpha=0.7, jitter=True)

# 标均值
for i, metric in enumerate(order):
    avg = metrics_long.query("Metric==@metric")['Value'].mean()
    plt.text(i, avg*1.05 if metric!='MAPE' else avg*1.15,
             f'{avg:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.title('留一法交叉验证（保留端点）— 单点插值误差分布', fontsize=16, weight='bold', pad=15)
plt.ylabel('Score', fontsize=13)
plt.xlabel('')
sns.despine(trim=True)
plt.tight_layout()
plt.show()

##################### 收集所有预测结果
y_true_list = []
y_pred_list = []

for t in time_list[1:-1]:
    masked = data.drop(index=t)
    interp = Rbf_insert(masked, 4)
    
    y_true_list.append(data.loc[t, 'Area'])
    y_pred_list.append(interp.loc[t, 'Area'])

# 一次性计算指标save_dir
y_true = pd.Series(y_true_list)
y_pred = pd.Series(y_pred_list)

m = calc_metrics(y_true, y_pred)
print(m)




metrics_df_linear=pd.read_csv(os.path.join(save_dir,'Area', 'linear插值效果评估_全部真实数据缺失重建.csv'),index_col=0)
metrics_df_rbf=pd.read_csv(os.path.join(save_dir,'Area', 'rbf插值效果评估_全部真实数据缺失重建.csv'),index_col=0)
metrics_df_spline=pd.read_csv(os.path.join(save_dir,'Area', 'spline插值效果评估_全部真实数据缺失重建.csv'),index_col=0)
metrics_df_ot=pd.read_csv(os.path.join(save_dir,'Area', 'ot插值效果评估_全部真实数据缺失重建.csv'),index_col=0)
# ========== 1. 合并长表 ==========
df_list = [metrics_df_linear, metrics_df_rbf,
           metrics_df_spline, metrics_df_ot]
key_list = ['Linear', 'RBF', 'Spline', 'OT']
combined = []
for df, key in zip(df_list, key_list):
    tmp = df.copy()
    tmp['Method'] = key
    combined.append(tmp)
long_df = pd.concat(combined, ignore_index=True)
# ========== 2. 指定要看的指标 ==========
METRIC = 'NRMSE'          # 想画哪列就改这里
#['R2', 'R2_shift', 'r', 'MAPE', 'RMSE', 'NRMSE']

#######################################直接绘图
# ========== 3. 绘图 ==========
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.figure(figsize=(8,4))
# 提琴图
sns.violinplot(data=long_df, x='Method', y=METRIC,
               palette='Set2', width=0.8, inner=None)   # inner=None 去掉内部箱线
# 内部散点
sns.stripplot(data=long_df, x='Method', y=METRIC,
              color='black', size=3, alpha=0.7, jitter=True)
# 标均值
for i, method in enumerate(key_list):
    avg = long_df.query("Method==@method")[METRIC].mean()
    plt.text(i, avg*1.03 if METRIC!='MAPE' else avg*1.08,
             f'{avg:.3f}', ha='center', va='bottom',
             fontsize=10, weight='bold', color='darkred')
# 美化
plt.title(f'{METRIC} ', fontsize=15, weight='bold', pad=15)
plt.ylabel(METRIC, fontsize=12)
plt.xlabel('')
sns.despine(trim=True)
plt.tight_layout()
# ========== 4. 保存 ===
#plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

############################################绘图-去极大极小值
# ========== 0. 异常点裁剪 ==========
LOWER = long_df[METRIC].quantile(0.01)
UPPER = long_df[METRIC].quantile(0.99)
long_df_clip = long_df[(long_df[METRIC] >= LOWER) & (long_df[METRIC] <= UPPER)].copy()

# ========== 1. 绘图（其余代码完全不变） ==========
plt.figure(figsize=(8,4))
sns.violinplot(data=long_df_clip, x='Method', y=METRIC,
               palette='Set2', width=0.8, inner=None)
sns.stripplot(data=long_df_clip, x='Method', y=METRIC,
              color='black', size=3, alpha=0.7, jitter=True)

# 标均值（用裁剪后数据）
for i, method in enumerate(key_list):
    avg = long_df_clip.query("Method==@method")[METRIC].mean()
    plt.text(i, avg*1.03 if METRIC != 'MAPE' else avg*1.08,
             f'{avg:.3f}', ha='center', va='bottom',
             fontsize=10, weight='bold', color='darkred')

plt.title(f'{METRIC} Distribution comparison (truncated from 1% to 99%)', fontsize=15, weight='bold')
plt.ylabel(METRIC); plt.xlabel(''); sns.despine(trim=True)
plt.tight_layout()
save_path = os.path.join(save_dir, METRIC+'指标多种插值方法比较.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



############################################绘图-去极大极小值--美化
# 1. 计算 1σ 区间（±std）用于内侧标记
summary = (long_df_clip.groupby('Method')[METRIC]
                       .agg(['mean','std'])
                       .reindex(key_list))
summary['low']  = summary['mean'] - summary['std']
summary['high'] = summary['mean'] + summary['std']

# 2. 画布
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', rc={'grid.linewidth': 0.2,
                                     'axes.edgecolor': '.2'})

fig, ax = plt.subplots(figsize=(9, 5))

# 3. 横置提琴图（orient='h'）
sns.violinplot(data=long_df_clip, y='Method', x=METRIC,
               palette={'Linear':'#5c92c7',
                        'RBF':'#51a34f',
                        'Spline':'#d99439',
                        'OT':'#c44c34'},
               width=1.1, inner=None, orient='h', alpha=0.75)

# 4. 内侧 1σ 区间短横线
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.plot([row['low'], row['high']], [i, i],
            color='white', linewidth=3, zorder=10)
    ax.plot([row['low'], row['high']], [i, i],
            color='black', linewidth=1.2, zorder=11)

# 5. 均值红点 + 数值标签
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.scatter(row['mean'], i, color='crimson', s=80, zorder=12,
               marker='o', edgecolor='white', linewidth=1.5)
    ax.text(row['mean'], i+0.25, f"{row['mean']:.3f}",
            color='crimson', fontsize=10, weight='bold', ha='center')

# 6.  cosmetic
ax.set_title(f'{METRIC} Distribution comparison (truncated from 1% to 99%)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel(METRIC, fontsize=13)
ax.set_ylabel('')
ax.set_yticks(range(len(key_list)))
ax.set_yticklabels(key_list)
sns.despine(left=True, bottom=True)   # 全去边框
ax.grid(axis='x', alpha=0.3)          # 仅横向淡网格

# 7. 保存
save_path = os.path.join(save_dir, METRIC+'指标多种插值方法比较--美化.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



############################################绘图-去极大极小值--美化---简化
# 1. 计算均值 & ±1σ
summary = (long_df_clip.groupby('Method')[METRIC]
                       .agg(['mean', 'std'])
                       .reindex(key_list))
summary['low']  = summary['mean'] - summary['std']
summary['high'] = summary['mean'] + summary['std']

# 2. 通用排版参数（单位：pt）
plt.rcParams.update({
    'font.family'        : 'Times New Roman',
    'axes.labelsize'     : 9,
    'xtick.labelsize'    : 8,
    'ytick.labelsize'    : 8,
    'figure.titlesize'   : 10,
    'legend.fontsize'    : 8,
    'axes.linewidth'     : 0.5,
    'xtick.major.width'  : 0.5,
    'ytick.major.width'  : 0.5,
    'xtick.major.size'   : 2,
    'ytick.major.size'   : 2,
    'savefig.dpi'        : 600,      # 出版级
    'savefig.format'     : 'pdf',     # 矢量
    'savefig.transparent': True,
})

# 3. 画布
fig, ax = plt.subplots(figsize=(3.2, 2.3))   # 单栏宽度 ≈ 8.3 cm

# 4. 提琴图（黑白灰 + 酒红均值）
sns.violinplot(data=long_df_clip, y='Method', x=METRIC,
               order=key_list,
               color='lightgray',   # 琴身统一灰
               width=1.2,
               inner=None,
               orient='h',
               linewidth=0.6)

# 5. 内侧 ±1σ 区间
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.plot([row['low'], row['high']], [i, i],
            color='black', linewidth=1.2, solid_capstyle='butt')

# 6. 均值标记（酒红圆点 + 数值）
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.scatter(row['mean'], i,
               s=30, color='maroon', marker='o', zorder=10,
               edgecolors='white', linewidths=0.8)
    ax.text(row['mean'], i+0.35,
            f'{row["mean"]:.2f}',
            color='maroon', fontsize=7, weight='bold',
            ha='center', va='bottom')

# 7. 坐标轴
ax.set_xlabel(f'{METRIC}', fontsize=9, labelpad=2)
ax.set_ylabel('')                    # 方法名已刻在 y 轴
ax.set_yticks(range(len(key_list)))
ax.set_yticklabels(key_list, fontsize=8)
# 轴范围：1%-99% 截断后自动，或手动微调
ax.set_ylim(len(key_list)-0.5, -0.5)  # 反向让顺序与表格一致
ax.tick_params(axis='x', pad=1)
# 淡化上右边框
sns.despine(ax=ax, left=True, right=True, top=True)

# 8. 标题（可选，正文常写在图注）
# ax.set_title(f'{METRIC} comparison', fontsize=8, pad=6)

# 9. 保存
save_path = os.path.join(save_dir, METRIC+'指标多种插值方法比较--美化--简化.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()



N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
#df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy6, filename)
#os.makedirs(save_dir, exist_ok=True) 
#df = pd.read_csv(df_path, index_col=0)
#U=inspect_nan(df,'zero')
#df=U[0] 
# 用于收集所有数据
data_list = []
for folder in csv_files:
    file_path = os.path.join(save_dir, 'ot插值效果评估_全部真实数据缺失重建_allParam.csv')
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)  # 读取两层列索引
    # 计算每个参数-指标组合的均值
    mean_series = df.mean(axis=0)  # 结果是 MultiIndex Series
    # 将 Series 转换为 DataFrame，并添加数据来源
    temp_df = mean_series.reset_index()
    temp_df['source'] = folder
    temp_df.rename(columns={'level_0': 'param', 'level_1': 'metric', 0: 'mean_value'}, inplace=True)
    data_list.append(temp_df)
# 合并所有数据
all_data = pd.concat(data_list, ignore_index=True)
# 透视表：行是 metric + source，列是 param，值是 mean_value
result = all_data.pivot_table(index=['metric', 'source'], columns='param', values='mean_value')
# 如果你想把 metric 和 source 作为两层行索引：
result.index.names = ['metric', 'source']
result.to_csv(os.path.join(workpy6, 'ot插值效果评估_全部真实数据缺失重建_allParam_ALLdata.csv'))

##################汇总
# 定义处理方式列表
methods = ['ot', 'spline', 'linear', 'rbf']
# 用于收集所有数据
all_data_list = []
for method in methods:
    filename = f'{method}插值效果评估_全部真实数据缺失重建_allParam_ALLdata.csv'
    file_path = os.path.join(workpy6, filename)
    # 读取 CSV（假设已有两层行索引：metric, source）
    df = pd.read_csv(file_path, index_col=[0, 1], header=0)
    # 增加一个层级：处理方式
    df['method'] = method
    df = df.set_index('method', append=True)
    # 重新排序索引层级为：metric, method, source
    df = df.reorder_levels(['metric', 'method', 'source'])
    all_data_list.append(df)
# 合并所有数据
final_df = pd.concat(all_data_list)
final_df.to_csv(os.path.join(workpy6, '四种方法插值效果评估_全部真实数据缺失重建_allParam_ALLdata.csv'))


###########可视化---热图
# ---------- 去极值 & 方向标准化 ----------
from scipy.stats import mstats
def winsorize_series(s, limits=(0.02, 0.02)):   # 2%-98% 缩尾
    return mstats.winsorize(s, limits=limits)
def direct_normalize(s, bigger_is_better: bool):
    """将序列映射到 0-1，方向统一为“越大越好”"""
    s = winsorize_series(s)          # 1. 去极值
    s = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s
    return s if bigger_is_better else 1 - s   # 2. 方向统一
# 1. 将行索引展开为列
df = final_df.reset_index()
# 2. 定义每个指标的“方向”：True 表示越大越好，False 表示越小越好
metric_direction = {
    'R2': True,
    'R2_shift': True,
    'r': True,
    'NRMSE': False,
    'RMSE': False,
    'MAPE': False
    # 添加更多指标
}
# 3. 获取所有指标
metrics = df['metric'].unique()
metrics=pd.array([ 'R2', 'R2_shift', 'r','RMSE', 'MAPE', 'NRMSE'])
# 4. 设置子图布局
n_metrics = len(metrics)
fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.5 * n_metrics), sharex=True)
if n_metrics == 1:
    axes = [axes]
# 只处理这 3 个指标；其余指标保持原始值
do_process = {'R2': True, 'R2_shift': True, 'r': True}
for metric in do_process:
    mask = df['metric'] == metric
    cols = df.columns.difference(['metric', 'method', 'source'])  # 仅参数列
    df.loc[mask, cols] = df.loc[mask, cols].apply(
        lambda x: direct_normalize(x, bigger_is_better=True), axis=0
    )
# =============== 低饱和蓝-白-红 无格线热图 ===============
import matplotlib.colors as mcolors
# 5++. 高亮边缘-浅珊瑚-白-淡青蓝
custom_colors = [
    (0.41, 0.74, 0.85, 1),   # 珊瑚再 +15 % 亮度，饱和度×1.25
    (1, 1, 1, 1),
    (1.00, 0.66, 0.54, 1)   # 淡青蓝再 +15 % 亮度，饱和度×1.25
]
'''
# 5++-. 加深边缘-浅珊瑚-白-淡青蓝
custom_colors = [
    (0.32, 0.68, 0.78, 1),   # 珊瑚亮度-8 %，饱和度×1.35
    (1, 1, 1, 1),
    (0.92, 0.58, 0.46, 1)   # 淡青蓝亮度-8 %，饱和度×1.35
]
'''
soft_cmap = mcolors.LinearSegmentedColormap.from_list('soft_blue_coral', custom_colors, N=256)
for ax, metric in zip(axes, metrics):
    subset = df[df['metric'] == metric]
    pivot = subset.set_index(['method', 'source']).drop(columns='metric')
    # ---- 截断上限到 80 分位（仅 RMSE/MAPE） ----
    if metric in ('RMSE', 'MAPE'):
        q80 = pivot.quantile(0.4)
        # 逐列建立截断 norm：vmin=列最小，vmax=列0.8分位
        norm = Normalize(vmin=pivot.min().min(), vmax=q80.max())
        # 超过 0.8 分位的值强制映射到 1.0（顶端色）
        pivot_clip = pivot.clip(upper=q80.max())
    else:
        norm = None
        pivot_clip = pivot
    cmap = soft_cmap if metric_direction[metric] else soft_cmap.reversed()
    n_skip = max(1, len(pivot.columns) // 8)
    x_labels = [label if i % n_skip == 0 else ''
                for i, label in enumerate(pivot.columns)]
    sns.heatmap(
        pivot_clip,
        cmap=cmap,
        annot=False,
        linewidths=0,
        cbar_kws={'label': metric, 'shrink': 0.5, 'aspect': 10},
        xticklabels=x_labels,
        yticklabels=True,
        ax=ax
    )
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8)
plt.subplots_adjust(hspace=0.08)
plt.tight_layout()
plt.savefig(os.path.join(workpy6, '四种方法插值效果评估_全部真实数据缺失重建_allParam_ALLdata.pdf'), 
            format='pdf', bbox_inches='tight')
plt.show()





