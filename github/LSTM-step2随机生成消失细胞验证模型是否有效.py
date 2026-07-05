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
    
def EMD_Right(data,parN=0):
    #data
    #parN=0
    data_par=np.array(data.iloc[:,parN])
    #par_name=data.columns[parN]
    #time_index = np.arange(len(data_par))  # 时间索引
    # 执行EMD分解
    emd = EMD()
    IMFs = emd.emd(data_par, np.arange(1, len(data_par)+1))  # 分解为10个IMFs
    # 可视化原始数据和 IMFs
    n_imfs = len(IMFs)  # IMF 的数量
    IMFdf=pd.DataFrame(IMFs)
    # 转置 DataFrame
    IMFdf = IMFdf.transpose()
    return IMFdf

def EMD_recue(IMFdfnew):
    # 还原数据
    reconstructed_data_new = np.zeros_like(IMFdfnew.index, dtype=float)
    # 累加IMFs
    for i in range(IMFdfnew.shape[0]):
        for imfn in IMFdfnew.columns:  # 假设 imfn 是列名
            reconstructed_data_new[i] += IMFdfnew[imfn].iloc[i]
    return reconstructed_data_new

def LSTM_right(data,window_size=3,forecast_horizon=10):
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    #scaled_data_all=scaled_data
    #scaled_data=scaled_data[:-10,:]
    scaled_data=scaled_data############使用前输入的数据是已知的，所以不需要留一部分来进行预测
    # 设置滑动窗口和预测步长
    #window_size =30  # 窗口大小
    #forecast_horizon = 10  # 未来要预测的时间步数
    # 准备训练数据（滚动滑动窗口）
    X_train = []
    y_train = []
    for i in range(len(scaled_data) - window_size - 1):
        X_train.append(scaled_data[i:i + window_size])  # 滑动窗口数据
        y_train.append(scaled_data[i + window_size])  # 下一步预测
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # 检查数据维度
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    # 构建LSTM模型
    model = Sequential()
    model.add(Input(shape=(window_size, data.shape[1])))  # 输入层
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=data.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    # 滚动预测
    last_window = scaled_data[-window_size:].copy()  # 使用最后一个窗口进行初始预测
    predicted_values = []
    for i in range(forecast_horizon):
        # 调整窗口形状，进行预测
        input_window = last_window.reshape(1, window_size, data.shape[1])
        predicted = model.predict(input_window)
        # 保存预测结果
        predicted_values.append(predicted[0])
        # 更新窗口，移除第一个时间步，添加新预测值
        last_window = np.vstack((last_window[1:], predicted))
    # 将预测结果反标准化
    predicted_values = np.array(predicted_values)
    predicted_values = scaler.inverse_transform(predicted_values)
    # 转换为DataFrame
    predicted_df = pd.DataFrame(predicted_values, columns=data.columns)
    return predicted_df

def compress_identity_pca(df,Time='Time'):
    """
    用分块 PCA 把 14 维核心特征压缩为 3 个主成分，并把 Time 列一并返回。
    """
    cell_identity_features = [
        'Area', 'Volume', 'Sphericity',
        'Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C',
        'Ellipticity (oblate)', 'Ellipticity (prolate)',
        'Average Speed', 'Net Displacement', 'Confinement Ratio', 'Arrest Coefficient', 'MSD',
        'Ellipsoid Axis A X', 'Ellipsoid Axis A Y', 'Ellipsoid Axis A Z'
    ]
    morph_cols  = cell_identity_features[:8]
    motion_cols = cell_identity_features[8:13]
    orient_cols = cell_identity_features[13:]

    scaler = StandardScaler()
    pca    = PCA(n_components=1, random_state=42)

    def block_pca(X_block):
        X_std = scaler.fit_transform(X_block)
        return pca.fit_transform(X_std).flatten()

    X_raw = df[cell_identity_features]

    df_reduced = pd.DataFrame(index=df.index)
    df_reduced['morph_PC1']  = block_pca(X_raw[morph_cols])
    df_reduced['motion_PC1'] = block_pca(X_raw[motion_cols])
    df_reduced['orient_PC1'] = block_pca(X_raw[orient_cols])
    df_reduced[Time]       = df[Time].values

    return df_reduced


def find_continuous_60pct_cells(df, time_col='Time', min_ratio=0.):
    """
    返回连续出现 ≥min_ratio 全局时刻的 TrackID 列表
    """
    global_times = df[time_col].drop_duplicates().sort_values().tolist()
    N_global = len(global_times)

    results = []
    for tid, group in df.groupby(level=0):          # level=0 即 TrackID
        times = group[time_col].sort_values().tolist()
        if not times:
            continue

        # 最长上升连续子序列
        longest = 1
        cur_len = 1
        for i in range(1, len(times)):
            if times[i] == times[i-1] + 1:
                cur_len += 1
                longest = max(longest, cur_len)
            else:
                cur_len = 1

        ratio = longest / N_global
        if ratio >= min_ratio:
            results.append(tid)

    return results
def sample_stable_trackids(df, time_col='Time', n_times=3, window=5):
    """
    随机选择 n_times 个时刻，并返回在这些时刻前后都至少有 `window` 个时刻的 TrackID 列表。

    参数
    ----
    df : DataFrame
        行索引为 TrackID，必须含 time_col
    time_col : str
        时间列名
    n_times : int
        要抽取的时刻数量
    window : int
        前后所需的最小时刻数（默认 5）

    返回
    ----
    dict
        {随机时刻: [满足条件的 TrackID 列表]}
    """
    # 全局有序时间列表（去重升序）
    all_t = df[time_col].drop_duplicates().sort_values().tolist()
    min_t, max_t = all_t[0], all_t[-1]

    # 候选时刻：前后至少 window 步
    valid_t = [t for t in all_t
               if (t - min_t) >= window * (all_t[1] - all_t[0]) and
               (max_t - t) >= window * (all_t[1] - all_t[0])]

    if len(valid_t) < n_times:
        raise ValueError("可用时刻不足，无法抽出指定数量。")

    chosen_t = random.sample(valid_t, n_times)

    # 结果字典
    result = {}
    for t in chosen_t:
        # 前后 window 步的时间范围
        idx = all_t.index(t)
        start_t = all_t[max(0, idx - window)]
        end_t   = all_t[min(len(all_t) - 1, idx + window)]

        # 前后都存在的 TrackID
        before = set(df[df[time_col].between(start_t, t, inclusive='neither')].index)
        after  = set(df[df[time_col].between(t, end_t, inclusive='neither')].index)
        stable = before & after
        result[t] = list(stable)

    return result


def split_before_after_with_drop(df, target_time,
                                 time_col='Time',
                                 window=5,
                                 max_drop=6):
    """
    1) 前后窗口内 **每个 TrackID 至少出现 `window` 个不同时间点**；
    2) 返回：
       - before_df：完整前窗口数据
       - after_df：后窗口数据，每个细胞随机删 0~max_drop 个最早时刻
    """

    # 全局有序时间点
    all_t = df[time_col].drop_duplicates().sort_values().tolist()
    idx_target = all_t.index(target_time)
    if idx_target < window or idx_target + window >= len(all_t):
        raise ValueError(f"{target_time} 前后不足 {window} 步")

    start_before = all_t[idx_target - window]
    end_before   = all_t[idx_target - 1]
    start_after  = all_t[idx_target]
    end_after    = all_t[idx_target + window]

    # 计算每个细胞在前后窗口内的**唯一时刻数**
    before_cnt = (df[df[time_col].between(start_before, end_before)]
                  .groupby(level=0)[time_col]
                  .nunique())
    after_cnt  = (df[df[time_col].between(start_after, end_after)]
                  .groupby(level=0)[time_col]
                  .nunique())

    # 仅保留前后都满足 ≥ window 的细胞
    valid_ids = set(before_cnt[before_cnt >= window].index) & \
                set(after_cnt[after_cnt >= window].index)

    # 构建子集
    before_df = df[df.index.isin(valid_ids) & (df[time_col] <= end_before)]

    after_df = df[df.index.isin(valid_ids) & (df[time_col] >= start_after)]

    # -------- 随机丢弃，但保证至少保留 window 帧 --------
    def _random_drop(g):
        # 最多能丢多少帧而不跌破 window
        max_can_drop = max(0, len(g) - window)
        drop_n = random.randint(0, min(max_drop, max_can_drop))
        return g.sort_values(time_col).iloc[drop_n:]

    after_df = after_df.groupby(level=0).apply(_random_drop).reset_index(drop=True)
    return before_df, after_df




def attach_close_after_ids(Ubefore, Uafter,
                           x='Position X', y='Position Y', z='Position Z',
                           id_col='ID', t_col='t', speed_col='Average_Speed',
                           out_col='Matched_After_IDs'):
    """
    每行 Ubefore → [满足动态半径的 Uafter.ID 列表]
    仅新增一列，由 out_col 控制列名。
    """
    pos_b = Ubefore[[x, y, z]].astype(float).values
    pos_a = Uafter[[x, y, z]].astype(float).values
    t_b   = Ubefore[t_col].astype(float).values
    t_a   = Uafter[t_col].astype(float).values
    v_b   = Ubefore[speed_col].astype(float).values
    id_a  = Uafter[id_col].values

    matched = []
    for i, (pos, v, t_cur) in enumerate(zip(pos_b, v_b, t_b)):
        delta_t = t_a - t_cur
        radius  = np.maximum(1.5 * v * delta_t, 15 * delta_t)
        dist    = np.linalg.norm(pos_a - pos, axis=1)
        mask    = dist <= radius
        matched.append(id_a[mask].tolist())

    # 只写一次，且列名由参数控制
    Ubefore[out_col] = matched
    return Ubefore


def merge_overlapping_ids(df, id_col='ID', match_col='Matched_After_IDs'):
    """
    按交集合并行
    返回：DataFrame 两列 ['Before_IDs', 'Matched_After_IDs']
    """
    # 1) 建图：节点 = 原始 ID
    G = nx.Graph()
    for _, row in df.iterrows():
        ids_in_row = row[id_col]
        targets = set(row[match_col])
        # 把当前行所有 ID 连成完全图
        for u in [ids_in_row]:
            for v in [ids_in_row]:
                G.add_edge(u, u)
            for t in targets:
                G.add_edge(u, t)

    # 2) 连通分量 → 合并行
    components = list(nx.connected_components(G))

    records = []
    for comp in components:
        before_ids = list(comp)
        after_ids  = list(set().union(
            *[row[match_col] for _, row in df.iterrows() if row[id_col] in comp]
        ))
        records.append({'Before_IDs': before_ids, 'Matched_After_IDs': after_ids})

    return pd.DataFrame(records)


def str_list_to_list(df):
    """
    把 DataFrame 中所有字符串形式的列表转为真正的 Python 列表。
    不影响非列表字符串的列。
    """
    for col in df.select_dtypes(include='object'):
        try:
            df[col] = df[col].apply(ast.literal_eval)
        except (ValueError, SyntaxError):
            pass  # 跳过无法转换的列
    return df

def predict_backward_dict(df_reduced, before_ids, params, n_steps=10):
    """
    对指定 ID 列表向后预测 n_steps 步，返回两层字典：
        {ID: {param: {time: value}}}
    """
    # 统一为列表
    before_ids = np.atleast_1d(before_ids)

    pred_dict = {}  # {ID: {param: {time: value}}}
    time_dict = {}  # {ID: 时间轴}

    for bid in before_ids:
        cell_df = df_reduced.loc[bid].sort_values('Time')
        last_t  = cell_df['Time'].iloc[-1]
        pred_t  = np.arange(last_t + 1, last_t + n_steps + 1)

        pred_dict[bid] = {}
        for par in params:
            par_idx = df_reduced.columns.get_loc(par)
            imf_df  = EMD_Right(cell_df, par_idx)
            pred    = np.full(n_steps, np.nan) if imf_df.shape[1] == 0 else \
                      EMD_recue(LSTM_right(imf_df, 3, n_steps))
            pred_dict[bid][par] = dict(zip(pred_t, pred))

        time_dict[bid] = pred_t

    return pred_dict, time_dict

def predict_forward_dict(df_reduced, after_ids, params, n_steps=10):
    """
    对指定 ID 列表向前（backward-in-time）预测 n_steps 步，
    返回两层字典：{ID: {param: {time: value}}}
    """
    # 统一为列表
    after_ids = np.atleast_1d(after_ids)

    pred_dict = {}  # {ID: {param: {time: value}}}
    time_dict = {}  # {ID: 时间轴}

    for aid in after_ids:
        cell_df = df_reduced.loc[aid].sort_values('Time')

        # 1. 反向时间序列
        reversed_df = cell_df.iloc[::-1].reset_index(drop=True)

        first_t = cell_df['Time'].iloc[0]          # 已知最早时间
        pred_t_raw = np.arange(first_t - 1, first_t - n_steps - 1, -1)  # 倒序
        pred_t_final = pred_t_raw[::-1]         # 真正想要的过去时间（正序）

        pred_dict[aid] = {}
        for par in params:
            par_idx = df_reduced.columns.get_loc(par)
            imf_df = EMD_Right(reversed_df, par_idx)

            # 2. LSTM 预测（在倒序时间轴上）
            pred_rev = np.full(n_steps, np.nan) if imf_df.shape[1] == 0 else \
                       EMD_recue(LSTM_right(imf_df, 3, n_steps))

            # 3. 将预测结果再翻转 + 对应到正确时间
            pred_final = pred_rev[::-1]
            pred_dict[aid][par] = dict(zip(pred_t_final, pred_final))

        time_dict[aid] = pred_t_final

    return pred_dict, time_dict


# ---------- 评价指标 ----------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.nan if mask.sum() == 0 else np.mean(np.abs(y_true[mask] - y_pred[mask]))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.nan if mask.sum() == 0 else np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def evaluate_prediction(
        pred_dict: dict,
        pred_t: dict,
        df_reduced_true: pd.DataFrame,   # 行索引 = ID（可重复），含列 'Time'
        true_ids: list,
        param: Union[str, None] = None,
        metric: str = "mae"
) -> pd.DataFrame:
    """
    计算预测轨迹与真实轨迹之间的误差矩阵。

    支持：
    - pred_ids 与 true_ids 长度可以不同；
    - 返回 DataFrame，行 = 预测 ID，列 = 真实 ID；
    - 可选择单参数或所有参数平均。

    Parameters
    ----------
    pred_dict : dict
        {pred_id: {param: {time: value}}}
    pred_t : dict
        {pred_id: np.ndarray}  预测所用时间戳
    df_reduced_true : pd.DataFrame
        行索引为重复 ID，含列 'Time' 及参数列
    true_ids : list
        候选真实 ID 列表
    param : str or None, default None
        - None  -> 所有参数平均误差
        - 'xxx' -> 仅计算该参数误差
    metric : {'mae', 'rmse'}, default 'mae'

    Returns
    -------
    score_df : pd.DataFrame
        行索引 = pred_id，列索引 = true_id，元素 = 误差值
    """
    metric_func = {"mae": mae, "rmse": rmse}[metric]

    pred_ids = list(pred_dict.keys())
    params_all = list(next(iter(pred_dict.values())).keys())

    # 需要计算的参数
    params_iter = params_all if param is None else [param]

    # 预生成矩阵
    score_mat = pd.DataFrame(index=pred_ids, columns=true_ids, dtype=float)

    for pred_id in pred_ids:
        pred_times = pred_t[pred_id]

        for true_id in true_ids:
            # 取出真实轨迹
            true_df = df_reduced_true.loc[[true_id]].sort_values("Time")
            if true_df.empty:
                score_mat.loc[pred_id, true_id] = np.nan
                continue

            errs = []
            for p in params_iter:
                pred_series = pd.Series(pred_dict[pred_id][p])  # index = time
                true_series = (true_df[true_df["Time"].isin(pred_times)]
                               .set_index("Time")[p]
                               .reindex(pred_series.index))

                errs.append(metric_func(true_series.values, pred_series.values))

            score_mat.loc[pred_id, true_id] = np.nanmean(errs)

    return score_mat

def rank_same_id(score_mat: pd.DataFrame) -> pd.DataFrame:
    """
    对 score_mat 每行升序排序，返回同名列的排名。
    结果：行索引与原表相同，单列名为 'Rank'。
    """
    rank_df = pd.DataFrame(index=score_mat.index, columns=['Rank'])

    for rid in score_mat.index:
        row = score_mat.loc[rid]                    # 取出该行
        sorted_row = row.sort_values()              # 升序
        try:
            rank = sorted_row.index.get_loc(rid) + 1   # 1-based
        except KeyError:
            rank = None
        rank_df.loc[rid, 'Rank'] = rank

    return rank_df.astype('Int64')   # Int64 可存 None

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
    
    

'''*******************************'''
###############################################################
# ######运行获得细胞消失模拟数据
workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy6  = os.path.join(workpy, '细胞失踪重定位')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
N=1
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy6, filename)
save_dir_test=os.path.join(save_dir, 'test')
os.makedirs(save_dir, exist_ok=True) 
os.makedirs(save_dir_test, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
events_path = os.path.join(save_dir, 'unique_events.csv')
events=pd.read_csv(events_path, index_col=None)
point=df.rename(columns={'Time': 't'})
point['ID']=point.index

# 使用示例
qualified_tracks = find_continuous_60pct_cells(df, time_col='Time', 
                                               min_ratio=0.5)
df = df[df.index.isin(qualified_tracks)]
df['ID']=df.index

##################开始，逐时刻模拟
for time in  (8,10,13,15,18,20):
    before, after = split_before_after_with_drop(df, target_time=time, 
                                                 window=5, max_drop=3)
    after.index=after['ID']

    point=before.rename(columns={'Time': 't'})
    point['ID']=point.index
    u=calculate_nearest_id_distances(point)
    # 创建新的索引列
    u['Index'] = u['ID'].astype(str) + '-' + u['t'].astype(str)
    point['Index'] = point['ID'].astype(str) + '-' + point['t'].astype(str)
    # 设置索引
    u.set_index('Index', inplace=True)
    point.set_index('Index', inplace=True)
    pa=('Volume', 'Ellipticity (oblate)', 'Ellipticity (prolate)','Area', 'Sphericity', 'Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C', 'Average Speed', 'Net Displacement', 'Confinement Ratio', 'Arrest Coefficient', 'MSD', 'Ellipsoid Axis A X', 'Ellipsoid Axis A Y', 'Ellipsoid Axis A Z')
    Ubefore=pd.concat([u, point.loc[:,pa]],axis=1)

    point=after.rename(columns={'Time': 't'})
    point['ID']=point.index
    u=calculate_nearest_id_distances(point)
    # 创建新的索引列
    u['Index'] = u['ID'].astype(str) + '-' + u['t'].astype(str)
    point['Index'] = point['ID'].astype(str) + '-' + point['t'].astype(str)
    # 设置索引
    u.set_index('Index', inplace=True)
    point.set_index('Index', inplace=True)
    Uafter=pd.concat([u, point.loc[:,pa]],axis=1)


    last_moment = Ubefore.groupby('ID')['t'].transform(max)
    Ubefore['Last_Moment'] = last_moment
    first_moment = Uafter.groupby('ID')['t'].transform(min)
    Uafter['First_Moment'] = first_moment

    # 添加一个标记列，用于区分细胞是消失还是出现
    Ubefore['Event_Type'] = np.where(Ubefore['t'] == Ubefore['Last_Moment'], 0,  # 消失
                                         np.nan) # 其他时刻标记为 NaN
    Uafter['Event_Type'] = np.where(Uafter['t'] == Uafter['First_Moment'], 1,  # 出现
                                         np.nan)  # 其他时刻标记为 NaN


    # 计算每个细胞的平均速度并存储在新列中
    average_speeds = Ubefore.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
    # 如果 data 里已存在同名列，则重命名新列
    if 'Average_Speed' in Ubefore.columns:
        average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
    Ubefore = Ubefore.merge(average_speeds, on='ID')
    Ubefore.index=Ubefore['ID']
    # 2. 只保留细胞消失时刻对应的行
    filtered_df_before = Ubefore.dropna(subset=['Event_Type']).copy()

    # 计算每个细胞的平均速度并存储在新列中
    average_speeds = Uafter.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
    # 如果 data 里已存在同名列，则重命名新列
    if 'Average_Speed' in Uafter.columns:
        average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
    Uafter = Uafter.merge(average_speeds, on='ID')
    Uafter.index=Uafter['ID']
    # 2. 只保留细胞出现时刻对应的行
    filtered_df_after = Uafter.dropna(subset=['Event_Type']).copy()


    matched_before = attach_close_after_ids(filtered_df_before, filtered_df_after)
    out_df = merge_overlapping_ids(matched_before)
    matched_before.head()
    out_df.head()

    matched_before.to_csv(os.path.join(save_dir_test, 't='+str(time)+'matched_before.csv'))
    out_df.to_csv(os.path.join(save_dir_test, 't='+str(time)+'out_df.csv'))
    ##matched_before:直接搜集相邻时空范围
    ##out_df：在matched_before基础上，将接近的消失出现时间汇总


'''*****************************'''
###################################################################开始检测效果
###########################检测细胞消失重定位效果
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
df['ID']=df.index

time=20
for time in (20,30,40,50,60,70,80,100):
    out_df=pd.read_csv(os.path.join(save_dir_test, 't='+str(time)+'out_df.csv'),index_col=0)
    matched_before=pd.read_csv(os.path.join(save_dir_test, 't='+str(time)+'matched_before.csv'),index_col=0)
    out_df = str_list_to_list(out_df)
    matched_before = str_list_to_list(matched_before)
    before, after = split_before_after_with_drop(df, target_time=time, 
                                                 window=5, max_drop=3)
    after.index=after['ID']
    df_reduced_before=compress_identity_pca(before,Time='Time')
    df_reduced_after=compress_identity_pca(after,Time='Time')
    params=df_reduced_before.columns.drop('Time')

    save_dir_test_t=os.path.join(save_dir_test, 'time='+str(time))
    save_dir_test_t_out=os.path.join(save_dir_test_t, 'out_df')
    save_dir_test_t_matched=os.path.join(save_dir_test_t, 'matched')
    #os.makedirs(save_dir_test_t_out, exist_ok=True) 
    #os.makedirs(save_dir_test_t_matched, exist_ok=True) 


    # ---------- 预存容器 ----------
    back_list, fwd_list = [], []
    match_records = []   # 新增：保存无需计算的行

    for n in range(out_df.shape[0]):##matched_before,out_df
        before_ids = out_df['Before_IDs'][n]##matched_before,out_df
        after_ids  = out_df['Matched_After_IDs'][n]##matched_before,out_df

        # 1. 处理 None 或单元素列表
        if before_ids is None or after_ids is None:
            # 至少一方为 None → 直接失败
            match_records.append({'ID': before_ids, 'match': False})
            continue

        # 保证是列表/数组，方便统一处理
        before_ids = np.atleast_1d(before_ids)
        after_ids  = np.atleast_1d(after_ids)

        if len(before_ids) == 1 and len(after_ids) == 1:
            # 两个单元素直接比对
            match_records.append({'ID': before_ids[0],
                                  'match': bool(before_ids[0] == after_ids[0])})
            continue

        # 2. 其余情况继续原有流程
        # ---------- 向后 ----------
        pred, t = predict_backward_dict(
            df_reduced_before, before_ids, params, n_steps=10)
        score_all = evaluate_prediction(
            pred, t,
            df_reduced_true=df_reduced_after,
            true_ids=before_ids,
            param=None)
        rank_result = rank_same_id(score_all)
        back_list.append(
            rank_result.reset_index()
                       .rename(columns={'index': 'ID', 'Rank': 'rank'})
                       .assign(eventid=n))

        # ---------- 向前 ----------
        fwd_pred, fwd_t = predict_forward_dict(
            df_reduced_after, after_ids, params, n_steps=10)
        score_all_fwd = evaluate_prediction(
            fwd_pred, fwd_t,
            df_reduced_true=df_reduced_before,
            true_ids=before_ids,
            param=None)
        rank_result_fwd = rank_same_id(score_all_fwd)
        fwd_list.append(
            rank_result_fwd.reset_index()
                           .rename(columns={'index': 'ID', 'Rank': 'rank'})
                           .assign(eventid=n))
        joblib.dump(
            {
                "pred": pred,
                "t":    t,
                "fwd_pred": fwd_pred,
                "fwd_t":    fwd_t,
                "score_all": score_all,
                "score_all_fwd":    score_all_fwd,
                "rank_result": rank_result,
                "rank_result_fwd":    rank_result_fwd,
            },
            os.path.join(save_dir_test_t_out, str(n)+"resultdata.pkl"),        # 文件名，可带 .pkl / .joblib
            compress=3           # 0-9，3 已足够快且小
        )
        # ↓↓↓ 新增：立即释放 + 强制回收 ↓↓↓
        del pred, t, fwd_pred, fwd_t, score_all, score_all_fwd, rank_result, rank_result_fwd
        
        gc.collect()
    # ---------- 合并 ----------
    df_rank = pd.concat(back_list, ignore_index=True)
    df_rank_fwd  = pd.concat(fwd_list,  ignore_index=True)
    # ---------- 无需计算的情况 ----------
    df_match_flag = pd.DataFrame(match_records)
    df_rank.to_csv(os.path.join(save_dir_test_t_out, "df_rank.csv"))
    df_rank_fwd.to_csv(os.path.join(save_dir_test_t_out, "df_rank_fwd.csv"))
    df_match_flag.to_csv(os.path.join(save_dir_test_t_out, "df_match_flag.csv"))
    del df_reduced_before, df_reduced_after
    gc.collect()



######################################统计结果
import pandas as pd
from pathlib import Path

# 1. 找到所有 df_rank.csv
csv_files = Path(workpy6).rglob('CX.csv/**/out_df/df_rank.csv')   # ** 递归任意层级
# 2. 一次性读取并拼接
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
# 3. table 效果：统计第二列（这里列名是 'rank'）
freq = df_all['rank'].value_counts().sort_index()
csv_match_files = Path(workpy6).rglob('CX.csv/**/out_df/df_match_flag.csv')   # ** 递归任意层级
# 2. 一次性读取并拼接
df_match_all = pd.concat([pd.read_csv(f) for f in csv_match_files], ignore_index=True)
count_true = df_match_all.iloc[:, 2].sum()
print(freq[1]/df_all.shape[0])
print(count_true/df_match_all.shape[0])
print((freq[1]+count_true)/(df_all.shape[0]+df_match_all.shape[0]))

print(df_all.shape[0]+df_match_all.shape[0])


