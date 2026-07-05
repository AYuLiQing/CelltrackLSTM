
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from PyEMD import EMD
import itertools
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import contextlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import gc, psutil

def distance_min_cell(pointXYZ,pointsdf):
    """
    找到与给定点空间距离最近的点及其距离。
    
    参数:
    pointXYZ -- 一个包含三个数值的列表，表示一个点的坐标。
    pointsdf -- 一个DataFrame，包含三列，每列对应一个坐标轴的数据，每行代表一个点。
    
    返回:
    nearest_point -- 与给定点空间距离最近的点的坐标。
    min_distance -- 最近点与给定点之间的距离。
    """
    # 确保pointXYZ是一个NumPy数组
    pointXYZ = np.array(pointXYZ)
    # 计算每个点与给定点的空间距离
    distances = np.sqrt(np.sum((pointsdf.values - pointXYZ) ** 2, axis=1))
    # 找到距离最小的点的索引
    min_distance_index = np.argmin(distances)
    # 获取最近点的坐标和距离
    nearest_point = pointsdf.iloc[min_distance_index].values
    min_distance = distances[min_distance_index]
    return min_distance_index,nearest_point, min_distance


from sklearn.linear_model import LinearRegression
def plot_r2(y_true, y_pred, title='R² Plot'):
    """
    绘制预测值与真实值的散点图，并显示最佳拟合线和置信区间。
    
    参数:
    y_true -- 真实值数组
    y_pred -- 预测值数组
    title -- 图表标题
    """
    # 计算R²值
    r2 = r2_score(y_true, y_pred)
    print(f"R²: {r2:.4f}")
    # 创建散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Forecast', s=10)
    # 使用线性回归找到最佳拟合线
    model = LinearRegression()
    model.fit(y_true.reshape(-1, 1), y_pred)
    y_pred_line = model.predict(y_true.reshape(-1, 1))
    # 绘制最佳拟合线
    plt.plot(y_true, y_pred_line, color='red', label='Least Squares Fitting')
    # 绘制置信区间
    residuals = y_pred - y_pred_line
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.fill_between(y_true, y_pred_line - 1.96 * std_residual, y_pred_line + 1.96 * std_residual, color='gray', alpha=0.2, label='Confidence Ellipse (Predicted)')
    # 设置图表标题和标签
    plt.title(title)
    plt.xlabel('True')
    plt.ylabel('Forecast')
    plt.legend()
    plt.grid(True) 
    # 显示图表
    plt.show()


def evaluate_and_visualize(A, B, a, b, metric_name='mse',
                           Aname='A',
                           Bname='B', save_path: str = None):
    """
    计算真实数据和预测数据之间的评价指标，并可视化为热图。
    参数:
        A, B : pd.DataFrame  真实数据
        a, b : pd.DataFrame  预测数据
        metric_name : str    评价指标名称
        save_path   : str    若提供，则将图片保存至此路径；默认 None
    返回:
        2×2 的指标 DataFrame
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 数值化
    A = A.apply(pd.to_numeric, errors='coerce')
    B = B.apply(pd.to_numeric, errors='coerce')
    a = a.apply(pd.to_numeric, errors='coerce')
    b = b.apply(pd.to_numeric, errors='coerce')

    row_labels=(Aname, Bname)
    col_labels=(Aname, Bname)
    # 计算指标
    results = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
    results.iloc[0, 0] = calculate_metric(A.values.ravel(), a.values.ravel(), metric_name)
    results.iloc[0, 1] = calculate_metric(A.values.ravel(), b.values.ravel(), metric_name)
    results.iloc[1, 0] = calculate_metric(B.values.ravel(), a.values.ravel(), metric_name)
    results.iloc[1, 1] = calculate_metric(B.values.ravel(), b.values.ravel(), metric_name)

    # 热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(results, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Evaluation Metrics ({metric_name.upper()})')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 
    return results

def calculate_moving_average(T, dataframe):
    #对DataFrame的每一列应用滑动窗口计算均线，并去除尾部的NaN值。
    #参数:
    #T -- 滑动窗口宽度
    #dataframe -- 输入的DataFrame，其中包含数值型数据
    #返回:
    #moving_avg_df -- 计算均线后的DataFrame
    # 对每一列应用滑动窗口计算均线
    moving_avg_df = dataframe.rolling(window=T).mean()
    # 去除尾部的NaN值
    moving_avg_df = moving_avg_df.dropna()
    return moving_avg_df


def select_tough_ID_T(IDT,dfdata,N):
    IDT=IDT[N]
    IDT=IDT.split("-")
    ID=IDT[0]
    T=IDT[1]
    dfdata=dfdata.loc[dfdata['ID']==int(ID),:]
    return dfdata

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


def truncate_data(data, N, from_head: bool = True):
    """
    返回一个新的 DataFrame 或 NumPy 数组，仅保留前 N 或后 N 行/元素。

    参数
    ----
    data : pd.DataFrame | np.ndarray
        输入数据。
    N : int
        保留的行数或元素个数。
    from_head : bool, default True
        True  → 保留前 N 行/元素  
        False → 保留后 N 行/元素

    返回
    ----
    pd.DataFrame | np.ndarray
        截取后的新对象。
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[:N].copy() if from_head else data.iloc[-N:].copy()
    elif isinstance(data, np.ndarray):
        return data[:N].copy() if from_head else data[-N:].copy()
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy array.")



def analyze_2x2_dataframe_Max(df):
    """
    分析2x2 DataFrame并根据值的位置和大小返回 -1, 0, 或 1。
    参数:
    df -- 一个2x2的DataFrame，索引为'A', 'B'，列为'a', 'b'
    返回:
    int -- 根据描述的规则返回 -1, 0, 或 1
    """
    # 确保输入是2x2 DataFrame
    if df.shape != (2, 2) :
        raise ValueError("DataFrame must be 2x2 with index ['A', 'B'] and columns ['a', 'b']")
    # 获取值
    A_a, A_b = df.iloc[0, 0], df.iloc[0, 1]
    B_a, B_b = df.iloc[1, 0], df.iloc[1, 1]
    # 找到最大和第二大的值
    values = [A_a, A_b, B_a, B_b]
    max_val = max(values)
    second_max_val = sorted(values, reverse=True)[1]
    # 根据规则确定返回值
    if (max_val == A_a and second_max_val == B_b) or (max_val == B_b and second_max_val == A_a):
        return 2
    elif (max_val == A_a and second_max_val != B_b) or (max_val == B_b and second_max_val != A_a):
        return 1
    elif (max_val == A_b and second_max_val != B_a) or (max_val == B_a and second_max_val != A_b):
        return -1
    elif (max_val == A_b and second_max_val == B_a) or (max_val == B_a and second_max_val == A_b):
        return -2
    else:
        return 0
    

def analyze_2x2_dataframe_Min(df):
    """
    分析2x2 DataFrame并根据值的位置和大小返回 -1, 0, 或 1。
    参数:
    df -- 一个2x2的DataFrame，索引为'A', 'B'，列为'a', 'b'
    返回:
    int -- 根据描述的规则返回 -1, 0, 或 1
    """
    # 确保输入是2x2 DataFrame
    if df.shape != (2, 2) :
        raise ValueError("DataFrame must be 2x2 with index ['A', 'B'] and columns ['a', 'b']")
    # 获取值
    A_a, A_b = df.iloc[0, 0], df.iloc[0, 1]
    B_a, B_b = df.iloc[1, 0], df.iloc[1, 1]
    # 找到最小和第二小的值
    values = [A_a, A_b, B_a, B_b]
    min_val = min(values)
    second_min_val = sorted(values, reverse=False)[1]
    # 根据规则确定返回值
    if (min_val == A_a and second_min_val == B_b) or (min_val == B_b and second_min_val == A_a):
        return 2
    elif (min_val == A_a and second_min_val != B_b) or (min_val == B_b and second_min_val != A_a):
        return 1
    elif (min_val == A_b and second_min_val != B_a) or (min_val == B_a and second_min_val != A_b):
        return -1
    elif (min_val == A_b and second_min_val == B_a) or (min_val == B_a and second_min_val == A_b):
        return -2
    else:
        return 0
def analyze_2x2_dataframe_Max_back(df):###从越小越正确出发，排除最大的匹配
    """
    分析2x2 DataFrame并根据值的位置和大小返回 -1, 0, 或 1。
    参数:
    df -- 一个2x2的DataFrame，索引为'A', 'B'，列为'a', 'b'
    返回:
    int -- 根据描述的规则返回 -1, 0, 或 1
    """
    # 确保输入是2x2 DataFrame
    if df.shape != (2, 2) :
        raise ValueError("DataFrame must be 2x2 with index ['A', 'B'] and columns ['a', 'b']")
    # 获取值
    A_a, A_b = df.iloc[0, 0], df.iloc[0, 1]
    B_a, B_b = df.iloc[1, 0], df.iloc[1, 1]
    # 找到最大和第二大的值
    values = [A_a, A_b, B_a, B_b]
    max_val = max(values)
    second_max_val = sorted(values, reverse=True)[1]
    # 根据规则确定返回值
    if (max_val == A_a and second_max_val == B_b) or (max_val == B_b and second_max_val == A_a):
        return -2
    elif (max_val == A_a and second_max_val != B_b) or (max_val == B_b and second_max_val != A_a):
        return -1
    elif (max_val == A_b and second_max_val != B_a) or (max_val == B_a and second_max_val != A_b):
        return 1
    elif (max_val == A_b and second_max_val == B_a) or (max_val == B_a and second_max_val == A_b):
        return 2
    else:
        return 0
##另一种分值越小越好的打分罗辑，同时加入最大值进行
def score_2x2(df: pd.DataFrame) -> int:
    """
    初始 0 分
    最小值在主对角线 +1，否则 -1
    最大值在非主对角线 +1，否则 -1
    """
    m = df.values.ravel()          # [0,0, 0,1, 1,0, 1,1]
    idx_min = np.argmin(m)
    idx_max = np.argmax(m)

    score = 0
    # 主对角线索引 0,3
    main = {0, 3}
    # 最小值
    score += 1 if idx_min in main else -1
    # 最大值
    score += 1 if idx_max not in main else -1
    return score
def fit_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, model

def calculate_metric(y_true, y_pred, metric_name):
    """
    计算指定的评价指标。
    参数:
    y_true -- 真实值数组
    y_pred -- 预测值数组
    metric_name -- 评价指标的名称，支持 'mse', 'mae', 'r2', 'pearson', 'spearman'
    返回:
    计算得到的评价指标值
    """
    if metric_name == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric_name == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == 'r2':
        return r2_score(y_true, y_pred)
    elif metric_name == 'pearson':
        return pearsonr(y_true, y_pred)[0]
    elif metric_name == 'spearman':
        return spearmanr(y_true, y_pred)[0]
    else:
        raise ValueError("Unsupported metric name. Choose from 'mse', 'mae', 'r2', 'pearson', 'spearman'.")



def extract_touch_pair_timeseries(df: pd.DataFrame,
                                  event_row: pd.Series,
                                  n: int=3,
                                  time_col: str = 'Time',
                                  feature_cols: list = None) -> tuple:
    """
    新增参数 n: int
    返回: A_sel, B_sel, touchcellT, prepost_flag (2×2 np.ndarray)
    其中 prepost_flag:
        [0,0] A_sel 在 touchcellT 前是否有 ≥n 个时刻 → 0/1
        [0,1] A_sel 在 touchcellT 后是否有 ≥n 个时刻 → 0/1
        [1,0] B_sel 在 touchcellT 前是否有 ≥n 个时刻 → 0/1
        [1,1] B_sel 在 touchcellT 后是否有 ≥n 个时刻 → 0/1
    """
    if time_col not in df.columns:
        raise ValueError(f'未找到时间列 {time_col}')

    trackA, trackB, touchcellT = event_row['TrackID_A'], event_row['TrackID_B'], event_row['Time']

    # 取出两条轨迹并按时间排序（保持你原先的逻辑）
    A_df = df.loc[trackA].copy()
    B_df = df.loc[trackB].copy()
    for obj in (A_df, B_df):
        if isinstance(obj, pd.Series):
            obj = obj.to_frame().T
    A_df = A_df.sort_values(by=time_col)
    B_df = B_df.sort_values(by=time_col)

    # 处理特征列（保持你原先的逻辑）
    if feature_cols is None:
        num_cols = set(A_df.select_dtypes(include=[np.number]).columns).intersection(
                    B_df.select_dtypes(include=[np.number]).columns)
        common = sorted(num_cols)
        if time_col not in common:
            common = [time_col] + common
        else:
            common = [time_col] + [c for c in common if c != time_col]
        feature_cols = common

    A_sel = A_df[feature_cols].reset_index(drop=True)
    B_sel = B_df[feature_cols].reset_index(drop=True)
    
    if 't' not in A_sel.columns:
        A_sel = A_sel.rename(columns={time_col: 't'})
    if 't' not in B_sel.columns:
        B_sel = B_sel.rename(columns={time_col: 't'})   
    # 统一时间列名


    # 计算 2×2 flag
    def flag_one(df_sel):
        t_series = df_sel['t'].values
        pre_cnt  = np.sum(t_series < touchcellT)
        post_cnt = np.sum(t_series > touchcellT)
        return [0 if pre_cnt < n else 1,
                0 if post_cnt < n else 1]

    flag_A = flag_one(A_sel)
    flag_B = flag_one(B_sel)
    prepost_flag = np.array([flag_A, flag_B], dtype=int)

    # 与原先保持一致的行索引命名
    A_sel.index = pd.Index([trackA] * len(A_sel), name='TrackID')
    B_sel.index = pd.Index([trackB] * len(B_sel), name='TrackID')

    return A_sel, B_sel, touchcellT, prepost_flag




def cell_identity_report(Apre, ApreValue, Aback, AbackValue,
                         Bpre, BpreValue, Bback, BbackValue):
    """
    一站式打印细胞身份连续性、区分性、准确性评估报告。
    所有输入可以是 0-D 的 numpy.ndarray，也可以是标量。
    """

    # ---------- 统一转成 float ----------
    Apre_f       = float(np.squeeze(Apre).mean())
    ApreValue_f  = float(np.squeeze(ApreValue).mean())
    Aback_f      = float(np.squeeze(Aback).mean())
    AbackValue_f = float(np.squeeze(AbackValue).mean())
    Bpre_f       = float(np.squeeze(Bpre).mean())
    BpreValue_f  = float(np.squeeze(BpreValue).mean())
    Bback_f      = float(np.squeeze(Bback).mean())
    BbackValue_f = float(np.squeeze(BbackValue).mean())

    # ---------- 计算指标 ----------
    Pre_Error_A = ApreValue_f - Apre_f
    Pre_Error_B = BpreValue_f - Bpre_f
    Back_Error_A = AbackValue_f - Aback_f
    Back_Error_B = BbackValue_f - Bback_f

    Continuity_A = AbackValue_f - ApreValue_f
    Continuity_B = BbackValue_f - BpreValue_f

    ICI_A = 1 - (Continuity_A / ((Continuity_A + AbackValue_f - BpreValue_f) + 1e-12))
    ICI_B = 1 - (Continuity_B / ((Continuity_B + BbackValue_f - ApreValue_f) + 1e-12))

    IDD_Pre = (ApreValue_f - BpreValue_f) / (
        abs(ApreValue_f - Apre_f) + abs(BpreValue_f - Bpre_f) + 1e-12)

    IDD_Back = (AbackValue_f - BbackValue_f) / (
        abs(AbackValue_f - Aback_f) + abs(BbackValue_f - Bback_f) + 1e-12)

    CAR_A = Continuity_A / (abs(Back_Error_A) + 1e-12)
    CAR_B = Continuity_B / (abs(Back_Error_B) + 1e-12)

    # ---------- 打印 ----------
    print("===== 1. 预测准确性 =====")
    print(f"A 细胞：预测值 {ApreValue_f:.3f}，真实值 {Apre_f:.3f}，绝对误差 {Back_Error_A:.3f}")
    print(f"B 细胞：预测值 {BpreValue_f:.3f}，真实值 {Bpre_f:.3f}，绝对误差 {Back_Error_B:.3f}")

    print("\n===== 2. 身份连续性指数 (ICI) =====")
    print(f"ICI_A = {ICI_A:.3f}  （>0.5 表示自关联强）")
    print(f"ICI_B = {ICI_B:.3f}  （>0.5 表示自关联强）")

    print("\n===== 3. 身份区分度 (IDD) =====")
    print(f"IDD_Pre  = {IDD_Pre:.3f}  （>1 表示细胞间差异大于误差）")
    print(f"IDD_Back = {IDD_Back:.3f}  （>1 表示细胞间差异大于误差）")

    print("\n===== 4. 连续性-准确性比率 (CAR) =====")
    print(f"CAR_A = {CAR_A:.3f}  （<1 优，>1 预警）")
    print(f"CAR_B = {CAR_B:.3f}  （<1 优，>1 预警）")
        # ---------- 返回字典 ----------
    metrics = dict(
        Pre_Error_A=Pre_Error_A, Pre_Error_B=Pre_Error_B,
        Back_Error_A=Back_Error_A, Back_Error_B=Back_Error_B,
        ICI_A=ICI_A, ICI_B=ICI_B,
        IDD_Pre=IDD_Pre, IDD_Back=IDD_Back,
        CAR_A=CAR_A, CAR_B=CAR_B
    )
    return metrics

def compress_identity_pca(df,Time='Time'):
    """
    用分块 PCA 把 14 维核心特征压缩为 3 个主成分，并把 Time 列一并返回。
    """
 
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
    
def regularize_df(df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """
    method: 'zscore'(默认) | 'minmax' | 'robust'
    """
    scalers = {
        'zscore': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scalers[method]
    return pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

'''*********************************************************'''
###########################################整理
'''*********************************************************'''
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


eventN=60
Param=['t', 'morph_PC1', 'motion_PC1', 'orient_PC1']
eventName=events[['Time', 'TrackID_A', 'TrackID_B']].astype(str).agg('-'.join, axis=1)
idenscore=pd.DataFrame(index=eventName,columns=list(Param))
idenbackscore=pd.DataFrame(index=eventName,columns=list(Param))
for eventN in range(len(eventName)):
    df_reduced=compress_identity_pca(df)
    touchcellpoint, othertouchcellpoint, touchcellT,prepost_flag = extract_touch_pair_timeseries(
        df_reduced, events.iloc[eventN,:], time_col='Time',n=5)
    Param=touchcellpoint.columns
    if np.any(prepost_flag == 0):
        continue 
    event_dir=os.path.join(save_dir,eventName[eventN] )
    os.makedirs(event_dir, exist_ok=True) 
    # 按't'列从小到大排序
    touchcellpoint = touchcellpoint.sort_values(by='t')
    othertouchcellpoint = othertouchcellpoint.sort_values(by='t')
    #进行正则化来强化特征
    # 截取t列值小于或大于touchcellT的行
    touchcellpoint_pre = touchcellpoint[touchcellpoint['t'] <= int(touchcellT)]
    othertouchcellpoint_pre = othertouchcellpoint[othertouchcellpoint['t'] <= int(touchcellT)]
    touchcellpoint_back = touchcellpoint[touchcellpoint['t'] > int(touchcellT)]
    othertouchcellpoint_back = othertouchcellpoint[othertouchcellpoint['t'] > int(touchcellT)]
    ###1,4,5,8
    ###2,3,6,
    for par in Param:
        parN= touchcellpoint.columns.get_loc(par) 
        #parN=1###################选择参数
        data=touchcellpoint.iloc[:,parN]
        otherdata=othertouchcellpoint.iloc[:,parN]
        IMFdf=EMD_Right(touchcellpoint_pre,parN)
        otherIMFdf=EMD_Right(othertouchcellpoint_pre,parN)
        if IMFdf.shape[1]==0 or otherIMFdf.shape[1]==0:
            continue
        predicted_df=LSTM_right(IMFdf,3,touchcellpoint_back.shape[0])
        otherpredicted_df=LSTM_right(otherIMFdf,3,othertouchcellpoint_back.shape[0])
        predicted_reconstructed_data=EMD_recue(predicted_df)
        predicted_other_reconstructed_data=EMD_recue(otherpredicted_df)
        #predicted_reconstructed_data,predicted_other_reconstructed_data；
        # 截取前3行或元素
        A = truncate_data(touchcellpoint_back, 5)
        B = truncate_data(othertouchcellpoint_back, 5)
        a = truncate_data(predicted_reconstructed_data, 5)
        b = truncate_data(predicted_other_reconstructed_data, 5)
        #'mse', 'mae', 'r2', 'pearson', 'spearman'
        results_value = evaluate_and_visualize(A.iloc[:,parN], 
        B.iloc[:,parN], 
        pd.DataFrame(a), 
        pd.DataFrame(b), metric_name='mae',Aname=A.index[0],Bname=B.index[0],
        save_path=os.path.join(event_dir, par+'参数预测混淆矩阵.pdf'))#save_path
        results_value.to_csv(os.path.join(event_dir, par+'参数预测混淆矩阵.csv'))
        result = analyze_2x2_dataframe_Min(results_value)#analyze_2x2_dataframe_Max
        idenscore.loc[eventName[eventN],par] = result

        touchcellpoint_back_re=touchcellpoint_back.iloc[::-1].reset_index(drop=True)
        othertouchcellpoint_back_re=othertouchcellpoint_back.iloc[::-1].reset_index(drop=True)
        IMFdf_back=EMD_Right(touchcellpoint_back,parN)
        otherIMFdf_back=EMD_Right(othertouchcellpoint_back,parN)
        predicted_df_back=LSTM_right(IMFdf_back,3,touchcellpoint_pre.shape[0])
        otherpredicted_df_back=LSTM_right(otherIMFdf_back,3,othertouchcellpoint_pre.shape[0])
        predicted_reconstructed_data_back=EMD_recue(predicted_df_back)
        predicted_other_reconstructed_data_back=EMD_recue(otherpredicted_df_back)
        predicted_reconstructed_data_back=predicted_reconstructed_data_back[::-1]
        predicted_other_reconstructed_data_back=predicted_other_reconstructed_data_back[::-1]
        #predicted_reconstructed_data_back，predicted_other_reconstructed_data_back
        # 截取前3行或元素
        A = truncate_data(touchcellpoint_pre, 5,False)
        B = truncate_data(othertouchcellpoint_pre, 5,False)
        a = truncate_data(predicted_reconstructed_data_back, 5,False)
        b = truncate_data(predicted_other_reconstructed_data_back, 5,False)
        #'mse', 'mae', 'r2', 'pearson', 'spearman'
        results_value_back = evaluate_and_visualize(A.iloc[:,parN], 
        B.iloc[:,parN], 
        pd.DataFrame(a), 
        pd.DataFrame(b), metric_name='mae',Aname=A.index[0],Bname=B.index[0],
        save_path=os.path.join(event_dir, par+'参数反向预测混淆矩阵.pdf'))#save_path
        results_value_back.to_csv(os.path.join(event_dir, par+'参数反向预测混淆矩阵.csv'))
        #'mse', 'mae', 用Min
        # 'r2', 'pearson', 'spearman'，用Max
        result_back = analyze_2x2_dataframe_Min(results_value_back)#analyze_2x2_dataframe_Max
        idenbackscore.loc[eventName[eventN],par] = result_back

        Aback=truncate_data(touchcellpoint_back, 5).iloc[:,parN]
        Bback= truncate_data(othertouchcellpoint_back, 5).iloc[:,parN]
        AbackValue=truncate_data(predicted_reconstructed_data, 5)
        BbackValue= truncate_data(predicted_other_reconstructed_data, 5)
        Apre=truncate_data(touchcellpoint_pre, 5,False).iloc[:,parN]
        Bpre=truncate_data(othertouchcellpoint_pre, 5,False).iloc[:,parN]
        ApreValue=truncate_data(predicted_reconstructed_data_back, 5,False)
        BpreValue=truncate_data(predicted_other_reconstructed_data_back, 5,False)
        Aback=np.array(Aback)
        Bback=np.array(Bback)
        Apre=np.array(Apre)
        Bpre=np.array(Bpre)
        cell_identity_report(Apre, ApreValue, Aback, AbackValue,
                         Bpre, BpreValue, Bback, BbackValue)

        with open( os.path.join(event_dir, 'cell_report.txt'), "a", encoding="utf-8") as f:##a为追加，w为覆盖
            with contextlib.redirect_stdout(f):
                print('************细胞碰撞分开后预测结果：'+par+'***************')
                cell_identity_report(Apre, ApreValue, Aback, AbackValue,
                                 Bpre, BpreValue, Bback, BbackValue)

    merged = pd.concat([idenscore.iloc[[eventN], :], idenbackscore.iloc[[eventN], :]], axis=0).T
    # 如果想给列起名字
    merged.columns = ['idenscore_60', 'idenbackscore_60']
    merged.to_csv(os.path.join(event_dir, '身份识别结果.csv'))

idenbackscore.to_csv(os.path.join(save_dir, '身份识别结果back.csv'))
Uback= idenbackscore.dropna()
U= idenscore.dropna()



'''***********************************************'''
##################################对于时刻部分不足以进行预判的碰撞事件

def generate_partial_prediction_report(prediction_results, available_windows, par_name):
    """
    生成部分预测的评估报告
    
    参数:
    prediction_results -- 包含四个方向预测结果的字典
    available_windows -- 可用时间窗口的字典
    par_name -- 参数名称
    
    返回:
    report_text -- 报告文本
    """
    report_lines = []
    report_lines.append(f"=== 参数 {par_name} 部分预测报告 ===")
    
    # 统计可用的预测方向
    total_directions = 4
    available_directions = sum(available_windows.values())
    report_lines.append(f"总预测方向: {total_directions}")
    report_lines.append(f"可用预测方向: {available_directions}")
    report_lines.append(f"预测覆盖率: {available_directions/total_directions*100:.1f}%")
    
    # 详细分析每个方向
    direction_names = {
        'A_pre': 'A细胞接触前预测',
        'A_post': 'A细胞接触后预测', 
        'B_pre': 'B细胞接触前预测',
        'B_post': 'B细胞接触后预测'
    }
    
    for direction, name in direction_names.items():
        status = "✓ 可用" if available_windows[direction] else "✗ 不可用"
        prediction_status = "✓ 成功" if prediction_results[direction] is not None else "✗ 失败"
        report_lines.append(f"{name}: {status} | 预测状态: {prediction_status}")
    
    # 建议
    if available_directions >= 2:
        report_lines.append("\n建议: 可以进行部分评估，重点关注可用的预测方向")
    else:
        report_lines.append("\n建议: 预测数据不足，建议增加时间窗口或调整参数")
    
    return "\n".join(report_lines)


def create_flexible_evaluation_matrix(prediction_results, available_windows, par_name):
    """
    创建灵活的评估矩阵，即使某些预测结果缺失
    
    参数:
    prediction_results -- 包含四个方向预测结果的字典
    available_windows -- 可用时间窗口的字典
    par_name -- 参数名称
    
    返回:
    evaluation_matrix -- 评估矩阵DataFrame
    """
    # 创建2x2矩阵，标记可用的预测方向
    matrix_data = []
    row_labels = ['A细胞', 'B细胞']
    col_labels = ['接触前', '接触后']
    
    for i, row_label in enumerate(row_labels):
        row_data = []
        for j, col_label in enumerate(col_labels):
            if i == 0:  # A细胞
                if j == 0:  # 接触前
                    direction = 'A_pre'
                else:  # 接触后
                    direction = 'A_post'
            else:  # B细胞
                if j == 0:  # 接触前
                    direction = 'B_pre'
                else:  # 接触后
                    direction = 'B_post'
            
            if available_windows[direction] and prediction_results[direction] is not None:
                row_data.append("✓ 可用")
            elif available_windows[direction]:
                row_data.append("⚠ 窗口可用但预测失败")
            else:
                row_data.append("✗ 不可用")
        
        matrix_data.append(row_data)
    
    evaluation_matrix = pd.DataFrame(matrix_data, index=row_labels, columns=col_labels)
    
    # 添加统计信息
    total_cells = 2
    total_timepoints = 2
    available_predictions = sum(1 for result in prediction_results.values() if result is not None)
    total_possible = sum(available_windows.values())
    
    stats = {
        '总细胞数': total_cells,
        '总时间点': total_timepoints,
        '可用预测数': available_predictions,
        '可用时间窗口数': total_possible,
        '预测成功率': f"{available_predictions/total_possible*100:.1f}%" if total_possible > 0 else "N/A"
    }
    
    return evaluation_matrix, stats


def generate_event_prediction_summary(event_name, all_prediction_results, all_available_windows, all_params):
    """
    生成整个事件的预测质量总结
    
    参数:
    event_name -- 事件名称
    all_prediction_results -- 所有参数的预测结果字典
    all_available_windows -- 所有参数的可用时间窗口字典
    all_params -- 所有参数列表
    
    返回:
    summary_text -- 总结文本
    """
    summary_lines = []
    summary_lines.append(f"=== 事件 {event_name} 预测质量总结 ===")
    
    # 统计总体情况
    total_params = len(all_params)
    total_directions = 4  # 每个参数4个方向
    
    # 计算每个参数的预测成功率
    param_success_rates = {}
    for param in all_params:
        if param in all_prediction_results:
            available_count = sum(all_available_windows[param].values())
            successful_count = sum(1 for result in all_prediction_results[param].values() if result is not None)
            success_rate = successful_count / available_count * 100 if available_count > 0 else 0
            param_success_rates[param] = success_rate
    
    # 总体统计
    total_available = sum(sum(windows.values()) for windows in all_available_windows.values())
    total_successful = sum(sum(1 for result in results.values() if result is not None) 
                          for results in all_prediction_results.values())
    overall_success_rate = total_successful / total_available * 100 if total_available > 0 else 0
    
    summary_lines.append(f"总参数数: {total_params}")
    summary_lines.append(f"总预测方向数: {total_params * total_directions}")
    summary_lines.append(f"可用时间窗口总数: {total_available}")
    summary_lines.append(f"成功预测总数: {total_successful}")
    summary_lines.append(f"总体预测成功率: {overall_success_rate:.1f}%")
    
    # 参数级别统计
    summary_lines.append(f"\n各参数预测成功率:")
    for param, rate in param_success_rates.items():
        summary_lines.append(f"  {param}: {rate:.1f}%")
    
    # 质量评估
    if overall_success_rate >= 80:
        quality = "优秀"
    elif overall_success_rate >= 60:
        quality = "良好"
    elif overall_success_rate >= 40:
        quality = "一般"
    else:
        quality = "需要改进"
    
    summary_lines.append(f"\n预测质量评估: {quality}")
    
    # 建议
    if overall_success_rate < 50:
        summary_lines.append("建议: 预测成功率较低，建议检查数据质量和模型参数")
    elif overall_success_rate < 80:
        summary_lines.append("建议: 预测成功率中等，可以考虑优化模型或增加训练数据")
    else:
        summary_lines.append("建议: 预测成功率较高，当前设置较为合适")
    
    return "\n".join(summary_lines)


def clean_and_validate_result_matrices(idenscore, idenbackscore, prediction_status):
    """
    清理和验证结果矩阵，确保数据一致性
    
    参数:
    idenscore -- 前向预测结果矩阵
    idenbackscore -- 后向预测结果矩阵
    prediction_status -- 预测状态矩阵
    
    返回:
    cleaned_idenscore, cleaned_idenbackscore -- 清理后的结果矩阵
    """
    print("=== 清理和验证结果矩阵 ===")
    
    # 创建清理后的矩阵副本
    cleaned_idenscore = idenscore.copy()
    cleaned_idenbackscore = idenbackscore.copy()
    
    # 检查预测状态，将无效的预测结果标记为NaN
    for event_idx in prediction_status.index:
        for param in prediction_status.columns:
            status = prediction_status.loc[event_idx, param]
            
            # 如果预测失败或不可用，将对应的结果设为NaN
            if '失败' in str(status) or '不可用' in str(status) or '时间窗口不足' in str(status):
                if event_idx in cleaned_idenscore.index and param in cleaned_idenscore.columns:
                    cleaned_idenscore.loc[event_idx, param] = np.nan
                if event_idx in cleaned_idenbackscore.index and param in cleaned_idenbackscore.columns:
                    cleaned_idenbackscore.loc[event_idx, param] = np.nan
    
    # 统计清理结果
    original_entries = len(idenscore) * len(idenscore.columns)
    cleaned_entries = len(cleaned_idenscore.dropna()) * len(cleaned_idenscore.columns)
    
    print(f"原始结果条目数: {original_entries}")
    print(f"清理后有效条目数: {cleaned_entries}")
    print(f"清理条目数: {original_entries - cleaned_entries}")
    
    # 检查数据一致性
    print("\n数据一致性检查:")
    print(f"前向预测矩阵形状: {cleaned_idenscore.shape}")
    print(f"后向预测矩阵形状: {cleaned_idenbackscore.shape}")
    print(f"预测状态矩阵形状: {prediction_status.shape}")
    
    # 验证索引一致性
    if (cleaned_idenscore.index.equals(cleaned_idenbackscore.index) and 
        cleaned_idenscore.index.equals(prediction_status.index)):
        print("✓ 索引一致性检查通过")
    else:
        print("⚠ 索引一致性检查失败")
    
    # 验证列一致性
    if (cleaned_idenscore.columns.equals(cleaned_idenbackscore.columns) and 
        cleaned_idenscore.columns.equals(prediction_status.columns)):
        print("✓ 列一致性检查通过")
    else:
        print("⚠ 列一致性检查失败")
    
    return cleaned_idenscore, cleaned_idenbackscore



 '''***********************************************'''
#################################################################运行
df_flag_log = pd.DataFrame(
    columns=['event', 'num_zeros_in_flag', 'was_skipped']
)
prediction_status = pd.DataFrame(index=eventName, columns=Param)
for eventN in range(len(eventName)):
    df_reduced=compress_identity_pca(df)
    touchcellpoint, othertouchcellpoint, touchcellT,prepost_flag = extract_touch_pair_timeseries(
        df_reduced, events.iloc[eventN,:], time_col='Time',n=5)
    Param=touchcellpoint.columns
    
    # 检查哪些时间窗口可用，而不是直接跳过
    available_windows = {
        'A_pre': prepost_flag[0, 0] == 1,   # A细胞接触前
        'A_post': prepost_flag[0, 1] == 1,  # A细胞接触后  
        'B_pre': prepost_flag[1, 0] == 1,   # B细胞接触前
        'B_post': prepost_flag[1, 1] == 1   # B细胞接触后
    }
    # 计算 0 的个数
    num_zeros = (prepost_flag == 0).sum() 
    should_skip = not any(available_windows.values())
    # 先写日志
    df_flag_log.loc[len(df_flag_log)] = [
        eventName[eventN],
        num_zeros,
        should_skip
    ]
    # 如果至少有一个时间窗口可用，就继续处理
    if not any(available_windows.values()):
        print(f"事件 {eventName[eventN]} 所有时间窗口都不满足条件，跳过")
        # 标记该事件为跳过状态
        for par in Param:
            prediction_status.loc[eventName[eventN], par] = '时间窗口不足'
        continue
        
    print(f"事件 {eventName[eventN]} 可用时间窗口: {available_windows}")
    
    event_dir=os.path.join(save_dir,eventName[eventN] )
    os.makedirs(event_dir, exist_ok=True) 
    
    # 按't'列从小到大排序
    touchcellpoint = touchcellpoint.sort_values(by='t')
    othertouchcellpoint = othertouchcellpoint.sort_values(by='t')
    
    # 截取t列值小于或大于touchcellT的行
    touchcellpoint_pre = touchcellpoint[touchcellpoint['t'] <= int(touchcellT)]
    othertouchcellpoint_pre = othertouchcellpoint[othertouchcellpoint['t'] <= int(touchcellT)]
    touchcellpoint_back = touchcellpoint[touchcellpoint['t'] > int(touchcellT)]
    othertouchcellpoint_back = othertouchcellpoint[othertouchcellpoint['t'] > int(touchcellT)]
    
    for par in Param:
        parN= touchcellpoint.columns.get_loc(par) 
        
        # 初始化预测结果存储
        prediction_results = {
            'A_pre': None, 'A_post': None, 'B_pre': None, 'B_post': None
        }
        
        # 前向预测（预测碰撞后的行为）
        # 需要碰撞前的数据充足，预测碰撞后的行为
        if available_windows['A_pre'] and available_windows['B_pre']:
            try:
                IMFdf=EMD_Right(touchcellpoint_pre,parN)
                otherIMFdf=EMD_Right(othertouchcellpoint_pre,parN)
                if IMFdf.shape[1] > 0 and otherIMFdf.shape[1] > 0:
                    predicted_df=LSTM_right(IMFdf,3,touchcellpoint_back.shape[0])
                    otherpredicted_df=LSTM_right(otherIMFdf,3,othertouchcellpoint_back.shape[0])
                    predicted_reconstructed_data=EMD_recue(predicted_df)
                    predicted_other_reconstructed_data=EMD_recue(otherpredicted_df)
                    prediction_results['A_post'] = predicted_reconstructed_data
                    prediction_results['B_post'] = predicted_other_reconstructed_data
                    print(f"参数 {par} 前向预测成功：用碰撞前数据预测碰撞后行为")
                else:
                    print(f"参数 {par} 前向预测失败：EMD分解结果为空")
            except Exception as e:
                print(f"参数 {par} 前向预测出错: {e}")
        
        # 后向预测（预测碰撞前的行为）
        # 需要碰撞后的数据充足，预测碰撞前的行为
        if available_windows['A_post'] and available_windows['B_post']:
            try:
                IMFdf_back=EMD_Right(touchcellpoint_back,parN)
                otherIMFdf_back=EMD_Right(othertouchcellpoint_back,parN)
                if IMFdf_back.shape[1] > 0 and otherIMFdf_back.shape[1] > 0:
                    predicted_df_back=LSTM_right(IMFdf_back,3,touchcellpoint_pre.shape[0])
                    otherpredicted_df_back=LSTM_right(otherIMFdf_back,3,othertouchcellpoint_pre.shape[0])
                    predicted_reconstructed_data_back=EMD_recue(predicted_df_back)
                    predicted_other_reconstructed_data_back=EMD_recue(otherpredicted_df_back)
                    predicted_reconstructed_data_back=predicted_reconstructed_data_back[::-1]
                    predicted_other_reconstructed_data_back=predicted_other_reconstructed_data_back[::-1]
                    prediction_results['A_pre'] = predicted_reconstructed_data_back
                    prediction_results['B_pre'] = predicted_other_reconstructed_data_back
                    print(f"参数 {par} 后向预测成功：用碰撞后数据预测碰撞前行为")
                else:
                    print(f"参数 {par} 后向预测失败：EMD分解结果为空")
            except Exception as e:
                print(f"参数 {par} 后向预测出错: {e}")
        
        # 根据可用的预测结果进行评估
        if prediction_results['A_post'] is not None and prediction_results['B_post'] is not None:
            # 前向预测评估（预测碰撞后的行为）
            A = truncate_data(touchcellpoint_back, 5)
            B = truncate_data(othertouchcellpoint_back, 5)
            a = truncate_data(prediction_results['A_post'], 5)
            b = truncate_data(prediction_results['B_post'], 5)
            
            try:
                results_value = evaluate_and_visualize(A.iloc[:,parN], 
                    B.iloc[:,parN], 
                    pd.DataFrame(a), 
                    pd.DataFrame(b), metric_name='mae',Aname=A.index[0],Bname=B.index[0],
                    save_path=os.path.join(event_dir, par+'参数预测混淆矩阵.pdf'))
                results_value.to_csv(os.path.join(event_dir, par+'参数预测混淆矩阵.csv'))
                result = analyze_2x2_dataframe_Min(results_value)
                idenscore.loc[eventName[eventN],par] = result
                prediction_status.loc[eventName[eventN], par] = f'前向预测成功(结果:{result})'
                print(f"参数 {par} 前向预测评估完成，结果: {result}")
            except Exception as e:
                print(f"参数 {par} 前向预测评估失败: {e}")
                prediction_status.loc[eventName[eventN], par] = f'前向预测评估失败:{str(e)[:30]}'
        else:
            prediction_status.loc[eventName[eventN], par] = '前向预测不可用'
        
        if prediction_results['A_pre'] is not None and prediction_results['B_pre'] is not None:
            # 后向预测评估（预测碰撞前的行为）
            A = truncate_data(touchcellpoint_pre, 5,False)
            B = truncate_data(othertouchcellpoint_pre, 5,False)
            a = truncate_data(prediction_results['A_pre'], 5,False)
            b = truncate_data(prediction_results['B_pre'], 5,False)
            
            try:
                results_value_back = evaluate_and_visualize(A.iloc[:,parN], 
                    B.iloc[:,parN], 
                    pd.DataFrame(a), 
                    pd.DataFrame(b), metric_name='mae',Aname=A.index[0],Bname=B.index[0],
                    save_path=os.path.join(event_dir, par+'参数反向预测混淆矩阵.pdf'))
                results_value_back.to_csv(os.path.join(event_dir, par+'参数反向预测混淆矩阵.csv'))
                result_back = analyze_2x2_dataframe_Min(results_value_back)
                idenbackscore.loc[eventName[eventN],par] = result_back
                prediction_status.loc[eventName[eventN], par] += f' | 后向预测成功(结果:{result_back})'
                print(f"参数 {par} 后向预测评估完成，结果: {result_back}")
            except Exception as e:
                print(f"参数 {par} 后向预测评估失败: {e}")
                prediction_status.loc[eventName[eventN], par] += f' | 后向预测评估失败:{str(e)[:30]}'
        else:
            if prediction_status.loc[eventName[eventN], par] == '前向预测不可用':
                prediction_status.loc[eventName[eventN], par] = '双向预测都不可用'
            else:
                prediction_status.loc[eventName[eventN], par] += ' | 后向预测不可用'
        
        # 生成部分预测报告
        partial_report = generate_partial_prediction_report(prediction_results, available_windows, par)
        print(partial_report)
        
        # 创建灵活的评估矩阵
        evaluation_matrix, stats = create_flexible_evaluation_matrix(prediction_results, available_windows, par)
        print(f"\n参数 {par} 评估矩阵:")
        print(evaluation_matrix)
        print(f"\n统计信息:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 保存部分预测报告到文件
        with open(os.path.join(event_dir, 'partial_prediction_report.txt'), "a", encoding="utf-8") as f:
            f.write(partial_report + "\n\n")
            f.write(f"评估矩阵:\n{evaluation_matrix.to_string()}\n\n")
            f.write("统计信息:\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "="*50 + "\n\n")
        
        # 细胞身份报告（只在两个方向都有预测结果时生成）
        if (prediction_results['A_pre'] is not None and prediction_results['B_pre'] is not None and
            prediction_results['A_post'] is not None and prediction_results['B_post'] is not None):
            
            Aback=truncate_data(touchcellpoint_back, 5).iloc[:,parN]
            Bback= truncate_data(othertouchcellpoint_back, 5).iloc[:,parN]
            AbackValue=truncate_data(prediction_results['A_post'], 5)
            BbackValue= truncate_data(prediction_results['B_post'], 5)
            Apre=truncate_data(touchcellpoint_pre, 5,False).iloc[:,parN]
            Bpre=truncate_data(othertouchcellpoint_pre, 5,False).iloc[:,parN]
            ApreValue=truncate_data(prediction_results['A_pre'], 5,False)
            BpreValue=truncate_data(prediction_results['B_pre'], 5,False)
            
            Aback=np.array(Aback)
            Bback=np.array(Bback)
            Apre=np.array(Apre)
            Bpre=np.array(Bpre)
            
            try:
                cell_identity_report(Apre, ApreValue, Aback, AbackValue,
                                 Bpre, BpreValue, Bback, BbackValue)
                
                with open( os.path.join(event_dir, 'cell_report.txt'), "a", encoding="utf-8") as f:
                    with contextlib.redirect_stdout(f):
                        print('************细胞碰撞分开后预测结果：'+par+'***************')
                        cell_identity_report(Apre, ApreValue, Aback, AbackValue,
                                         Bpre, BpreValue, Bback, BbackValue)
            except Exception as e:
                print(f"参数 {par} 细胞身份报告生成失败: {e}")
        else:
            print(f"参数 {par} 缺少完整的预测结果，跳过细胞身份报告")

    # 生成事件预测总结
    event_summary = generate_event_prediction_summary(
        eventName[eventN], 
        {par: prediction_results for par in Param}, 
        {par: available_windows for par in Param}, 
        Param
    )
    print(f"\n{event_summary}")
    
    # 保存事件总结到文件
    with open(os.path.join(event_dir, 'event_prediction_summary.txt'), "w", encoding="utf-8") as f:
        f.write(event_summary)
    
    merged = pd.concat([idenscore.iloc[[eventN], :], idenbackscore.iloc[[eventN], :]], axis=0).T
    # 如果想给列起名字
    merged.columns = ['idenscore_60', 'idenbackscore_60']
    merged.to_csv(os.path.join(event_dir, '身份识别结果.csv'))
    # 在循环末尾加
    locals_to_del = [
        'touchcellpoint', 'othertouchcellpoint',
        'touchcellpoint_pre', 'othertouchcellpoint_pre',
        'touchcellpoint_back', 'othertouchcellpoint_back',
        'IMFdf', 'otherIMFdf', 'IMFdf_back', 'otherIMFdf_back',
        'predicted_df', 'otherpredicted_df',
        'predicted_df_back', 'otherpredicted_df_back',
        'predicted_reconstructed_data',
        'predicted_other_reconstructed_data',
        'predicted_reconstructed_data_back',
        'predicted_other_reconstructed_data_back',
        'A', 'B', 'a', 'b',
        'results_value', 'results_value_back',
        'result', 'result_back',
        'partial_report', 'evaluation_matrix'
    ]
    for name in locals_to_del:
        if name in locals():
            del locals()[name]
    gc.collect()                       # Python 垃圾回收
    plt.close('all')                   # 关闭所有 Matplotlib 图
    





##########################################统计各数据集合的各事件预测结果
import os
import pandas as pd
from pathlib import Path

######统计各数据集合的各事件预测结果-区分是否不包含缺失（即是否完全可以预测）
def sign_stats(df):
    """返回 DataFrame 的列级及 row2-4 行和的正负零计数"""
    # 列级
    col_stats = df.iloc[:, :-1].apply(lambda s: pd.Series({
        'positive': (s > 0).sum(),
        'negative': (s < 0).sum(),
        'zero':     (s == 0).sum()
    })).T

    # 第2,3,4列的行和（索引1,2,3）
    row_sum = df.iloc[:, [1, 2, 3]].sum(axis=1)
    row_stats = pd.Series({
        'positive': (row_sum > 0).sum(),
        'negative': (row_sum < 0).sum(),
        'zero':     (row_sum == 0).sum()
    })

    return col_stats, row_stats

def statistic_result(result_root,df,eventName):
    # 只读一个 csv 拿行列名（所有文件行列一致）
    sample_csv = next(Path(result_root).rglob('身份识别结果.csv'))
    row_names = pd.read_csv(sample_csv, index_col=0).index.tolist()[:4]
    col_names = pd.read_csv(sample_csv, index_col=0).columns.tolist()[:2]

    records = []
    for sub in Path(result_root).iterdir():
        if not sub.is_dir():
            continue
        csv_path = next(sub.glob('身份识别结果.csv'), None)
        if csv_path is None:
            continue

        df0 = pd.read_csv(csv_path, index_col=0)  # 4×2
        row_sums = df0.sum(axis=1).tolist()
        col_sums = df0.sum(axis=0).tolist()

        # 行列名直接做列名
        record = dict(zip(row_names + col_names, row_sums + col_sums))
        record['subfolder'] = sub.name
        records.append(record)

    summary_df = pd.DataFrame(records).set_index('subfolder')

    # 第 2、3、4 列的行和（即第 1、2、3 列索引对应的列）
    # 2. 第2,3,4列的行和
    rows_2_3_4_all = summary_df.iloc[:, [1, 2, 3]].sum(axis=1)
    # 3. 第2,3,4列的行和的正数/负数/0 计数
    rows_2_3_4_stats_all  = pd.Series({
        'pos': (rows_2_3_4_all > 0).sum(),
        'neg': (rows_2_3_4_all < 0).sum(),
        'zero': (rows_2_3_4_all == 0).sum()
    })
    #rows_2_3_4_stats_all 

    ########确定存在缺失和不存在缺失的事件数量
    flag_df = (
        pd.Series(index=eventName, data=0)
        .where(
            [np.any(extract_touch_pair_timeseries(
                compress_identity_pca(df),
                events.iloc[i],
                time_col='Time',
                n=5
            )[3] == 0) for i in range(len(eventName))],
            1
        )
    )
    sum(flag_df)
    records = []
    for sub in Path(result_root).iterdir():
        if not sub.is_dir():
            continue
        csv_path = next(sub.glob('身份识别结果.csv'), None)
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path, index_col=0)  # 4×2
        row_sums = df.sum(axis=1).tolist()
        col_sums = df.sum(axis=0).tolist()

        # 行列名直接做列名
        record = dict(zip(row_names + col_names, row_sums + col_sums))
        record['subfolder'] = sub.name
        record['has_na']    = int(df.isnull().any().any())  # 只要有一个缺失值就标记 1
        records.append(record)

    summary_df = pd.DataFrame(records).set_index('subfolder')
    merged = summary_df.join( pd.DataFrame(flag_df, columns=['Flag']), how='outer')
    merged = merged.drop(columns=['t'])
    # 把含缺失 / 不含缺失的分开
    mask_na      = merged['Flag'] == 0
    df_no_na     = merged[~mask_na]
    df_with_na   = merged[mask_na]
    # 分别计算
    col_no_na, row_no_na   = sign_stats(df_no_na)
    col_with_na, row_with_na = sign_stats(df_with_na)
    # 结果
    #col_no_na, row_no_na, col_with_na, row_with_na
    colrowstatistic={'col_no_na':col_no_na,'row_no_na':row_no_na,
                     'col_with_na':col_with_na,'row_with_na':row_with_na}

    rows_2_3_4 = df_no_na.iloc[:, [0, 1, 2]].sum(axis=1)
    # 3. 第2,3,4列的行和的正数/负数/0 计数
    rows_2_3_4_stats = pd.Series({
        'pos': (rows_2_3_4 > 0).sum(),
        'neg': (rows_2_3_4 < 0).sum(),
        'zero': (rows_2_3_4 == 0).sum()
    })
    #rows_2_3_4_stats
    return rows_2_3_4_stats_all,rows_2_3_4_stats,colrowstatistic,merged


N = 0
root = os.path.join(workpy, '碰撞检测与预测', csv_files[N])
result_root = os.path.join(root, 'result')

filename  = csv_files[N]          # 取文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy4, filename)
#os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
events_path = os.path.join(save_dir, 'unique_events.csv')
events=pd.read_csv(events_path, index_col=None)
Param=['t', 'morph_PC1', 'motion_PC1', 'orient_PC1']
eventName=events[['Time', 'TrackID_A', 'TrackID_B']].astype(str).agg('-'.join, axis=1)
statistic_result(result_root,df,eventName)


















