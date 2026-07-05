import ast
from functools import partial
from functools import reduce
from pathlib import Path
'''
def attach_close_after_ids(
        Ubefore, Uafter,
        x='Position X', y='Position Y', z='Position Z',
        id_col='ID', t_col='t',
        speed_col='Average_Speed',
        disp_col='Average_displacement',
        out_col='Matched_After_IDs',
        target_n=20):
    """
    改进点：
    1. 先做位移+速度双阈值上限，避免巨量候选；
    2. 每一步都按真实距离升序取前 target_n，而非简单截断；
    3. 不足时再逐级放宽速度系数，直到够数或遍历完。
    """
    pos_b = Ubefore[[x, y, z]].astype(float).values
    pos_a = Uafter[[x, y, z]].astype(float).values
    t_b   = Ubefore[t_col].astype(float).values
    t_a   = Uafter[t_col].astype(float).values
    id_a  = Uafter[id_col].values

    disp_b = Ubefore[disp_col].astype(float).values
    v_b    = Ubefore[speed_col].astype(float).values

    matched = []
    for i, (pos, d, v, t_cur) in enumerate(zip(pos_b, disp_b, v_b, t_b)):
        delta_t = t_a - t_cur
        dist    = np.linalg.norm(pos_a - pos, axis=1)   # 一次算完

        # ---------- 阶段 1：保守上限 ----------
        r_max = np.maximum(3.0 * d * delta_t, 30 * delta_t)  # 可再调严
        mask  = dist <= r_max
        idx   = np.where(mask)[0]          # 候选下标
        if idx.size == 0:                  # 一个都没有
            matched.append([])
            continue

        # 按距离升序，只保留前 target_n 个
        ord_idx = idx[np.argsort(dist[idx])][:target_n]
        cand    = id_a[ord_idx]

        # ---------- 阶段 2：若不够，再逐级放宽 ----------
        if cand.size < target_n:
            for factor in np.arange(1.5, 10.1, 0.5):
                r_max = np.maximum(factor * v * delta_t, 15 * delta_t)
                mask  = dist <= r_max
                idx   = np.where(mask)[0]
                ord_idx = idx[np.argsort(dist[idx])][:target_n]
                cand = id_a[ord_idx]
                if cand.size >= target_n:
                    break

        matched.append(cand.tolist())

    Ubefore[out_col] = matched
    return Ubefore

def attach_close_after_ids(
        Ubefore, Uafter,
        x='Position X', y='Position Y', z='Position Z',
        id_col='ID', t_col='t',
        speed_col='Average_Speed',
        disp_col='Average_displacement',
        out_col='Matched_After_IDs',
        target_n=20,
        weight_dist=1.0,
        weight_time=0.5):
    """
    优先用 Average_displacement 做保守上限；
    在候选中按“距离 + 时间差”联合得分升序取前 target_n；
    不足时逐级放宽速度半径，直到够数或遍历完。
    """
    pos_b = Ubefore[[x, y, z]].astype(float).values
    pos_a = Uafter[[x, y, z]].astype(float).values
    t_b   = Ubefore[t_col].astype(float).values
    t_a   = Uafter[t_col].astype(float).values
    id_a  = Uafter[id_col].values

    disp_b = Ubefore[disp_col].astype(float).values
    v_b    = Ubefore[speed_col].astype(float).values

    matched = []
    for i, (pos, d, v, t_cur) in enumerate(zip(pos_b, disp_b, v_b, t_b)):
        delta_t = t_a - t_cur
        dist    = np.linalg.norm(pos_a - pos, axis=1)

        # ---------- 阶段 1：保守上限 ----------
        r_max = np.maximum(3.0 * d * delta_t, 30 * delta_t)
        idx   = np.where(dist <= r_max)[0]
        if idx.size == 0:
            matched.append([])
            continue

        # 联合得分排序
        norm_dist = dist[idx] / (r_max[idx] + 1e-6)
        norm_dt   = delta_t[idx] / (delta_t[idx].max() + 1e-6)
        score     = weight_dist * norm_dist + weight_time * norm_dt
        ord_idx   = idx[np.argsort(score)][:target_n]
        cand      = id_a[ord_idx]

        # ---------- 阶段 2：逐级放宽 ----------
        if cand.size < target_n:
            for factor in np.arange(1.5, 10.1, 0.5):
                r_max = np.maximum(factor * v * delta_t, 15 * delta_t)
                idx   = np.where(dist <= r_max)[0]
                if idx.size == 0:
                    continue
                norm_dist = dist[idx] / (r_max[idx] + 1e-6)
                norm_dt   = delta_t[idx] / (delta_t[idx].max() + 1e-6)
                score     = weight_dist * norm_dist + weight_time * norm_dt
                ord_idx   = idx[np.argsort(score)][:target_n]
                cand      = id_a[ord_idx]
                if cand.size >= target_n:
                    break

        matched.append(cand.tolist())

    Ubefore[out_col] = matched
    return Ubefore
'''
def attach_close_after_ids(
        Ubefore, Uafter,
        x='Position X', y='Position Y', z='Position Z',
        id_col='ID', t_col='t',
        speed_col='Average_Speed',
        disp_col='Average_displacement',
        out_col='Matched_After_IDs',
        target_n=20,
        weight_dist=1.0,
        weight_time=0.5,
        v_min=10.0):               # 速度圈下限
    """
    基础圈：1.5 * d * delta_t
    放大终点：1.5 * max(v, v_min) * delta_t
    每档联合得分取前 target_n，达到数量或终点即停。
    """
    pos_b = Ubefore[[x, y, z]].astype(float).values
    pos_a = Uafter[[x, y, z]].astype(float).values
    t_b   = Ubefore[t_col].astype(float).values
    t_a   = Uafter[t_col].astype(float).values
    id_a  = Uafter[id_col].values

    disp_b = Ubefore[disp_col].astype(float).values
    v_b    = Ubefore[speed_col].astype(float).values

    matched = []
    for i, (pos, d, v, t_cur) in enumerate(zip(pos_b, disp_b, v_b, t_b)):
        delta_t = t_a - t_cur
        dist    = np.linalg.norm(pos_a - pos, axis=1)

        # ---------- 阶段 1：基础圈 ----------
        base_radius = 1.5 * d * delta_t
        idx   = np.where(dist <= base_radius)[0]
        cand  = (id_a[idx[np.argsort(dist[idx])][:target_n]]
                 if idx.size else np.array([], dtype=id_a.dtype))

        # ---------- 阶段 2：放大到速度圈下限 ----------
        if cand.size < target_n:
            v_eff = np.maximum(v, v_min)          # 速度圈下限
            max_factor = v_eff / (d + 1e-6)       # 终点=1.5*v_eff*Δt
            factors = np.linspace(1.0, max_factor, 20)
            for f in factors:
                r = f * base_radius               # 始终放大基础圈
                idx = np.where(dist <= r)[0]
                if idx.size == 0:
                    continue
                norm_dist = dist[idx] / (r + 1e-6)
                norm_dt   = delta_t[idx] / (delta_t[idx].max() + 1e-6)
                score = weight_dist * norm_dist + weight_time * norm_dt
                cand = id_a[idx[np.argsort(score)][:target_n]]
                if cand.size >= target_n:
                    break
                if f >= max_factor:               # 已到达速度圈仍不足
                    break

        matched.append(cand.tolist())

    Ubefore[out_col] = matched
    return Ubefore
'''
def slice_temporal(dfA, dfFull,
                   id_col='ID', time_col='Time',
                   full_id='ID', full_time='Time',
                   direction='before'):
    """
    direction: 'before' 取 ≤ 时刻；'after' 取 ≥ 时刻
    返回：dfFull 中满足 ID 与时刻条件的子集
    """
    cut_dict = dict(zip(dfA[id_col], dfA[time_col]))

    if direction == 'before':
        mask = dfFull[full_time] <= dfFull[full_id].map(cut_dict)
    elif direction == 'after':
        mask = dfFull[full_time] >= dfFull[full_id].map(cut_dict)
    else:
        raise ValueError("direction must be 'before' or 'after'")

    return dfFull[dfFull[full_id].isin(cut_dict.keys()) & mask].copy()
'''
def slice_temporal(dfA, dfFull,
                   id_col='ID', time_col='Time',
                   full_id='ID', full_time='Time',
                   direction='before',
                   perturb=None,
                   tmin=None,             # 全局绝对上下限（可选）
                   tmax=None,
                   random_seed=42):
    """
    对 after 方向：扰动后时间再与每个 ID 自身时间范围取交集
    """
    if perturb is not None and direction == 'after':
        rng = np.random.default_rng(random_seed)
        noise = rng.integers(-int(perturb), int(perturb) + 1, size=len(dfA))
        dfA = dfA.copy()
        dfA[time_col] = dfA[time_col] + noise

        # ****** 新增：每个 ID 自己的时间范围 ******
        id_range = dfFull.groupby(full_id)[full_time].agg(['min', 'max'])
        dfA = dfA.merge(id_range, left_on=id_col, right_index=True, how='left')
        # 先与全局上下限交集，再与 ID 自身范围交集
        dfA[time_col] = dfA[time_col].clip(lower=dfA['min'], upper=dfA['max'])
        if tmin is not None or tmax is not None:
            dfA[time_col] = dfA[time_col].clip(lower=tmin, upper=tmax)
        dfA = dfA.drop(columns=['min', 'max'])

    cut_dict = dict(zip(dfA[id_col], dfA[time_col]))

    if direction == 'before':
        mask = dfFull[full_time] <= dfFull[full_id].map(cut_dict)
    elif direction == 'after':
        mask = dfFull[full_time] >= dfFull[full_id].map(cut_dict)
    else:
        raise ValueError("direction must be 'before' or 'after'")

    return dfFull[dfFull[full_id].isin(cut_dict.keys()) & mask].copy()

# ---------- 2. 两个纯函数（显式参数） ----------
def calc_dist_row(row, df_cache):
    df = df_cache.get(row['files'])
    if df is None:
        return [np.nan] * len(row['Matched_After_IDs'])

    cand_ids   = ast.literal_eval(row['Matched_After_IDs'])
    cand_times = ast.literal_eval(row['Matched_After_Times'])

    try:
        self_xyz = df.loc[(row['ID'], row['time']-1),
                          ['Position X','Position Y','Position Z']].values
    except KeyError:
        return [np.nan] * len(cand_ids)

    dist_list = []
    for cid, ct in zip(cand_ids, cand_times):
        try:
            xyz = df.loc[(cid, ct),
                         ['Position X','Position Y','Position Z']].values
            dist_list.append(np.linalg.norm(xyz - self_xyz))
        except KeyError:
            dist_list.append(np.nan)
    return dist_list
def self_dist_row(row, df_cache):
    df = df_cache.get(row['files'])
    if df is None:
        return np.nan
    id_, t1, t2 = row['ID'], row['time']-1, row['ID_Time']
    try:
        xyz1 = df.loc[(id_, t1), ['Position X','Position Y','Position Z']].values
        xyz2 = df.loc[(id_, t2), ['Position X','Position Y','Position Z']].values
        return float(np.linalg.norm(xyz2 - xyz1))
    except KeyError:
        return np.nan

def add_lenmatched(total_df, matched_dirs, workpy6='workpy6'):
    """
    就地新增列 lenmatched_row
    matched_dirs: 路径集合（Path 对象）
    workpy6: 顶级根目录，用于拼接最终绝对路径
    """
    workpy6 = Path(workpy6)

    # 先给 matched_dirs 建立快速映射：
    # key = (test, files, time)   value = csv 目录（matched_dirs.parent.parent）
    dir_map = {}
    for mdir in matched_dirs:
        csv_dir = mdir.parent.parent               # t={t}matched_before.csv 所在目录
        test    = csv_dir.name              # test 类型（固定）

        # 向上搜两级（跳过 test 本身）
        files = None
        for p in [csv_dir.parent.parent, csv_dir.parent]:
            if '.csv' in p.name:
                files = p.name
                break

        key = (files,test, int(mdir.parent.name.split('=')[-1]))
        dir_map[key] = csv_dir
    def _read_len_map(row):
        key = (row['files'], row['test'], int(row['time']))
        csv_dir = dir_map.get(key)
        if csv_dir is None:
            return np.nan
        csv_path = csv_dir / f"t={row['time']}matched_before.csv"
        if not csv_path.exists():
            return np.nan
        tmp = pd.read_csv(csv_path, index_col=0)
        tmp['Matched_After_IDs'] = tmp['Matched_After_IDs'].apply(ast.literal_eval)
        tmp['len'] = tmp['Matched_After_IDs'].str.len()
        return tmp['len'].get(row['ID'], np.nan)
    total_df['lenmatched_row'] = total_df.apply(_read_len_map, axis=1)
def add_len_to_others(src_df, other_files):##扩展到随机扰动的实验结果
    """
    src_df : 已经包含 lenmatched_row 的 DataFrame（当前 random_num 结果）
    other_files : 需要补充 lenmatched_row 的其他 csv 路径列表
    就地修改，无返回值
    """
    # 1. 做成 (ID,time,test,files) -> lenmatched_row 映射
    key_cols = ['ID', 'time', 'test', 'files']
    mapping = src_df.set_index(key_cols)['lenmatched_row'].to_dict()

    # 2. 逐个给其他文件补充列
    for f in other_files:
        df = pd.read_csv(f, index_col=0)
        df['lenmatched_row'] = df.set_index(key_cols).index.map(mapping)
        df.to_csv(f)          # 覆盖保存

def adaptive_bins(series, min_samples=50):
    """返回合并后的 IntervalIndex，保证每箱 ≥min_samples"""
    bins = np.arange(0, series.max() + 2, 2)
    cuts = pd.cut(series, bins=bins, right=False)
    count = cuts.value_counts().sort_index()
    # 从前往后合并直到 ≥30
    new_bins, cur_cnt, cur_left = [bins[0]], 0, bins[0]
    for iv, cnt in count.items():
        cur_cnt += cnt
        if cur_cnt >= min_samples:
            new_bins.append(iv.right)
            cur_cnt, cur_left = 0, iv.right
    if new_bins[-1] < bins[-1]:               # 兜底最后一个右端点
        new_bins.append(bins[-1])
    return np.array(new_bins)



###############################################################
# ######运行获得细胞消失模拟数据
#workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy0 = r'D:\LJY\CellPara\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy6  = os.path.join(workpy, '细胞失踪重定位')
workpy7  = os.path.join(workpy, '细胞失踪重定位-真实数据失踪情况')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy6, filename)
save_dir_test=os.path.join(save_dir, 'test')
#save_dir_test=os.path.join(save_dir, 'long_test_3_6')
#save_dir_test=os.path.join(save_dir, 'long_test_6_10')
#save_dir_test=os.path.join(save_dir, 'long_test_10_15')
#save_dir_test=os.path.join(save_dir, 'long_test_15_20')
os.makedirs(save_dir, exist_ok=True) 
os.makedirs(save_dir_test, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]
#events_path = os.path.join(save_dir, 'unique_events.csv')
#events=pd.read_csv(events_path, index_col=None)
point=df.rename(columns={'Time': 't'})
point['ID']=point.index



'''************************************************************************'''
'''检查真实数据的消失情况'''
###################################################################
u=calculate_nearest_id_distances(point)
# 创建新的索引列
u['Index'] = u['ID'].astype(str) + '-' + u['t'].astype(str)
point['Index'] = point['ID'].astype(str) + '-' + point['t'].astype(str)
# 设置索引
u.set_index('Index', inplace=True)
point.set_index('Index', inplace=True)
pa=('Volume', 'Ellipticity (oblate)', 'Ellipticity (prolate)','Area', 'Sphericity', 'Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C', 'Average Speed', 'Net Displacement', 'Confinement Ratio', 'Arrest Coefficient', 'MSD', 'Ellipsoid Axis A X', 'Ellipsoid Axis A Y', 'Ellipsoid Axis A Z')
Upoint=pd.concat([u, point.loc[:,pa]],axis=1)
last_moment = Upoint.groupby('ID')['t'].transform(max)
Upoint['Last_Moment'] = last_moment
first_moment = Upoint.groupby('ID')['t'].transform(min)
Upoint['First_Moment'] = first_moment
# 添加一个标记列，用于区分细胞是消失还是出现
Upoint['Event_Type'] = np.where(Upoint['t'] == Upoint['Last_Moment'], 0, 
                                np.where(Upoint['t'] == Upoint['First_Moment'], 1,  # 出现
                                     np.nan)) # 消失为0出现为1其他时刻标记为 NaN
# 计算每个细胞的平均速度并存储在新列中
average_speeds = Upoint.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
# 如果 data 里已存在同名列，则重命名新列
if 'Average_Speed' in Upoint.columns:
    average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
Upoint = Upoint.merge(average_speeds, on='ID')
# 计算每个细胞的平均位移并存储在新列中
average_displacement = Upoint.groupby('ID').apply(calculate_total_displacement).reset_index(name='Average_displacement')
# 如果 data 里已存在同名列，则重命名新列
if 'Average_displacement' in Upoint.columns:
    average_displacement = average_displacement.rename(columns={'Average_displacement': 'Average_displacement_adjust'})
Upoint = Upoint.merge(average_displacement, on='ID')
Upoint.index=Upoint['ID']
# 2. 只保留细胞消失或出现时刻对应的行
filtered_df_before = Upoint.dropna(subset=['Event_Type']).copy()
filtered_df_after = Upoint[Upoint['Event_Type'] == 1]
filtered_df_before = Upoint[Upoint['Event_Type'] == 0]
matched_before = attach_close_after_ids(filtered_df_before, filtered_df_after)
out_df = merge_overlapping_ids(matched_before)
matched_before.head()
out_df.head()
matched_before.to_csv(os.path.join(save_dir_test, 't='+str(time)+'matched_before.csv'))
out_df.to_csv(os.path.join(save_dir_test, 't='+str(time)+'out_df.csv'))
##matched_before:直接搜集相邻时空范围
##out_df：在matched_before基础上，将接近的消失出现时间汇总






'''************************************************************************'''
'''根据平均位移修正候选细胞的选择情况------随机扰动版本增加可信度'''
###################################################################运行进行稳定性验证
import numpy as np
target_num=5
#random_num=0
for random_num in np.arange(0, 101, 10)   :
    all_frames = []                       # 收集容器
    for matched_dir in matched_dirs:
        matched_dir = Path(matched_dir)
        print(f'\n>>> 处理 matched 文件夹: {matched_dir}')
        matched_csvfile=find_csv_dir(matched_dir)[1]
        if filename != matched_csvfile:
            filename=matched_csvfile
            df_path   = os.path.join(directory,filename)
            df = pd.read_csv(df_path, index_col=0)
            U=inspect_nan(df,'zero')
            df=U[0]
            df['ID']=df.index
            point=df.rename(columns={'Time': 't'})
            point['ID']=point.index
        # ---------- 1 解析 time ----------
        time_dir  = matched_dir.parent
        time_str  = time_dir.name            # 'time=20'
        time_val  = int(time_str.split('=')[-1])
        csv_dir = matched_dir.parent.parent
        # ---------- 2 读 before / after ----------
        before_file = csv_dir  / f't={time_val}before_time.csv'
        after_file  = csv_dir  / f't={time_val}after_time.csv'
        if not (before_file.exists() and after_file.exists()):
            print('⚠️  缺少 before/after csv，跳过')
            continue
        before_time = pd.read_csv(before_file)
        after_time  = pd.read_csv(after_file)
        timemax=max(df['Time'])
        Ubefore=slice_temporal(before_time, df,direction='before')
        Uafter=slice_temporal(after_time, df,direction='after',perturb=10,
                              tmin=time_val+1,tmax=timemax,random_seed=random_num)
        Ubefore=Ubefore.rename(columns={'Time': 't'})
        Uafter=Uafter.rename(columns={'Time': 't'})

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
        # 计算每个细胞的平均位移并存储在新列中
        average_displacement = Ubefore.groupby('ID').apply(calculate_total_displacement).reset_index(name='Average_displacement')
        # 如果 data 里已存在同名列，则重命名新列
        if 'Average_displacement' in Ubefore.columns:
            average_displacement = average_displacement.rename(columns={'Average_displacement': 'Average_displacement_adjust'})
        Ubefore = Ubefore.merge(average_displacement, on='ID')
        Ubefore.index=Ubefore['ID']
        # 2. 只保留细胞消失时刻对应的行
        filtered_df_before = Ubefore.dropna(subset=['Event_Type']).copy()

        # 计算每个细胞的平均速度并存储在新列中
        average_speeds = Uafter.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
        # 如果 data 里已存在同名列，则重命名新列
        if 'Average_Speed' in Uafter.columns:
            average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
        Uafter = Uafter.merge(average_speeds, on='ID')
        # 计算每个细胞的平均位移并存储在新列中
        average_displacement = Uafter.groupby('ID').apply(calculate_total_displacement).reset_index(name='Average_displacement')
        # 如果 data 里已存在同名列，则重命名新列
        if 'Average_displacement' in Uafter.columns:
            average_displacement = average_displacement.rename(columns={'Average_displacement': 'Average_displacement_adjust'})
        Uafter = Uafter.merge(average_displacement, on='ID')
        Uafter.index=Uafter['ID']
        # 2. 只保留细胞出现时刻对应的行
        filtered_df_after = Uafter.dropna(subset=['Event_Type']).copy()

        matched_before = attach_close_after_ids(filtered_df_before, filtered_df_after,target_n=target_num)
        # --- 新增部分：提取扰动后的对应时刻 ---
        matched_after_times = []
        for ids in matched_before['Matched_After_IDs']:
            if not isinstance(ids, list):
                ids = []  # 防止空值报错
            times = []
            for cid in ids:
                if cid in filtered_df_after.index:
                    times.append(float(filtered_df_after.loc[cid, 't']))
                else:
                    times.append(np.nan)
            matched_after_times.append(times)
        matched_before['Matched_After_Times'] = matched_after_times
        matched_before['ID_Time'] = matched_before['ID'].map(
            lambda cid: float(filtered_df_after.loc[cid, 't']) if cid in filtered_df_after.index else np.nan
        )
        #matched_file = csv_dir  / f't={time_val}matched_before_adjustbyth.csv'
        #matched_before.to_csv(matched_file)
        framedata=matched_before[['ID', 'Matched_After_IDs','Matched_After_Times','ID_Time']].copy()
        framedata['time']=time_val
        framedata['test']=csv_dir.name
        framedata['files']=csv_dir.parent.name
        all_frames.append(
            framedata.copy()
        )

    # 合并全部轮次
    total_df = pd.concat(all_frames, ignore_index=True)
    # 统计有多少行满足 ID 出现在对应 matched 列表里
    total_df['in_matched'] = total_df.apply(
        lambda r: r['ID'] in r['Matched_After_IDs'], axis=1
    )
    total_df['in_matched'].value_counts()
    total_df.to_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_{random_num}.csv'))

# ---------- 1. 建缓存（保持原样） ----------
df_cache = {}
for matched_dir in matched_dirs:
    csv_file = find_csv_dir(matched_dir)[1]      # files 列值
    if csv_file not in df_cache:
        df_path = os.path.join(directory, csv_file)
        df = pd.read_csv(df_path, index_col=0)
        df['ID'] = df.index
        df = df.rename(columns={'Time': 't'})
        df = df.drop_duplicates(subset=['ID', 't'])
        df_cache[csv_file] = df.set_index(['ID', 't'])

random_nums = np.arange(0, 101, 10)          # [0,10,...,100]
target_num=3
for rn in random_nums:
    total_df=pd.read_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_{rn}.csv'),index_col=0)
    # ---------- 3. 应用（显式传参） ----------
    total_df['Distances']     = total_df.apply(partial(calc_dist_row,  df_cache=df_cache), axis=1)
    total_df['selfDistances'] = total_df.apply(partial(self_dist_row, df_cache=df_cache), axis=1)
    total_df.to_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_{rn}.csv'))

summary = []               # 收集 Series
for rn in random_nums:
    fpath = os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_{rn}.csv')
    tmp = pd.read_csv(fpath, index_col=0)
    vc = tmp['in_matched'].value_counts()    # 返回 True/False 两数
    vc.name = rn                             # 列名
    summary.append(vc)
# 拼成 2×12 DataFrame，缺失补 0
df_summary = pd.concat(summary, axis=1).fillna(0).astype(int)
df_summary.index = ['True', 'False']         # 保证顺序
df_summary['Total'] = df_summary.sum(axis=1) # 累计列
print(df_summary)
df_summary.to_csv(os.path.join(workpy6,f'LSTM-step2检查真实数据-修正细胞搜索策略并检查效果---随机种子扰动检查细胞搜寻策略的效果稳定性_target_num_{target_num}.csv'))


###########################可视化---成功捕捉候选细胞的环状图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data=pd.read_csv(os.path.join(workpy6,f'LSTM-step2检查真实数据-修正细胞搜索策略并检查效果---随机种子扰动检查细胞搜寻策略的效果稳定性_target_num_{target_num}.csv'))
data = data.set_index(data.columns[0], drop=True)
data.index = data.index.astype(str)
# ==============================
# 数据准备
# ==============================
plot_df = data.copy()      # 12 列（含 total）
titles  = data.columns.astype(str) # 对应列名
cols    = plot_df.columns[:]             # 保留全部 12 列
# ==============================
# 配色
# ==============================
# ② 配色（四选一）
colors = ['#D8BFD8', '#FFF8DC']         # 柔和对比,Set2
# colors = sns.color_palette('Paired', 2)      # 高饱和配对色,Paired
# colors = sns.color_palette('Pastel1', 2)     # 淡雅 pastel,Pastel1
'''# 1 柔和莫兰迪
colors = ['#A8D8AD', '#F7A4A4']
# 7 冰淇淋马卡龙
colors = ['#D8BFD8', '#FFF8DC']
# 10 莫奈睡莲
colors = ['#7FB3D3', '#F7CAC9']'''
plt.style.use('default')
from matplotlib import gridspec
fig = plt.figure(figsize=(16, 12))
# 4×4 网格：左侧 3 列放环图，右侧第 3 列（索引 3）整列合并为竖直柱图
gs = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.3)
# 左侧 3 列：0-2 列，4 行全用 → 12 个环图
axes = [fig.add_subplot(gs[r, c], aspect='equal')
        for r in range(4) for c in range(3)]
# 右侧第 3 列整列合并为竖直柱状图
ax_bar = fig.add_subplot(gs[:, 3])
# ---------- 下面你的绘图循环完全不变 ----------
for ax, col, ttl in zip(axes, cols, titles):
    sizes = plot_df[col]
    ax.pie(sizes, labels=sizes.index, colors=colors,
           autopct=lambda p: f'{p:.1f}', pctdistance=0.75,
           startangle=90, textprops={'fontsize': 10, 'weight': 'semibold'})
    ax.add_artist(plt.Circle((0, 0), 0.55, fc='white'))
    ttl_clean = 'Total' if 'total' in str(ttl).lower() else f'random_seed={ttl}'
    ax.set_title(ttl_clean, size=13, weight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])
# ---------- 3. 右侧竖直柱状图 ----------
tot_true = plot_df['Total'].loc['True']
tot_fals = plot_df['Total'].loc['False']
x_pos = [0, 1]; vals = [tot_true, tot_fals]
width = 0.6
bars = ax_bar.bar(x_pos, vals, color=colors, width=width)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(['True', 'False'],
                       fontsize=10, weight='semibold', color='black')
ax_bar.set_ylabel('Cumulative count', fontsize=12)
ax_bar.set_title('Total Summary', fontsize=13, weight='bold')
# 数值标签置顶
for bar, v in zip(bars, vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f'{v:,.1f}', ha='center', va='bottom', fontsize=11, weight='semibold')
sns.despine(left=True, bottom=True, ax=ax_bar)
plt.tight_layout()
plt.savefig(
    os.path.join(workpy6, 'LSTM-step2随机种子扰动检查细胞搜寻策略的效果稳定性.pdf'),
    bbox_inches='tight'
)
plt.show()



######################可视化---成功捕捉候选细胞比例随候选细胞数量分布的情况（target_num=5的情况下）
total_df=pd.read_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_real.csv'),index_col=0)

add_lenmatched(total_df, workpy6=workpy6,matched_dirs=matched_dirs) 
total_df.to_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_real.csv'))

src_file   = os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_real.csv')   # 已带 lenmatched_row
other_list = [os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_{rn}.csv')
              for rn in range(0, 101, 10) ]   # 除 50 外的所有 random_num
src_df = pd.read_csv(src_file, index_col=0)
add_len_to_others(src_df, other_list)##计算原本候选细胞数量

 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from matplotlib import pyplot as plt
# ---------- 1. 分箱 ----------
bins = np.arange(0, total_df['lenmatched_row'].max() + 2, 2)   # 2 为箱宽，可改
total_df['bin'] = pd.cut(total_df['lenmatched_row'], bins=bins, right=False)
# ---------- 2. 每箱计算 True 占比 ----------
bin_stats = (total_df.groupby('bin')['in_matched']
                     .agg(['sum', 'count'])
                     .assign(percent=lambda x: x['sum'] / x['count'] * 100))
# 取区间中点作为横坐标
bin_stats['mid'] = bin_stats.index.apply(lambda b: b.mid)

# ---------- 1. 主题 & 颜色 ----------
sns.set_theme(style='whitegrid', font='Arial', palette='Set2')
line_color  = sns.color_palette('Set2')[0]   # 柔和蓝绿
fill_color  = sns.color_palette('Set2')[0]   # 同色系渐变
bg_bar_color = 'white'                       # 淡背景
# ---------- 2. 渐变面积图 ----------
plt.figure(figsize=(9, 5))
# 背景：半透明面积
plt.fill_between(bin_stats['mid'], bin_stats['percent'],
                 color=fill_color, alpha=0.25, label='True percentage area')
# 前景：圆润折线
plt.plot(bin_stats['mid'], bin_stats['percent'],
         color=line_color, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         linewidth=2.5, label='True percentage')
# ---------- 3. 自动 y 轴（含 margin） ----------
y_min, y_max = bin_stats['percent'].min()*0.9, bin_stats['percent'].max()*1.05
margin = (y_max - y_min) * 0.05
plt.ylim(y_min - margin, y_max + margin)
# ---------- 4. 细节 ----------
plt.xlabel('Matched length (bins)', fontsize=13)
plt.ylabel('True percentage (%)', fontsize=13)
plt.title('Recall Rate vs. Matched Length', fontsize=14, weight='semibold')
# 右上角小卡片：总样本数
n_total = len(total_df)
plt.text(0.98, 0.95, f'n = {n_total:,}', transform=plt.gca().transAxes,
         ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))
plt.legend(loc='lower right', frameon=False)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(
    os.path.join(workpy6, 'LSTM-step2细胞搜寻策略的效果稳定性与候选数量关系_target=5.pdf'),
    bbox_inches='tight'
)
plt.show()


############################美化并扩展
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec
target_num=3
param_line='selfDistances'#'lenmatched_row','dist_time','selfDistances'
'''
# ---------- 1. 颜色 & 主题 ----------
sns.set_theme(style='white', font='Arial')
line_cmap = sns.cubehelix_palette(
    start=2.8, rot=-0.4, n_colors=11, gamma=0.9, light=0.75, dark=0.25
)
mean_color = '#d62728'

# ---------- 2. 收集曲线 ----------
all_curves, all_mids = [], []
for rn in random_nums:
    total_df = pd.read_csv(os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_{rn}.csv'), index_col=0)
    total_df['dist_time']=total_df['ID_Time']-total_df['time']
    # 0. 对原始长度做自适应分箱（勿用已切箱列）
    raw_series = total_df[param_line]
    bins = adaptive_bins(raw_series , min_samples=50)
    total_df['bin'] = pd.cut(raw_series , bins=bins, right=False)
    curve = (total_df.groupby('bin')['in_matched']
             .agg(['sum', 'count'])
             .assign(percent=lambda x: x['sum'] / x['count'] * 100))
    curve['mid'] = pd.IntervalIndex(curve.index).mid
    curve['random_num'] = rn
    all_curves.append(curve[['mid', 'percent', 'random_num']])
    all_mids.append(curve.set_index('mid')['percent'])

df_curve = pd.concat(all_curves, ignore_index=True)

# ---------- 3. 计算 min-max 区间 ----------
mid_points = df_curve['mid'].unique()
min_max = (pd.concat(all_mids, axis=1)
           .reindex(mid_points)
           .agg(['min', 'max'], axis=1))
span = min_max['max'] - min_max['min']
min_max = min_max.assign(
    lower=min_max['min'] - span * 0.05,
    upper=min_max['max'] + span * 0.05
)

# ---------- 4. 图布局 ----------
fig = plt.figure(figsize=(11, 6))
gs = GridSpec(1, 20, figure=fig, wspace=0.05)  # 主图占大部分宽度
ax = fig.add_subplot(gs[0, :17])               # 主折线图
ax_box = fig.add_subplot(gs[0, 18:], sharey=ax)  # 右侧箱式图，共用 y 轴

# ---------- 5. 主图 ----------
ax.fill_between(mid_points, min_max['lower'], min_max['upper'],
                color='grey', alpha=0.15, label='Min–Max range (+5%)')

linestyles = [(0, (4, 3)), (0, (5, 2)), (0, (3, 4, 1, 4))]
for i, (rn, sub) in enumerate(df_curve.groupby('random_num')):
    style = linestyles[i % len(linestyles)]
    alpha = 0.5 + 0.05 * np.sin(i)
    ax.plot(sub['mid'], sub['percent'],
            color=line_cmap[i],
            linewidth=1.3,
            alpha=alpha,
            linestyle=style)

mean_curve = df_curve.groupby('mid')['percent'].mean()
ax.plot(mean_curve.index, mean_curve.values,
        color=mean_color, linewidth=3,
        marker='o', markersize=5,
        markerfacecolor='white', markeredgewidth=1.5,
        label='Mean')

ax.set_xlabel('Matched length (bins)', fontsize=13)
ax.set_ylabel('True percentage (%)', fontsize=13)
ax.set_title(f'Recall Rate vs. Matched Length (target={target_num})',
             fontsize=14, weight='semibold')
# 修改图例位置到左下角
ax.legend(loc='lower left', frameon=False)

y_min, y_max = min_max['lower'].min(), min_max['upper'].max()
ax.set_ylim(y_min - y_max*0.02, y_max + y_max*0.05)
sns.despine(ax=ax, right=True)


# ---------- 6. 箱式图 ----------
box = sns.boxplot(y=df_curve['percent'], ax=ax_box,
            width=0.5, color='#B0B0B0',
            fliersize=3, linewidth=1.2,
            boxprops={'alpha': 0.4})

# 去除 x 轴刻度与标签
ax_box.set_xticks([])
ax_box.set_xlabel('')
ax_box.set_ylabel('')

# 获取箱线图的关键统计值
stats = df_curve['percent'].describe()
quartiles = np.percentile(df_curve['percent'], [25, 50, 75])
iqr = quartiles[2] - quartiles[0]

# 检查max和Q3是否相同
show_max = not np.isclose(stats['max'], quartiles[2])

# 准备标注内容和位置
annotations = [
    {'text': f'min: {stats["min"]:.1f}%', 'val': stats['min'], 'xpos': 0.1},
    {'text': f'Q1: {quartiles[0]:.1f}%', 'val': quartiles[0], 'xpos': 0.3},
    {'text': f'median: {quartiles[1]:.1f}%', 'val': quartiles[1], 'xpos': 0.1},
    {'text': f'Q3: {quartiles[2]:.1f}%', 'val': quartiles[2], 'xpos': 0.3}
]

if show_max:
    annotations.append({'text': f'max: {stats["max"]:.1f}%', 'val': stats['max'], 'xpos': 0.1})

# 添加标注和连接线
for i, ann in enumerate(annotations):
    # 微小垂直偏移避免重叠
    y_offset = 0.02 if i % 2 else -0.02
    ax_box.text(ann['xpos'], ann['val'] + y_offset, ann['text'],
               fontsize=9, color='gray', va='center', ha='left')
    # 添加连接线
    ax_box.plot([ann['xpos']-0.05, ann['xpos']-0.01], [ann['val'], ann['val']],
               color='gray', linestyle='-', linewidth=0.8, alpha=0.6)

sns.despine(ax=ax_box, left=True)
plt.tight_layout()
plt.savefig(
    os.path.join(workpy6, f'LSTM-step2细胞搜寻策略的效果稳定性与候选数量关系_target={target_num}_{param_line}.pdf'),
    bbox_inches='tight'
)
plt.show()
'''
# %%

param_line = 'dist_time'  # 可选：'lenmatched_row', 'dist_time', 'selfDistances'



sns.set_theme(style='white', font='Arial')
line_cmap = sns.cubehelix_palette(
    start=2.8, rot=-0.4, n_colors=11, gamma=0.9, light=0.75, dark=0.25
)
mean_color = '#d62728'

# ============================================================
# 1️⃣ 统一计算全局分箱边界
# ============================================================
all_vals = []
for rn in random_nums:
    df_tmp = pd.read_csv(os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_{rn}.csv'), index_col=0)
    df_tmp['dist_time'] = df_tmp['ID_Time'] - df_tmp['time']
    all_vals.extend(df_tmp[param_line].dropna().values)

# 去除极端值
q_low, q_high = np.percentile(all_vals, [1, 99])
all_vals = np.array(all_vals)
all_vals = all_vals[(all_vals >= q_low) & (all_vals <= q_high)]

# 统一自适应分箱
bins = adaptive_bins(all_vals, min_samples=50)
print(f"📦 全局统一分箱数量: {len(bins)-1}")

# ============================================================
# 2️⃣ 主循环：使用统一分箱绘制各随机轮次曲线
# ============================================================
all_curves, all_mids = [], []
for rn in random_nums:
    total_df = pd.read_csv(os.path.join(workpy6, f'取top{target_num}搜集出现细胞召回结果_{rn}.csv'), index_col=0)
    total_df['dist_time'] = total_df['ID_Time'] - total_df['time']

    # 同样去掉极端值
    total_df = total_df[(total_df[param_line] >= q_low) & (total_df[param_line] <= q_high)]

    # 用统一分箱
    total_df['bin'] = pd.cut(total_df[param_line], bins=bins, right=False)
    curve = (total_df.groupby('bin')['in_matched']
             .agg(['sum', 'count'])
             .assign(percent=lambda x: x['sum'] / x['count'] * 100))
    curve['mid'] = pd.IntervalIndex(curve.index).mid
    curve['random_num'] = rn
    all_curves.append(curve[['mid', 'percent', 'random_num']])
    all_mids.append(curve.set_index('mid')['percent'])

df_curve = pd.concat(all_curves, ignore_index=True)

# ============================================================
# 3️⃣ 计算 min–max 区间
# ============================================================
mid_points = df_curve['mid'].unique()
mid_points = np.sort(mid_points)
min_max = (pd.concat(all_mids, axis=1)
           .reindex(mid_points)
           .agg(['min', 'max'], axis=1))
span = min_max['max'] - min_max['min']
min_max = min_max.assign(
    lower=min_max['min'] - span * 0.05,
    upper=min_max['max'] + span * 0.05
)

# ============================================================
# 4️⃣ 图布局
# ============================================================
fig = plt.figure(figsize=(11, 6))
gs = GridSpec(1, 20, figure=fig, wspace=0.05)
ax = fig.add_subplot(gs[0, :17])
ax_box = fig.add_subplot(gs[0, 18:], sharey=ax)

# ============================================================
# 5️⃣ 主折线图 + 多项式回归拟合均值曲线
# ============================================================
ax.fill_between(mid_points, min_max['lower'], min_max['upper'],
                color='grey', alpha=0.15, label='Min–Max range (+5%)')

linestyles = [(0, (4, 3)), (0, (5, 2)), (0, (3, 4, 1, 4))]
for i, (rn, sub) in enumerate(df_curve.groupby('random_num')):
    style = linestyles[i % len(linestyles)]
    alpha = 0.45 + 0.05 * np.cos(i)
    ax.plot(sub['mid'], sub['percent'],
            color=line_cmap[i],
            linewidth=1.3,
            alpha=alpha,
            linestyle=style)

# ---------- 均值多项式拟合 ----------
mean_curve = df_curve.groupby('mid')['percent'].mean()
x = mean_curve.index.values
y = mean_curve.values

# 使用三阶多项式拟合
coeffs = np.polyfit(x, y, deg=3)
poly_func = np.poly1d(coeffs)

x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = poly_func(x_smooth)

# 绘制拟合曲线
ax.plot(x_smooth, y_smooth,
        color=mean_color, linewidth=3,
        label='Mean (Polyfit, deg=3)')

# 绘制均值散点
ax.scatter(x, y,
           color='white', edgecolor=mean_color,
           linewidth=1.5, s=45, zorder=5)

# ---------- 细节美化 ----------
ax.set_xlabel(param_line, fontsize=13)
ax.set_ylabel('True percentage (%)', fontsize=13)
ax.set_title(f'Recall Rate vs. {param_line} (target={target_num})',
             fontsize=14, weight='semibold')
ax.legend(loc='lower left', frameon=False)

y_min, y_max = min_max['lower'].min(), min_max['upper'].max()
ax.set_ylim(y_min - y_max * 0.02, y_max + y_max * 0.05)
sns.despine(ax=ax, right=True)


# ============================================================
# 6️⃣ 箱式图
# ============================================================
sns.boxplot(y=df_curve['percent'], ax=ax_box,
            width=0.5, color='#B0B0B0',
            fliersize=3, linewidth=1.2,
            boxprops={'alpha': 0.4})
ax_box.set_xticks([])
ax_box.set_xlabel('')
ax_box.set_ylabel('')

# 标注关键位置
stats = df_curve['percent'].describe()
for name, val, dy in [('min', stats['min'], -2),
                      ('median', stats['50%'], 0),
                      ('max', stats['max'], 2)]:
    ax_box.text(0.3, val + dy, f'{name}: {val:.1f}%',
                fontsize=9, color='gray', va='center')

sns.despine(ax=ax_box, left=True)
plt.tight_layout()

plt.savefig(
    os.path.join(workpy6, f'LSTM-step2细胞搜寻策略稳定性_vs_{param_line}_target={target_num}.pdf'),
    bbox_inches='tight'
)
plt.show()








'''************************************************************************'''
'''根据平均位移修正候选细胞的选择情况'''
###################################################################真实运行获得数据
#########检查根据timegap修正的权值情况：
matched_dirs = [p for p in Path(workpy6).rglob('matched') if p.is_dir()]
print(f'共发现 {len(matched_dirs)} 个 matched 文件夹')

def find_csv_dir(start_dir: Path):
    """从 start_dir 逐级向上，返回第一个名字里带 '.csv' 的文件夹"""
    for par in start_dir.parents:          # 包含自己、祖父…
        if '.csv' in par.name:
            return par, par.name
    return None, None

import warnings
warnings.filterwarnings('ignore')          # ① 全局静默
all_frames = []                       # 收集容器
target_num=5
for matched_dir in matched_dirs:
    matched_dir = Path(matched_dir)
    print(f'\n>>> 处理 matched 文件夹: {matched_dir}')
    matched_csvfile=find_csv_dir(matched_dir)[1]
    if filename != matched_csvfile:
        filename=matched_csvfile
        df_path   = os.path.join(directory,filename)
        df = pd.read_csv(df_path, index_col=0)
        U=inspect_nan(df,'zero')
        df=U[0]
        df['ID']=df.index
        point=df.rename(columns={'Time': 't'})
        point['ID']=point.index
    # ---------- 1 解析 time ----------
    time_dir  = matched_dir.parent
    time_str  = time_dir.name            # 'time=20'
    time_val  = int(time_str.split('=')[-1])
    csv_dir = matched_dir.parent.parent
    # ---------- 2 读 before / after ----------
    before_file = csv_dir  / f't={time_val}before_time.csv'
    after_file  = csv_dir  / f't={time_val}after_time.csv'
    if not (before_file.exists() and after_file.exists()):
        print('⚠️  缺少 before/after csv，跳过')
        continue
    before_time = pd.read_csv(before_file)
    after_time  = pd.read_csv(after_file)

    Ubefore=slice_temporal(before_time, df,direction='before')
    Uafter=slice_temporal(after_time, df,direction='after')
    #Uafter=slice_temporal(after_time, df,direction='after',perturb=10,tmin=time_val+1,tmax=timemax)
    Ubefore=Ubefore.rename(columns={'Time': 't'})
    Uafter=Uafter.rename(columns={'Time': 't'})

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
    # 计算每个细胞的平均位移并存储在新列中
    average_displacement = Ubefore.groupby('ID').apply(calculate_total_displacement).reset_index(name='Average_displacement')
    # 如果 data 里已存在同名列，则重命名新列
    if 'Average_displacement' in Ubefore.columns:
        average_displacement = average_displacement.rename(columns={'Average_displacement': 'Average_displacement_adjust'})
    Ubefore = Ubefore.merge(average_displacement, on='ID')
    Ubefore.index=Ubefore['ID']
    # 2. 只保留细胞消失时刻对应的行
    filtered_df_before = Ubefore.dropna(subset=['Event_Type']).copy()

    # 计算每个细胞的平均速度并存储在新列中
    average_speeds = Uafter.groupby('ID').apply(calculate_average_speed).reset_index(name='Average_Speed')
    # 如果 data 里已存在同名列，则重命名新列
    if 'Average_Speed' in Uafter.columns:
        average_speeds = average_speeds.rename(columns={'Average_Speed': 'Average_Speed_adjust'})
    Uafter = Uafter.merge(average_speeds, on='ID')
    # 计算每个细胞的平均位移并存储在新列中
    average_displacement = Uafter.groupby('ID').apply(calculate_total_displacement).reset_index(name='Average_displacement')
    # 如果 data 里已存在同名列，则重命名新列
    if 'Average_displacement' in Uafter.columns:
        average_displacement = average_displacement.rename(columns={'Average_displacement': 'Average_displacement_adjust'})
    Uafter = Uafter.merge(average_displacement, on='ID')
    Uafter.index=Uafter['ID']
    # 2. 只保留细胞出现时刻对应的行
    filtered_df_after = Uafter.dropna(subset=['Event_Type']).copy()

    matched_before = attach_close_after_ids(filtered_df_before, filtered_df_after,target_n=target_num)
    matched_file = csv_dir  / f't={time_val}matched_before_adjustbyth.csv'
    matched_before.to_csv(matched_file)
    framedata=matched_before[['ID', 'Matched_After_IDs']].copy()
    framedata['time']=time_val
    framedata['test']=csv_dir.name
    framedata['files']=csv_dir.parent.name
    all_frames.append(
        framedata.copy()
    )

# 合并全部轮次
total_df = pd.concat(all_frames, ignore_index=True)

# 统计有多少行满足 ID 出现在对应 matched 列表里
total_df['in_matched'] = total_df.apply(
    lambda r: r['ID'] in r['Matched_After_IDs'], axis=1
)
total_df['in_matched'].value_counts()
total_df.to_csv(os.path.join(workpy6,f'取top{target_num}搜集出现细胞召回结果_real.csv'))





'''************************************************************************'''
'''运行匹配'''
###################################################################
for matched_dir in matched_dirs:
    matched_dir = Path(matched_dir)
    print(f'\n>>> 处理 matched 文件夹: {matched_dir}')
    matched_csvfile=find_csv_dir(matched_dir)[1]
    if filename != matched_csvfile:
        filename=matched_csvfile
        df_path   = os.path.join(directory,filename)
        df = pd.read_csv(df_path, index_col=0)
        U=inspect_nan(df,'zero')
        df=U[0]
        df['ID']=df.index
    # ---------- 1 解析 time ----------
    time_dir  = matched_dir.parent
    time_str  = time_dir.name            # 'time=20'
    time_val  = int(time_str.split('=')[-1])
    csv_dir = matched_dir.parent.parent
    # ---------- 2 读 before / after ----------
    before_file = csv_dir  / f't={time_val}before_time.csv'
    after_file  = csv_dir  / f't={time_val}after_time.csv'
    matched_file  = csv_dir  / f't={time_val}matched_before_adjustbyth.csv'
    if not (before_file.exists() and after_file.exists()):
        print('⚠️  缺少 before/after csv，跳过')
        continue
    before_time = pd.read_csv(before_file)
    after_time  = pd.read_csv(after_file)
    matched_before  = pd.read_csv(matched_file,usecols=['ID', 'Matched_After_IDs'],index_col=0)

    # ---------- 3 一次性算 timelen / timelen_fwd ----------
    timelen = {ID: len(df.loc[ID, 'Time'].loc[lambda t: t <  time_val])
               for ID in before_time['ID']}
    timelen_fwd = {ID: len(df.loc[ID, 'Time'].loc[lambda t: t >= time_val])
                   for ID in after_time['ID']}
    # ---------- 4 组装两个 df（保留你原风格） ----------
    df_T = before_time[['ID','Time']].copy()
    df_T['timelen'] = df_T['ID'].map(timelen)
    df_T.index = df_T['ID']
    df_T_after = after_time[['ID','Time']].copy()
    df_T_after['timelen_fwd'] = df_T_after['ID'].map(timelen_fwd)
    df_T_after.index = df_T_after['ID']
    df_T_after['time_gap_after']=df_T_after['Time']-time_val+1
    # ---------- 5 写回当前 matched 文件夹 ----------
    df_T.to_csv(csv_dir / f't={time_val}df_T.csv')
    df_T_after.to_csv(csv_dir / f't={time_val}df_T_after.csv')
    # ---------- 6 你原来的 pkl 处理（master/master_fwd） ----------
    pkl_files = [f for f in matched_dir.iterdir() if f.suffix == '.pkl']
    master, master_fwd = {}, {}
    for pkl_file in pkl_files:
        data = joblib.load(matched_dir / pkl_file)
        key = pkl_file.stem   # 去掉 .pkl
        # master
        sc = data['score_all']
        # 1. 确保是列表
        wanted_cols = ast.literal_eval(
            matched_before.loc[sc.index[0], 'Matched_After_IDs']
        )                                           # 现在是 list[str]
        # 2. 只保留存在的列（防止 KeyError）
        exist_cols = sc.columns.intersection(wanted_cols)
        sc_filtered = sc[exist_cols]
        master[key] = {r: {c: sc_filtered.loc[r, c] for c in sc_filtered.columns} for r in sc_filtered.index}
        # master_fwd
        sc_fwd = data['score_all_fwd'].T
        # 1. 确保是列表
        wanted_cols = ast.literal_eval(
            matched_before.loc[sc_fwd.index[0], 'Matched_After_IDs']
        )                                           # 现在是 list[str]
        # 2. 只保留存在的列（防止 KeyError）
        exist_cols = sc_fwd.columns.intersection(wanted_cols)
        sc_filtered_fwd = sc_fwd[exist_cols]
        master_fwd[key] = {r: {c: sc_filtered_fwd.loc[r, c] for c in sc_filtered_fwd.columns} for r in sc_filtered_fwd.index}
    result_df=sparse_min_cost_matching_gap_threshold(master, master_fwd, df_T, df_T_after,top_k=None)
    timebf=before_time.merge(after_time, on='ID', how='left')
    timebf['time_gap']=timebf.iloc[:,2]-timebf.iloc[:,1]-1
    timebf['A_id']=timebf['ID']
    timebf.to_csv(os.path.join(matched_dir, "timeGAP_beforeid.csv"))
    result_df_gaptime=result_df.merge(timebf.loc[:,['time_gap','A_id']], on='A_id', how='left')
    result_df_gaptime=result_df_gaptime.merge(df_T.rename(columns={'ID': 'A_id'}).loc[:,['timelen','A_id']], on='A_id', how='left')
    result_df_gaptime=result_df_gaptime.merge(df_T_after.rename(columns={'ID': 'A_id'}).loc[:,['timelen_fwd','A_id']], on='A_id', how='left')
    result_df_gaptime.to_csv(os.path.join(matched_dir, "df_rank_Adjust_merge0.csv"))





#检查全部的结果：成功矩阵
# 收集 (time_gap, Goal) 行
gap_goal = []          # list[DataFrame]
for md in matched_dirs:
    f = Path(md) / 'df_rank_Adjust_merge0.csv'
    if f.exists():
        # 只读需要的两列
        gap_goal.append(pd.read_csv(f, usecols=['time_gap', 'Goal']))

# 拼成长表
df_long = pd.concat(gap_goal, ignore_index=True)
# 计数表：行=time_gap，列=Goal，值=出现次数
count_tbl = (df_long.pivot_table(index='time_gap',
                                columns='Goal',
                                aggfunc='size',
                                fill_value=0)
                    .rename_axis('Goal', axis=1))   # 可选：给列索引起个名字
count_tbl['ratio']=count_tbl[1.0]/(count_tbl[-1.0]+count_tbl[1.0])
print(count_tbl)
df_long['Goal'].value_counts(dropna=False)



#检查全部的结果：单匹配-整体成功率
match = []                         # 收集 (match, time_gap) 行
for md in matched_dirs:
    f = Path(md) / 'df_match_flag.csv'
    g = Path(md) / 'timeGAP_beforeid.csv'
    if not (f.exists() and g.exists()):
        continue
    F = pd.read_csv(f, index_col=0)
    G = pd.read_csv(g, index_col=0)
    if F.shape[0] == 0:
        continue
    # 合并后只取需要的两列
    Fgap = F.merge(G[['ID', 'time_gap']], on='ID', how='left')
    match.append(Fgap[['match', 'time_gap']])   # ← 双括号保持 DataFrame

# 纵向连接
match_all = pd.concat(match, ignore_index=True)
match_all.value_counts() 
# 计数表：行=time_gap，列=Goal，值=出现次数
count_tbl_match = (pd.DataFrame(match_all).pivot_table(index='time_gap',
                                columns='match',
                                aggfunc='size',
                                fill_value=0)
                    .rename_axis('match', axis=1))   # 可选：给列索引起个名字
print(count_tbl_match)
































































def combine_with_gap_threshold(c_fwd, c_bwd, wA, wB, lenA, lenB, gap,
                               nA=1.2, nB=1.8,
                               discard_ratio=True,
                               penalty_factor_A=2.0,
                               penalty_factor_B=2.0):
    """
    基于 gap 阈值的方向加权（支持软惩罚）：
    - 若 discard_ratio=True：不满足 len < gap*n 则丢弃该方向
    - 若 discard_ratio=False：不满足方向乘上 penalty_factor 惩罚后再参与加权
    - 若两者均无效 → 返回 None
    """
    a_valid = (not np.isnan(c_fwd))
    b_valid = (not np.isnan(c_bwd))
    if not a_valid and not b_valid:
        return None

    # 检查方向是否满足 gap 阈值
    a_satisfy = (a_valid and lenA >= gap * nA)
    b_satisfy = (b_valid and lenB >= gap * nB)

    # ---- 硬删除模式 ----
    if discard_ratio:
        if not a_satisfy and not b_satisfy:
            return None
        elif a_satisfy and b_satisfy:
            return (wA * c_fwd + wB * c_bwd) / (wA + wB)
        elif a_satisfy:
            return c_fwd
        elif b_satisfy:
            return c_bwd
        else:
            return None

    # ---- 软惩罚模式 ----
    else:
        if not a_satisfy:
            if a_valid:
                c_fwd *= penalty_factor_A
        if not b_satisfy:
            if b_valid:
                c_bwd *= penalty_factor_B

        # 若两者都有效或经过惩罚，则继续组合
        if a_valid and b_valid:
            return (wA * c_fwd + wB * c_bwd) / (wA + wB)
        elif a_valid:
            return c_fwd
        elif b_valid:
            return c_bwd
        else:
            return None


def sparse_min_cost_matching_gap_threshold(master, master_fwd, df_T, df_T_after,
                                           weight_func=np.sqrt,
                                           candidate_tau=None,
                                           top_k=5,
                                           add_consistency_penalty=False,
                                           penalty_lambda=0.0,
                                           fill_unmatched=True,
                                           # ---- gap阈值参数 ----
                                           nA=1.2, nB=1.8,
                                           discard_ratio=True,
                                           penalty_factor_A=1.5,
                                           penalty_factor_B=1.5):
    """
    稀疏最小费用匹配算法（支持 gap 阈值 + 软惩罚模式）
    -------------------------------------------------
    返回列：
    ['A_id','B_id','cost','c_fwd','c_bwd','wA','wB','gap','Goal']
    """

    # -------- flatten --------
    master = flatten_master(master)
    master_fwd = flatten_master_fwd(master_fwd)

    A_all = sorted(master.keys())
    B_set = set()
    for a in master:
        B_set.update(master[a].keys())
    B_all = sorted(B_set)

    def w_for_A(a):
        return weight_func(df_T.loc[a, 'timelen']) if (a in df_T.index and 'timelen' in df_T.columns) else 1.0
    def w_for_B(b):
        return weight_func(df_T_after.loc[b, 'timelen_fwd']) if (b in df_T_after.index and 'timelen_fwd' in df_T_after.columns) else 1.0

    edges = []
    for a in A_all:
        for b in master[a].keys():
            c_fwd = master[a].get(b, np.nan)
            c_bwd = master_fwd.get(b, {}).get(a, np.nan)
            if np.isnan(c_fwd) and np.isnan(c_bwd):
                continue

            if (a not in df_T.index) or (b not in df_T_after.index):
                continue

            time_A = df_T.loc[a, 'Time']
            time_B = df_T_after.loc[b, 'Time']
            gap = abs(float(time_B) - float(time_A))

            lenA = float(df_T.loc[a, 'timelen'])
            lenB = float(df_T_after.loc[b, 'timelen_fwd'])
            wA = w_for_A(a)
            wB = w_for_B(b)

            c_comb_base = combine_with_gap_threshold(
                c_fwd, c_bwd, wA, wB, lenA, lenB, gap,
                nA=nA, nB=nB,
                discard_ratio=discard_ratio,
                penalty_factor_A=penalty_factor_A,
                penalty_factor_B=penalty_factor_B
            )
            if c_comb_base is None:
                continue

            if add_consistency_penalty:
                c_comb_base = c_comb_base + penalty_lambda * abs(
                    (c_fwd if not np.isnan(c_fwd) else c_comb_base) -
                    (c_bwd if not np.isnan(c_bwd) else c_comb_base)
                )

            edges.append((a, b, float(c_comb_base),
                          float(c_fwd) if not np.isnan(c_fwd) else np.nan,
                          float(c_bwd) if not np.isnan(c_bwd) else np.nan,
                          float(wA), float(wB), float(gap)))

    edges_df = pd.DataFrame(edges, columns=['A','B','c_comb','c_fwd','c_bwd','wA','wB','gap'])

    # -------- 候选筛选 --------
    if candidate_tau is not None:
        cand_df = edges_df[edges_df['c_comb'] <= candidate_tau].copy()
    elif top_k is None:
        cand_df = edges_df.copy()
    else:
        edges_df_sorted = edges_df.sort_values(['A','c_comb'])
        cand_list = []
        for a, g in edges_df_sorted.groupby('A'):
            top = g.head(top_k)
            cand_list.append(top)
        cand_df = pd.concat(cand_list, ignore_index=True) if cand_list else pd.DataFrame(columns=edges_df.columns)

    if cand_df.empty:
        if fill_unmatched:
            out = pd.DataFrame({
                'A_id': A_all,
                'B_id': [None]*len(A_all),
                'cost': [np.nan]*len(A_all),
                'c_fwd': [np.nan]*len(A_all),
                'c_bwd': [np.nan]*len(A_all),
                'wA': [np.nan]*len(A_all),
                'wB': [np.nan]*len(A_all),
                'gap':[np.nan]*len(A_all),
                'Goal':[0]*len(A_all)
            })
            return out
        else:
            return pd.DataFrame(columns=['A_id','B_id','cost','c_fwd','c_bwd','wA','wB','gap','Goal'])

    # -------- 建立流图 --------
    G = nx.DiGraph()
    source, sink = 'SRC', 'SNK'
    G.add_node(source); G.add_node(sink)

    for a in A_all:
        G.add_node(('A', a))
        G.add_edge(source, ('A', a), capacity=1, weight=0)
    for b in B_all:
        G.add_node(('B', b))
        G.add_edge(('B', b), sink, capacity=1, weight=0)

    scale = 100000
    for _, row in cand_df.iterrows():
        a, b = row['A'], row['B']
        cost = float(row['c_comb'])
        int_cost = int(round(cost * scale))
        G.add_edge(('A', a), ('B', b), capacity=1, weight=int_cost)

    try:
        flowDict = nx.max_flow_min_cost(G, source, sink)
    except Exception:
        flowDict = nx.min_cost_flow(G)

    # -------- 抽取匹配结果 --------
    matches = []
    for a in A_all:
        from_node = ('A', a)
        if from_node not in flowDict:
            continue
        for to_node, f in flowDict[from_node].items():
            if f > 0 and isinstance(to_node, tuple) and to_node[0] == 'B':
                b = to_node[1]
                row = cand_df[(cand_df['A']==a)&(cand_df['B']==b)].iloc[0]
                matches.append({
                    'A_id': a,
                    'B_id': b,
                    'cost': row['c_comb'],
                    'c_fwd': row['c_fwd'],
                    'c_bwd': row['c_bwd'],
                    'wA': row['wA'],
                    'wB': row['wB'],
                    'gap': row['gap']
                })

    result_df = pd.DataFrame(matches)

    if fill_unmatched:
        allA_df = pd.DataFrame({'A_id': A_all})
        result_df = allA_df.merge(result_df, how='left', on='A_id')


    # 先统一加 Goal 列（无论是否空表）
    if result_df.empty:
        result_df['Goal'] = pd.Series(dtype=int)
    else:
        result_df['Goal'] = np.where(
            result_df['B_id'].isna(),          # NaN
            0,
            np.where(result_df['A_id'] == result_df['B_id'], 1, -1)
        )

    return result_df[['A_id','B_id','cost','c_fwd','c_bwd','wA','wB','gap','Goal']].sort_values('cost').reset_index(drop=True)



























