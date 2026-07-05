
# ----------------- 自动检测不同cluster的碰撞事件 -----------------

def _choose_position_columns(columns: list) -> list:
    """根据可用列选择用于碰撞检测的位置列，优先XYZ，其次XY。"""
    xyz = ['Position X', 'Position Y', 'Position Z']
    xy = ['Position X', 'Position Y']
    if all(c in columns for c in xyz):
        return xyz
    if all(c in columns for c in xy):
        return xy
    # 备用：Center of Image Mass ch1
    xyz2 = ['Center of Image Mass X_ch1', 'Center of Image Mass Y_ch1', 'Center of Image Mass Z_ch1']
    xy2 = ['Center of Image Mass X_ch1', 'Center of Image Mass Y_ch1']
    if all(c in columns for c in xyz2):
        return xyz2
    if all(c in columns for c in xy2):
        return xy2
    return []

def calculate_data_statistics(df: pd.DataFrame, 
                              time_col: str = 'Time',
                              cluster_col: str = 'cluster') -> dict:
    """
    计算数据集的统计参考值，用于动态阈值计算。
    
    返回:
    dict: 包含各种参数的统计信息
    """
    stats = {}
    
    # 1. 体积统计
    if 'Volume' in df.columns:
        volume_data = df['Volume'].dropna()
        if not volume_data.empty:
            stats['avg_volume'] = volume_data.mean()
            stats['std_volume'] = volume_data.std()
            stats['median_volume'] = volume_data.median()
        else:
            stats['avg_volume'] = 1000  # 默认值
            stats['std_volume'] = 200
            stats['median_volume'] = 1000
    
    # 2. 面积统计
    if 'Area' in df.columns:
        area_data = df['Area'].dropna()
        if not area_data.empty:
            stats['avg_area'] = area_data.mean()
            stats['std_area'] = area_data.std()
            stats['median_area'] = area_data.median()
        else:
            stats['avg_area'] = 100  # 默认值
            stats['std_area'] = 20
            stats['median_area'] = 100
    
    # 3. 椭球体轴长统计
    axis_cols = ['Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C']
    if all(col in df.columns for col in axis_cols):
        all_axes = []
        for col in axis_cols:
            axis_data = df[col].dropna()
            all_axes.extend(axis_data.tolist())
        
        if all_axes:
            stats['avg_axis_length'] = np.mean(all_axes)
            stats['std_axis_length'] = np.std(all_axes)
            stats['median_axis_length'] = np.median(all_axes)
        else:
            stats['avg_axis_length'] = 10  # 默认值
            stats['std_axis_length'] = 2
            stats['median_axis_length'] = 10
    else:
        stats['avg_axis_length'] = 10
        stats['std_axis_length'] = 2
        stats['median_axis_length'] = 10
    
    # 4. 速度统计
    if 'Speed' in df.columns:
        speed_data = df['Speed'].dropna()
        if not speed_data.empty:
            stats['avg_speed'] = speed_data.mean()
            stats['std_speed'] = speed_data.std()
            stats['median_speed'] = speed_data.median()
        else:
            stats['avg_speed'] = 10  # 默认值
            stats['std_speed'] = 2
            stats['median_speed'] = 10
    else:
        stats['avg_speed'] = 10
        stats['std_speed'] = 2
        stats['median_speed'] = 10
    
    # 5. 加速度统计
    if 'Acceleration' in df.columns:
        accel_data = df['Acceleration'].dropna()
        if not accel_data.empty:
            stats['avg_acceleration'] = accel_data.mean()
            stats['std_acceleration'] = accel_data.std()
            stats['median_acceleration'] = accel_data.median()
            stats['max_acceleration'] = accel_data.abs().max()
        else:
            stats['avg_acceleration'] = 5  # 默认值
            stats['std_acceleration'] = 1
            stats['median_acceleration'] = 5
            stats['max_acceleration'] = 10
    else:
        stats['avg_acceleration'] = 5
        stats['std_acceleration'] = 1
        stats['median_acceleration'] = 5
        stats['max_acceleration'] = 10
    
    # 6. 球形度统计
    if 'Sphericity' in df.columns:
        sphericity_data = df['Sphericity'].dropna()
        if not sphericity_data.empty:
            stats['avg_sphericity'] = sphericity_data.mean()
            stats['std_sphericity'] = sphericity_data.std()
            stats['median_sphericity'] = sphericity_data.median()
        else:
            stats['avg_sphericity'] = 0.5  # 默认值
            stats['std_sphericity'] = 0.1
            stats['median_sphericity'] = 0.5
    else:
        stats['avg_sphericity'] = 0.5
        stats['std_sphericity'] = 0.1
        stats['median_sphericity'] = 0.5
    
    return stats

def calculate_dynamic_collision_threshold(cell_data: pd.DataFrame,
                                          time_col: str = 'Time',
                                          base_threshold: float = 2.0,
                                          volume_weight: float = 0.3,
                                          speed_weight: float = 0.2,
                                          shape_weight: float = 0.2,
                                          safety_factor: float = 1.5,
                                          data_stats: dict = None) -> pd.Series:
    """
    为单个细胞计算动态碰撞阈值，基于其几何特征和运动状态。
    
    参数:
    cell_data: 单个细胞的时序数据DataFrame
    time_col: 时间列名
    base_threshold: 基础阈值
    volume_weight: 体积影响权重
    speed_weight: 速度影响权重  
    shape_weight: 形状影响权重
    safety_factor: 安全系数
    data_stats: 数据集统计信息，如果为None则自动计算
    
    返回:
    每个时刻的动态阈值Series，索引为时间
    """
    # 如果没有提供统计信息，则从cell_data计算
    if data_stats is None:
        data_stats = calculate_data_statistics(cell_data, time_col)
    
    thresholds = []
    
    for _, row in cell_data.iterrows():
        threshold = base_threshold
        
        # 1. 体积影响（Volume或Area）
        volume_factor = 1.0
        if 'Volume' in row and not pd.isna(row['Volume']):
            # 使用实际数据的平均体积作为参考
            ref_volume = data_stats['avg_volume']
            volume_factor = (row['Volume'] / ref_volume) ** (1/3)
        elif 'Area' in row and not pd.isna(row['Area']):
            # 使用实际数据的平均面积作为参考
            ref_area = data_stats['avg_area']
            volume_factor = (row['Area'] / ref_area) ** (1/2)
        
        # 2. 形状影响（椭球体轴长、球形度）
        shape_factor = 1.0
        if all(col in row for col in ['Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C']):
            if not any(pd.isna(row[col]) for col in ['Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C']):
                # 使用实际数据的平均轴长作为参考
                ref_axis = data_stats['avg_axis_length']
                max_axis = max(row['Ellipsoid Axis Length A'], row['Ellipsoid Axis Length B'], row['Ellipsoid Axis Length C'])
                shape_factor = max_axis / ref_axis
        
        # 3. 速度影响
        speed_factor = 1.0
        if 'Speed' in row and not pd.isna(row['Speed']):
            # 使用实际数据的平均速度作为参考
            ref_speed = data_stats['avg_speed']
            speed_factor = 1 + (row['Speed'] / ref_speed)
        
        # 4. 加速度影响
        accel_factor = 1.0
        if 'Acceleration' in row and not pd.isna(row['Acceleration']):
            # 使用实际数据的最大加速度作为参考
            ref_accel = data_stats['max_acceleration']
            accel_factor = 1 + abs(row['Acceleration']) / ref_accel
        
        # 5. 球形度影响（越接近球形，阈值越小）
        sphericity_factor = 1.0
        if 'Sphericity' in row and not pd.isna(row['Sphericity']):
            # 使用实际数据的平均球形度作为参考
            ref_sphericity = data_stats['avg_sphericity']
            # 球形度差异的影响：越接近平均球形度，因子越接近1
            sphericity_diff = abs(row['Sphericity'] - ref_sphericity)
            sphericity_factor = 0.8 + 0.4 * (1 - sphericity_diff)  # 差异越小，因子越接近1.2
        
        # 综合计算动态阈值
        dynamic_threshold = base_threshold * (
            volume_weight * volume_factor +
            shape_weight * shape_factor +
            speed_weight * speed_factor +
            (1 - volume_weight - shape_weight - speed_weight) * accel_factor
        ) * sphericity_factor * safety_factor
        
        thresholds.append(dynamic_threshold)
    
    return pd.Series(thresholds, index=cell_data[time_col])

def calculate_cell_effective_radius(df: pd.DataFrame,
                                    time_col: str = 'Time',
                                    volume_col: str = 'Volume',
                                    dramanum: int = 1,
                                    area_col: str = 'Area') -> pd.DataFrame:
    """
    为每个细胞计算有效半径，基于细胞自身的体积和形状参数。
    
    参数:
    df: 包含细胞数据的DataFrame
    time_col: 时间列名
    volume_col: 体积列名
    area_col: 面积列名
    
    返回:
    添加了'Effective_Radius'列的DataFrame
    """
    df_with_radius = df.copy()
    df_with_radius['Effective_Radius'] = np.nan
    
    for track_id in df.index.unique():
        cell_data = df.loc[track_id]
        if isinstance(cell_data, pd.Series):
            cell_data = cell_data.to_frame().T
        
        for idx, row in cell_data.iterrows():
            effective_radius = 0.0
            
            # 1. 基于体积计算基础球半径
            if volume_col in row and not pd.isna(row[volume_col]):
                # 假设细胞为球形，V = (4/3)πr³，则 r = (3V/(4π))^(1/3)
                volume = row[volume_col]
                if volume > 0:
                    base_radius = (3 * volume / (4 * np.pi)) ** (1/3)
                    effective_radius = base_radius
            elif area_col in row and not pd.isna(row[area_col]):
                # 如果没有体积数据，使用面积估算
                area = row[area_col]
                if area > 0:
                    # 假设为圆形，A = πr²，则 r = sqrt(A/π)
                    base_radius = np.sqrt(area / np.pi)
                    effective_radius = base_radius
            
            # 2. 根据形状参数调整半径
            if effective_radius > 0:
                # 椭球体轴长调整
                if all(col in row for col in ['Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C']):
                    if not any(pd.isna(row[col]) for col in ['Ellipsoid Axis Length A', 'Ellipsoid Axis Length B', 'Ellipsoid Axis Length C']):
                        axis_a = row['Ellipsoid Axis Length A']
                        axis_b = row['Ellipsoid Axis Length B']
                        axis_c = row['Ellipsoid Axis Length C']
                        
                        # 计算椭球体的等效半径（保持体积不变）
                        # V_ellipsoid = (4/3)πabc，等效球半径 r = (abc)^(1/3)
                        ellipsoid_radius = (axis_a * axis_b * axis_c) ** (1/3)
                        
                        # 使用椭球体等效半径和体积计算半径的加权平均
                        effective_radius = 0.7 * effective_radius + 0.3 * ellipsoid_radius
                
                # 球形度调整
                if 'Sphericity' in row and not pd.isna(row['Sphericity']):
                    sphericity = row['Sphericity']
                    # 球形度越高，越接近理想球形，使用体积计算的半径
                    # 球形度越低，形状越不规则，需要更大的半径来覆盖
                    sphericity_factor = 0.8 + 0.4 * sphericity  # 0.8-1.2
                    effective_radius *= sphericity_factor
                
                # 椭球度调整
                if 'Ellipticity (oblate)' in row and not pd.isna(row['Ellipticity (oblate)']):
                    ellipticity = row['Ellipticity (oblate)']
                    # 椭球度越高，形状越扁平，需要更大的半径
                    ellipticity_factor = 1.0 + 0.3 * ellipticity
                    effective_radius *= ellipticity_factor
                elif 'Ellipticity (prolate)' in row and not pd.isna(row['Ellipticity (prolate)']):
                    ellipticity = row['Ellipticity (prolate)']
                    # 长椭球度越高，形状越细长，需要更大的半径
                    ellipticity_factor = 1.0 + 0.3 * ellipticity
                    effective_radius *= ellipticity_factor
            
            # 3. 运动状态调整（可选，用于更精确的碰撞检测）
            if 'Speed' in row and not pd.isna(row['Speed']):
                speed = row['Speed']
                # 速度越快，需要更大的安全半径
                speed_factor = 1.0 + 0.1 * speed  # 速度每增加10单位，半径增加10%
                effective_radius *= speed_factor
            
            # 4. 确保最小半径
            min_radius = 1.0  # 最小半径阈值
            effective_radius = max(effective_radius, min_radius)
            
            # 将计算的有效半径添加到DataFrame
            df_with_radius.loc[idx, 'Effective_Radius'] = effective_radius*dramanum
    
    return df_with_radius

def detect_cluster_collisions_radius_based(df: pd.DataFrame,
                                          time_col: str = 'Time',
                                          dramanum: int = 1,
                                          cluster_col: str = 'cluster') -> pd.DataFrame:
    """
    基于细胞有效半径的碰撞检测。
    两个细胞的距离小于它们有效半径之和时判定为碰撞。
    """
    if time_col not in df.columns:
        raise ValueError(f"未找到时间列 {time_col}")
    if cluster_col not in df.columns:
        raise ValueError(f"未找到标签列 {cluster_col}")

    pos_cols = _choose_position_columns(df.columns.tolist())
    if not pos_cols:
        raise ValueError("未找到用于碰撞检测的位置列（Position或Center of Image Mass）")

    # 计算每个细胞的有效半径
    print("Calculating effective radius for each cell...")
    df_with_radius = calculate_cell_effective_radius(df, time_col,dramanum=dramanum)
    
    # 保存有效半径数据（可选）
    radius_stats = df_with_radius['Effective_Radius'].describe()
    print(f"Effective radius statistics:\n{radius_stats}")

    events = []
    
    # 按时间分组处理
    for t, g in df_with_radius.groupby(time_col):
        # 仅保留拥有完整位置、cluster和有效半径的行
        g_valid = g.dropna(subset=pos_cols + [cluster_col, 'Effective_Radius']).copy()
        if g_valid.empty:
            continue
        
        # 取索引（TrackID）与位置矩阵
        track_ids = g_valid.index.values
        positions = g_valid[pos_cols].values.astype(float)
        clusters = g_valid[cluster_col].values
        radii = g_valid['Effective_Radius'].values
        
        # 两两配对
        for i, j in itertools.combinations(range(len(track_ids)), 2):
            if clusters[i] == clusters[j]:
                continue
                
            # 计算距离
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # 计算两个细胞有效半径之和
            radius_sum = radii[i] + radii[j]
            
            # 判定碰撞：距离小于半径之和
            if dist <= radius_sum:
                events.append({
                    'Time': t,
                    'TrackID_A': track_ids[i],
                    'TrackID_B': track_ids[j],
                    'Cluster_A': clusters[i],
                    'Cluster_B': clusters[j],
                    'Distance': float(dist),
                    'Radius_A': float(radii[i]),
                    'Radius_B': float(radii[j]),
                    'Radius_Sum': float(radius_sum),
                    'Overlap': float(radius_sum - dist)  # 重叠程度
                })
    
    return pd.DataFrame(events)

def analyze_collision_thresholds(df: pd.DataFrame,
                                 time_col: str = 'Time',
                                 cluster_col: str = 'cluster',
                                 base_threshold: float = 2.0) -> pd.DataFrame:
    """
    分析所有细胞的动态阈值分布，用于调试和参数优化。
    """
    all_thresholds = []
    
    for track_id in df.index.unique():
        cell_data = df.loc[track_id]
        if isinstance(cell_data, pd.Series):
            cell_data = cell_data.to_frame().T
            
        thresholds = calculate_dynamic_collision_threshold(cell_data, time_col, base_threshold)
        
        for time, threshold in thresholds.items():
            cluster = cell_data[cell_data[time_col] == time][cluster_col].iloc[0] if not cell_data[cell_data[time_col] == time].empty else None
            
            all_thresholds.append({
                'TrackID': track_id,
                'Time': time,
                'Cluster': cluster,
                'Dynamic_Threshold': threshold
            })
    
    threshold_df = pd.DataFrame(all_thresholds)
    
    # 统计分析
    print("动态阈值统计:")
    print(f"平均阈值: {threshold_df['Dynamic_Threshold'].mean():.3f}")
    print(f"标准差: {threshold_df['Dynamic_Threshold'].std():.3f}")
    print(f"最小值: {threshold_df['Dynamic_Threshold'].min():.3f}")
    print(f"最大值: {threshold_df['Dynamic_Threshold'].max():.3f}")
    
    # 按cluster分组统计
    if cluster_col in threshold_df.columns:
        cluster_stats = threshold_df.groupby('Cluster')['Dynamic_Threshold'].agg(['mean', 'std', 'min', 'max'])
        print("\n按Cluster分组的阈值统计:")
        print(cluster_stats)
    
    return threshold_df

def extract_touch_pair_timeseries(df: pd.DataFrame,
                                  event_row: pd.Series,
                                  time_col: str = 'Time',
                                  feature_cols: list = None) -> tuple:
    """
    根据一条碰撞事件记录，提取两条细胞的全时序数据与接触时刻。
    返回: touchcellpoint, othertouchcellpoint, touchcellT
    注：保持列包含时间列与特征列；若feature_cols为None，则保留数值列与时间列。
    """
    if time_col not in df.columns:
        raise ValueError(f'未找到时间列 {time_col}')

    trackA = event_row['TrackID_A']
    trackB = event_row['TrackID_B']
    touchcellT = event_row['Time']

    A_df = df.loc[trackA].copy()
    B_df = df.loc[trackB].copy()
    # 若单行会成为Series，统一为DataFrame
    if isinstance(A_df, pd.Series):
        A_df = A_df.to_frame().T
    if isinstance(B_df, pd.Series):
        B_df = B_df.to_frame().T

    A_df = A_df.sort_values(by=time_col)
    B_df = B_df.sort_values(by=time_col)

    if feature_cols is None:
        # 选取数值列
        num_cols_A = A_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_B = B_df.select_dtypes(include=[np.number]).columns.tolist()
        common = sorted(set(num_cols_A).intersection(num_cols_B))
        # 保留时间列
        if time_col not in common:
            common = [time_col] + common
        else:
            # 确保时间列在首位
            common = [time_col] + [c for c in common if c != time_col]
        feature_cols = common

    # 子集并重置索引（可选）
    A_sel = A_df[feature_cols].reset_index(drop=True)
    B_sel = B_df[feature_cols].reset_index(drop=True)

    # 若上游逻辑使用't'命名的时间列，复制一份
    if 't' not in A_sel.columns:
        A_sel = A_sel.rename(columns={time_col: 't'}) if time_col in A_sel.columns else A_sel
    if 't' not in B_sel.columns:
        B_sel = B_sel.rename(columns={time_col: 't'}) if time_col in B_sel.columns else B_sel

    # 补充行名
    A_sel.index = pd.Index([trackA] * len(A_sel), name='TrackID')
    B_sel.index = pd.Index([trackB] * len(B_sel), name='TrackID')
    return A_sel, B_sel, touchcellT

def filter_unique_cell_collisions(events_df: pd.DataFrame,
                                  time_col: str = 'Time',
                                  trackid_a_col: str = 'TrackID_A',
                                  trackid_b_col: str = 'TrackID_B') -> pd.DataFrame:
    """
    每个时刻，若 TrackID_A 或 TrackID_B 出现 ≥2 次，则丢弃该行。
    仅保留所有 TrackID 在该时刻唯一出现的行。

    返回：去重后的 DataFrame（行数 ≥ 原始行数/2）。
    """
    if events_df.empty:
        return events_df

    # 1. 统计每个 (Time, TrackID) 出现次数
    counts = (
        events_df
        .melt(id_vars=[time_col],
              value_vars=[trackid_a_col, trackid_b_col],
              value_name='TrackID')
        .groupby([time_col, 'TrackID'])
        .size()
        .rename('cnt')
    )

    # 2. 标记每行是否含重复 TrackID
    def _has_duplicate(row):
        cnt_a = counts.get((row[time_col], row[trackid_a_col]), 0)
        cnt_b = counts.get((row[time_col], row[trackid_b_col]), 0)
        return cnt_a > 1 or cnt_b > 1

    events_df = events_df.copy()
    events_df['dup_flag'] = events_df.apply(_has_duplicate, axis=1)

    # 3. 保留无重复行
    result = events_df[~events_df['dup_flag']].drop(columns='dup_flag')

    # 4. 统计
    original = len(events_df)
    retained = len(result)
    print(f"原始行数: {original} → 去重后行数: {retained} "
          f"(保留率 {retained/original*100:.1f}%)")

    return result.reset_index(drop=True)

def analyze_collision_patterns(events_df: pd.DataFrame,
                               time_col: str = 'Time',
                               trackid_a_col: str = 'TrackID_A',
                               trackid_b_col: str = 'TrackID_B') -> dict:
    """
    分析碰撞模式，统计多细胞碰撞的情况。
    
    返回:
    包含碰撞模式统计信息的字典
    """
    if events_df.empty:
        return {}
    
    analysis = {}
    
    # 按时间分组分析
    time_analysis = {}
    for time, time_group in events_df.groupby(time_col):
        # 统计每个细胞在该时刻参与的碰撞次数
        cell_collision_count = {}
        
        for _, row in time_group.iterrows():
            cell_a = row[trackid_a_col]
            cell_b = row[trackid_b_col]
            
            cell_collision_count[cell_a] = cell_collision_count.get(cell_a, 0) + 1
            cell_collision_count[cell_b] = cell_collision_count.get(cell_b, 0) + 1
        
        # 统计碰撞模式
        single_collision_cells = sum(1 for count in cell_collision_count.values() if count == 1)
        multi_collision_cells = sum(1 for count in cell_collision_count.values() if count > 1)
        
        time_analysis[time] = {
            'total_cells': len(cell_collision_count),
            'single_collision_cells': single_collision_cells,
            'multi_collision_cells': multi_collision_cells,
            'collision_events': len(time_group)
        }
    
    analysis['time_analysis'] = time_analysis
    
    # 总体统计
    total_events = len(events_df)
    total_cells = set()
    for _, row in events_df.iterrows():
        total_cells.add(row[trackid_a_col])
        total_cells.add(row[trackid_b_col])
    
    analysis['summary'] = {
        'total_events': total_events,
        'total_unique_cells': len(total_cells),
        'total_time_points': len(time_analysis)
    }
    
    return analysis



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM全参数预测与不同cluster碰撞检测')
    parser.add_argument('--input_csv', type=str, required=True, help='原始df的CSV路径，索引为TrackID')
    parser.add_argument('--index_col', type=int, default=0, help='读取CSV时作为索引的列号，默认0')
    parser.add_argument('--time_col', type=str, default='Time', help='时间列名，默认Time')
    parser.add_argument('--cluster_col', type=str, default='cluster', help='真实标签列名，默认cluster')
    parser.add_argument('--distance_threshold', type=float, default=5.0, help='固定碰撞距离阈值，默认5.0')
    parser.add_argument('--use_dynamic_threshold', action='store_true', help='使用基于有效半径的碰撞检测')
    parser.add_argument('--save_radius_data', action='store_true', help='是否保存有效半径数据')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 读取数据
    print(f'Reading data from: {args.input_csv}')
    df = pd.read_csv(args.input_csv, index_col=args.index_col)

    # 2) 碰撞检测
    print('Detecting collisions of different clusters...')
    if args.use_dynamic_threshold:
        print('Using radius-based collision detection...')
        # 计算有效半径
        df_with_radius = calculate_cell_effective_radius(df, time_col=args.time_col)
        
        # 保存有效半径数据（可选）
        if args.save_radius_data:
            radius_path = os.path.join(args.output_dir, 'effective_radius_data.csv')
            df_with_radius.to_csv(radius_path)
            print(f'Effective radius data saved to {radius_path}')
            
            # 保存半径统计信息
            radius_stats = df_with_radius['Effective_Radius'].describe()
            stats_path = os.path.join(args.output_dir, 'radius_statistics.txt')
            with open(stats_path, 'w') as f:
                f.write("Effective Radius Statistics:\n")
                f.write(str(radius_stats))
            print(f'Radius statistics saved to {stats_path}')
        
        events = detect_cluster_collisions_radius_based(df_with_radius,
                                                       time_col=args.time_col,
                                                       cluster_col=args.cluster_col)
    else:
        print('Using fixed threshold method...')
        events = detect_cluster_collisions(df,
                                           time_col=args.time_col,
                                           cluster_col=args.cluster_col,
                                           distance_threshold=args.distance_threshold)
    
    events_path = os.path.join(args.output_dir, 'collision_events.csv')
    events.to_csv(events_path, index=False)
    print(f'Found {len(events)} events. Saved to {events_path}')

    if events.empty:
        print('No collision events found. Exit.')
        raise SystemExit(0)




workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy4  = os.path.join(workpy, '碰撞检测与预测')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]


for N in range(len(csv_files)):
    use_dynamic_threshold=True
    save_radius_data=True

    filename  = csv_files[N]          # 取第二个文件（可按需改）
    df_path   = os.path.join(directory, filename)
    save_dir = os.path.join(workpy4, filename)
    os.makedirs(save_dir, exist_ok=True) 

    df = pd.read_csv(df_path, index_col=0)
    # --- 缺失值处理：用 0 填充 ---
    df = df.fillna(0)
    # 2. 提取特征、标签、时间列（按需修改）
    # 假设除了 'Time'、'Cell' 外其余都是数值特征
    feature_cols = [c for c in df.columns if c not in ['Time', 'Cell','cluster']]
    data = df[feature_cols]
    data.index=data['TrackID']
    df.index=df['TrackID']
    
    # 2) 碰撞检测
    print('Detecting collisions of different clusters...')
    if use_dynamic_threshold==True:
        print('Using radius-based collision detection...')
        # 计算有效半径
        df_with_radius = calculate_cell_effective_radius(df, time_col='Time')
        
        # 保存有效半径数据（可选）
        if save_radius_data==True:
            radius_path=os.path.join(save_dir, '细胞碰撞半径模拟.csv')
            df_with_radius.to_csv(radius_path)
            print(f'Effective radius data saved to {radius_path}')
            
            # 保存半径统计信息
            radius_stats = df_with_radius['Effective_Radius'].describe()
            stats_path =os.path.join(save_dir, '细胞碰撞半径模拟统计信息.csv')
            with open(stats_path, 'w') as f:
                f.write("Effective Radius Statistics:\n")
                f.write(str(radius_stats))
            print(f'Radius statistics saved to {stats_path}')
        
        events = detect_cluster_collisions_radius_based(df_with_radius,
                                                       time_col='Time',
                                                       cluster_col='cluster')
    else:
        print('Using fixed threshold method...')
        events = detect_cluster_collisions(df,
                                           time_col='Time',
                                           cluster_col='cluster',
                                           distance_threshold=args.distance_threshold)
    
    events_path = os.path.join(save_dir, 'events.csv')
    events.to_csv(events_path, index=False)
    print(f'Found {len(events)} events. Saved to {events_path}')

    if events.empty:
        print('No collision events found. Exit.')
        raise SystemExit(0)
    
    unique_events=filter_unique_cell_collisions(events)
    unique_events_path = os.path.join(save_dir, 'unique_events.csv')
    unique_events.to_csv(unique_events_path, index=False)

    u=analyze_collision_patterns(events)
    U=pd.DataFrame.from_dict(u['time_analysis'], orient='index')
    U = U.reset_index().rename(columns={'index': 'Time'})
    unique_events_path = os.path.join(save_dir, 'events_statistic.csv')
    U.to_csv(unique_events_path, index=False)

    u=analyze_collision_patterns(unique_events)
    U=pd.DataFrame.from_dict(u['time_analysis'], orient='index')
    U = U.reset_index().rename(columns={'index': 'Time'})
    unique_events_path = os.path.join(save_dir, 'unique_events_statistic.csv')
    U.to_csv(unique_events_path, index=False)


'''
filename  = csv_files[N]          # 取第二个文件（可按需改）
save_dir = os.path.join(workpy4, filename)
events_path = os.path.join(save_dir, 'events.csv')
events=pd.read_csv(events_path, index_col=None)
unique_events=filter_unique_cell_collisions(events)
unique_events_path = os.path.join(save_dir, 'unique_events.csv')
unique_events.to_csv(unique_events_path, index=False)
'''
