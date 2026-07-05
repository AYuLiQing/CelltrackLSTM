

'''******************************************************************************************'''
##################################################检查LSTM、EMD效果
workpy0 = r'H:\Work\海南大学课题-细胞动态处理与预测\数据\iMaris提取参数数据'
workpy   = os.path.join(workpy0, '整理')
workpy2=workpy+'\多参数选择'
workpy7  = os.path.join(workpy, '预测结果整理图片')
directory0 = os.path.join(workpy, 'cell')
directory  = os.path.join(directory0, '多通道附标签合并', 'daynamicPara')
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy7, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]

param=df.columns#.drop(['Time'])
param=param.drop('cluster')
IDs=df.index.drop_duplicates()
ID=IDs[0]
touchcellpoint=df.loc[ID].sort_values(by='Time')
touchcellpoint=touchcellpoint.drop(columns='cluster')
#t=touchcellpoint['Time'][-5]
touchcellpoint_pre=touchcellpoint.iloc[:-5,:]
#for parN in (1,4,5,8,2,3,6):
    parN=5###################选择参数
    data=touchcellpoint.iloc[:,parN]
    IMFdf=EMD_Right(touchcellpoint_pre,parN)
    predicted_df=LSTM_right(IMFdf,3,5)
    predicted_reconstructed_data=EMD_recue(predicted_df)
    ###连接前面的原数据和后面的预测数据
    IMFdfnew=pd.concat([IMFdf,predicted_df],ignore_index=True)
    reconstructed_data=EMD_recue(IMFdfnew)


    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_df.iloc[:, 0], label='Predicted Value', color='red')  # 假设我们只展示第一个参数
    plt.title('Parameter 1')
    plt.legend()
    plt.show()
    # 可视化预测结果和真实值
    plt.figure(figsize=(10, 6))
    plt.plot(pd.array(touchcellpoint.iloc[:,parN]), label='True Value', color='blue')  # 真实曲线，假设展示第一个参数
    plt.plot(reconstructed_data, label='Predicted Value', color='red', linestyle='--')  # 预测曲线，假设展示第一个参数
    plt.title('Parameter 1 Comparison')
    plt.legend()
    plt.show()






'''************************************************************************************'''
############################################lstm和lstm-数据分解（三种）方法预测效果比较
# 1. 全局美化风格
sns.set_style("whitegrid")               # 带网格的清爽背景
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.1
plt.rcParams["ytick.major.width"] = 1.1
plt.rcParams["xtick.direction"] = "in"   # 刻度朝内
plt.rcParams["ytick.direction"] = "in"

# 2. 颜色方案（红-蓝互补，色盲友好）
color_true = "#1f77b4"   # 深蓝
color_pred = "#d62728"   # 朱红
######################################################################### 可视化
parN=5
###############################检查lstm预测与真实数值差异：
lstmdf=LSTM_right(touchcellpoint_pre,3,5)
lstmdf=pd.concat([touchcellpoint_pre, lstmdf], axis=0)
'''
# 可视化预测结果和真实值
plt.figure(figsize=(10, 6))
plt.plot(pd.array(touchcellpoint.iloc[:,parN]), label='True Value', color='blue')  # 真实曲线，假设展示第一个参数
plt.plot(pd.array(lstmdf.iloc[:,parN]), label='Predicted Value', color='red', linestyle='--')  # 预测曲线，假设展示第一个参数
plt.title('Parameter 1 Comparison')
plt.legend()
plt.show()
'''
###########美化
# 3. 绘图
fig, ax = plt.subplots(figsize=(10, 5))
# 真实值
ax.plot(pd.array(touchcellpoint.iloc[:,parN]),
        label="True Value",
        color=color_true,
        linewidth=2)
# 预测值
ax.plot(pd.array(lstmdf.iloc[:,parN]),
        label="Predicted Value",
        color=color_pred,
        linewidth=2,
        linestyle="--",
        dashes=(5, 4))      # 自定义虚线密度
# 4. 细节装饰
ax.set_title(f" {param[parN]}  Comparison", fontsize=14, weight="bold")
ax.set_xlabel("Time Step", fontsize=13)
ax.set_ylabel("Value", fontsize=13)
ax.legend(frameon=True, fancybox=True, shadow=True, loc="best")
ax.grid(alpha=0.3)
# 5. 保存为 PDF（放在 plt.show() 之前）
fig.savefig(os.path.join(save_dir, 'LSTM预测效果', ID+param[parN]+'预测效果展示-美化.pdf'), format='pdf', bbox_inches='tight')
plt.show()

################重复十次： 
'''
plt.figure(figsize=(10, 6))
# 1. 真实值
plt.plot(pd.array(touchcellpoint.iloc[:, parN]),
         label='True Value', color='black', linewidth=1.5)
# 2. 收集 10 次预测，用于求均值
pred_list = []
cmap = plt.cm.tab10
for i in range(10):
    lstmdf = LSTM_right(touchcellpoint_pre, 3, 5)
    lstmdf = pd.concat([touchcellpoint_pre, lstmdf], axis=0)
    pred_curve = lstmdf.iloc[:, parN].values      # 取出 1-D array
    pred_list.append(pred_curve)
    # 单次虚线
    plt.plot(pred_curve,
             linestyle='--', color=cmap(i / 9),
             label=f'Pred {i+1}')
# 3. 十次均值（红色实线，加粗）
mean_pred = np.mean(pred_list, axis=0)
plt.plot(mean_pred,
         label='Mean of 10 preds', color='red',
         linewidth=2.5)
plt.title(f'{param[parN]} Comparison (10 predictions + mean)')
plt.legend(b out()
plt.savefig(os.path.join(save_dir, 'LSTM预测效果','随机五十次 进行LSTM预测10X预测.pdf'), 
            dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
plt.show()
'''
###########美化
# ================= 统一风格 =================
sns.set_style("whitegrid")                 # 清爽网格
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# 色盲友好配色
color_true  = "#1f77b4"      # 深蓝（与单张图一致）
color_mean  = "#d62728"      # 朱红（均值突出）
color_single = plt.cm.tab10  # 10 条预测虚线

# ================= 10 次重复预测可视化 =================
# ---------------- 开始绘图 -----------------
fig, ax = plt.subplots(figsize=(18, 6))
# 1. 真实值
ax.plot(pd.array(touchcellpoint.iloc[:, parN]),
        label="True Value", color=true_color,
        linewidth=2, zorder=10)

# 2. 10 次预测 & 均值
pred_list = []
for i in range(10):
    lstmdf = LSTM_right(touchcellpoint_pre, 3, 5)
    lstmdf = pd.concat([touchcellpoint_pre, lstmdf], axis=0)
    pred_curve = lstmdf.iloc[:, parN].values
    pred_list.append(pred_curve)
    # 单次预测：细线 + 半透明
    ax.plot(pred_curve, linestyle='--', linewidth=0.8,
            color=cmap(i), alpha=0.7, label=f'Pred {i+1}')

# 3. 均值 & 标准差阴影
mean_pred = np.mean(pred_list, axis=0)
std_pred  = np.std (pred_list, axis=0)
ax.plot(mean_pred, color=mean_color,
        linewidth=3, label='Mean of 10 preds', zorder=9)
ax.fill_between(range(len(mean_pred)),
                mean_pred - std_pred,
                mean_pred + std_pred,
                color=mean_color, alpha=0.15, label='±1σ')

# 4. 标题 & 轴
ax.set_title(f'{param[parN]}  Comparison  (10 predictions + mean)',
             fontsize=14, weight='bold')
ax.set_xlabel('Time Step', fontsize=13)
ax.set_ylabel('Value', fontsize=13)

# 5. 图例放外部，不遮挡曲线
ax.legend(frameon=True, fancybox=True, shadow=True,
          bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()

# 5. 保存为 PDF（放在 plt.show() 之前）
fig.savefig(os.path.join(save_dir, 'LSTM预测效果', ID+param[parN]+'预测效果展示10XLSTM-美化.pdf'), format='pdf', bbox_inches='tight')
plt.show()





##########################LSTM+数据分解
METHODS = ['EMD', 'CEEMDAN','TVFEMD']  #####数据分解还原通过的三个方法
m=METHODS[0]
decomp_fun, recon_fun = method_dict[m]
###############################检查lstm-EMD预测与真实数值差异：
#data=touchcellpoint.iloc[:,parN]
IMFdf=decomp_fun(touchcellpoint_pre,parN)
predicted_df=LSTM_right(IMFdf,3,5)
predicted_reconstructed_data=EMD_recue(predicted_df)
'''
###连接前面的原数据和后面的预测数据
IMFdfnew=pd.concat([IMFdf,predicted_df],ignore_index=True)
reconstructed_data=recon_fun(IMFdfnew)
# 可视化预测结果和真实值
plt.figure(figsize=(10, 6))
plt.plot(pd.array(touchcellpoint.iloc[:,parN]), label='True Value', color='blue')  # 真实曲线，假设展示第一个参数
plt.plot(reconstructed_data, label='Predicted Value', color='red', linestyle='--')  # 预测曲线，假设展示第一个参数
plt.title('Parameter 1 Comparison')
plt.legend()
plt.show()
'''
###########美化
# 3. 绘图
fig, ax = plt.subplots(figsize=(10, 5))
# 真实值
ax.plot(pd.array(touchcellpoint.iloc[:,parN]),
        label="True Value",
        color=color_true,
        linewidth=2)
# 预测值
ax.plot(pd.array(reconstructed_data),
        label="Predicted Value",
        color=color_pred,
        linewidth=2,
        linestyle="--",
        dashes=(5, 4))      # 自定义虚线密度
# 4. 细节装饰
ax.set_title(f" {param[parN]}  Comparison", fontsize=14, weight="bold")
ax.set_xlabel("Time Step", fontsize=13)
ax.set_ylabel("Value", fontsize=13)
ax.legend(frameon=True, fancybox=True, shadow=True, loc="best")
ax.grid(alpha=0.3)
# 5. 保存为 PDF（放在 plt.show() 之前）
fig.savefig(os.path.join(save_dir, 'LSTM-EMD预测效果', ID+param[parN]+'预测效果展示-美化.pdf'), format='pdf', bbox_inches='tight')
plt.show()
################重复十次：
'''
# 1. 真实值（黑色实线）
plt.plot(pd.array(touchcellpoint.iloc[:, parN]),
         label='True Value', color='black', linewidth=1.5)
# 2. 收集 10 次重构序列
recon_list = []
cmap = plt.cm.tab10
for i in range(10):
    IMFdf = decomp_fun(touchcellpoint_pre, parN)
    predicted_df = LSTM_right(IMFdf, 3, 5)
    predicted_reconstructed_data = recon_fun(predicted_df)   # 仅预测段重构
    # 把“历史+预测”拼起来再整体重构
    IMFdfnew = pd.concat([IMFdf, predicted_df], ignore_index=True)
    reconstructed_data = recon_fun(IMFdfnew)               # 完整重构序列
    recon_list.append(reconstructed_data)           # 存为 array
    # 单次虚线
    plt.plot(reconstructed_data,
             linestyle='--', color=cmap(i / 9),
             label=f'Pred {i+1}')
# 3. 十次均值（红色实线，加粗）
mean_recon = np.mean(recon_list, axis=0)
plt.plot(mean_recon,
         label='Mean of 10 preds', color='red',
         linewidth=2.5)
plt.title(f'Parameter {parN+1} Comparison (10 predictions + mean)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
'''
####美化：
plt.figure(figsize=(18, 6))
# 1. 真实值（深黑，粗线，置顶层）
plt.plot(pd.array(touchcellpoint.iloc[:, parN]),
         label='True Value', color='#222222', linewidth=2.2, zorder=10)
# 2. 10 次重构序列（细虚线 + 半透明彩带）
recon_list = []
cmap = plt.cm.tab10
for i in range(10):
    IMFdf = decomp_fun(touchcellpoint_pre, parN)
    predicted_df = LSTM_right(IMFdf, 3, 5)
    predicted_reconstructed_data = recon_fun(predicted_df)
    IMFdfnew = pd.concat([IMFdf, predicted_df], ignore_index=True)
    reconstructed_data = recon_fun(IMFdfnew)
    recon_list.append(reconstructed_data)
    # 单次预测：细虚线 + 半透明
    plt.plot(reconstructed_data,
             linestyle='--', linewidth=0.8,
             color=cmap(i / 9), alpha=0.7, label=f'Pred {i+1}')
# 3. 均值 & 不确定阴影
mean_recon = np.mean(recon_list, axis=0)
std_recon  = np.std (recon_list, axis=0)
plt.plot(mean_recon,
         label='Mean of 10 preds', color='#D62728',
         linewidth=3, zorder=9)
plt.fill_between(range(len(mean_recon)),
                 mean_recon - std_recon,
                 mean_recon + std_recon,
                 color='#D62728', alpha=0.15, label='±1σ')
# 4. 标题 & 图例
plt.title(f'Parameter {parN+1}  Comparison  (10 reconstructions + mean)',
          fontsize=14, weight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True,
           bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'LSTM-{m}预测效果',ID+param[parN]+'预测效果展示10XLSTM-美化.pdf'), 
            dpi=300, bbox_inches='tight', transparent=False,facecolor='white')
plt.show()



calc_all_metrics(pd.Series(touchcellpoint.iloc[:, parN]),pd.Series(lstmdf.iloc[:, parN]))
calc_all_metrics(pd.Series(touchcellpoint.iloc[:, parN]),pd.array(reconstructed_data))

calc_all_metrics(pd.Series(touchcellpoint.iloc[-5:, parN]),pd.array(lstmdf.iloc[-5:, parN]))
calc_all_metrics(pd.Series(touchcellpoint.iloc[-5:, parN]),pd.array(reconstructed_data[-5:]))


'''***********************************************'''
######################################################################### 数值获取
# 出现次数≥5 的 TrackID
IDs = df.index.value_counts().loc[lambda x: x >= 15].index.to_numpy()
# 从原 IDs 中无放回随机抽取 100 个
IDs = np.random.default_rng().choice(IDs, size=300, replace=False)
param=df.columns
param=param.drop('cluster')

ID=IDs[0]
touchcellpoint=df.loc[ID].sort_values(by='Time')
touchcellpoint=touchcellpoint.drop(columns='cluster')
#t=touchcellpoint['Time'][-5]
touchcellpoint_pre=touchcellpoint.iloc[:-5,:]


from tqdm import tqdm   # 进度条，可选
############################################################LSTM随机五十次评估
all_res = []            # 收集每个 ID 的 DataFrame
for uid in tqdm(IDs, desc='ID progress'):
    # 1. 取单 ID 序列并排序
    touchcellpoint = df.loc[uid].sort_values('Time').drop(columns='cluster')
    # 2. 划分训练 / 预测窗口
    touchcellpoint_pre = touchcellpoint.iloc[:-5, :]   # 训练段
    # 3. 10 次预测并平均
    lstmdf_list = []
    for _ in range(10):
        lstmdf = LSTM_right(touchcellpoint_pre, 3, 5)
        lstmdf_list.append(lstmdf)
    lstmdf_mean = pd.concat(lstmdf_list).groupby(level=0).mean()   # 10 次均值
    lstmdf_mean['Time']=touchcellpoint['Time'][-5:].values
    lstmdf_mean.index=[uid]*lstmdf_mean.shape[0]
    # 4. 拼回「训练 + 预测」
    full_pred = pd.concat([touchcellpoint_pre, lstmdf_mean], axis=0)
    # 5. 把 ID 挂回行索引（MultiIndex：ID + Time）
    full_pred = full_pred.assign(ID=uid).set_index('ID', append=True)
    full_pred = full_pred.swaplevel().sort_index()   # 现在索引 -> (ID, Time)
    all_res.append(full_pred)
# 6. 纵向合并所有 ID
final_df = pd.concat(all_res, axis=0)   # 行索引：(ID, Time)
final_df.index = final_df.index.get_level_values(1)
final_df.to_csv(os.path.join(save_dir, 'LSTM预测效果','随机三百次进行LSTM预测10X预测.csv'))

metrics_df = quantify_diff(df, final_df, last_k=5, cols=['Area'])
metrics_df.to_csv(os.path.join(save_dir, 'LSTM预测效果','随机三百次进行LSTM预测10X预测误差_Area.csv'))




############################################################LSTM-数据分解随机五十次评估
all_res_emd = []            # 收集每个 ID 的 DataFrame
for uid in tqdm(IDs[0:2], desc='ID progress'):
    # 1. 取单 ID 序列并排序
    touchcellpoint = df.loc[uid].sort_values('Time').drop(columns='cluster')
    # 2. 划分训练 / 预测窗口
    touchcellpoint_pre = touchcellpoint.iloc[:-5, :]   # 训练段
    lstmdf_mean = pd.DataFrame(index=touchcellpoint.index[-5:],columns=param)   # 存放各参数最终预测均值
    for parN in range(len(param)):                 # 逐个参数（单列）循环
        # 1. 只取这一列
        ts_raw = touchcellpoint_pre[param[parN]]

        # 2. 10 次预测 → 重构
        recon_list = []
        for i in range(10):
            # 2.1 分解（输入：Series，输出：DataFrame 或你需要的格式）
            IMFdf = decomp_fun(touchcellpoint_pre, parN)      # 仅对单列分解
            # 2.2 LSTM 预测（仅预测段）
            predicted_df = LSTM_right(IMFdf, window_size=3, forecast_horizon=5)  # 返回 DataFrame
            # 2.3 把预测段重构回时域（Series）
            pred_recon = recon_fun(predicted_df)   # 返回 Series（索引=未来 5 期）
            # 2.4 如果需要“历史+预测”整体再重构，可在这里拼接后整体重构
            #     下面示例仅对未来 5 期做平均，如需整段可再调整
            recon_list.append(pred_recon)   # 存为 array（长度=5）

        # 3. 十次平均
        mean_recon = np.mean(recon_list, axis=0)   # shape=(5,)

        # 4. 转成 Series 并放进结果表
        #    假设 pred_recon 的索引就是未来 5 期的日期
        lstmdf_mean[param[parN]] = pd.Series(mean_recon, index=touchcellpoint.index[-5:])
    # 4. 拼回「训练 + 预测」
    full_pred = pd.concat([touchcellpoint_pre, lstmdf_mean], axis=0)
    # 5. 把 ID 挂回行索引（MultiIndex：ID + Time）
    full_pred = full_pred.assign(ID=uid).set_index('ID', append=True)
    full_pred = full_pred.swaplevel().sort_index()   # 现在索引 -> (ID, Time)
    all_res_emd.append(full_pred)
# 6. 纵向合并所有 ID
final_df_emd = pd.concat(all_res_emd, axis=0)   # 行索引：(ID, Time)
final_df_emd.index = final_df_emd.index.get_level_values(1)
metrics_df = quantify_diff(df, final_df_emd, last_k=10, cols=['Area'])


############################################################LSTM-数据分解随机五十次评估---缩减运行时长
METHODS = ['EMD', 'CEEMDAN','TVFEMD']  #####数据分解还原通过的三个方法
m=METHODS[0]
decomp_fun, recon_fun = method_dict[m]
'''
# 出现次数≥5 的 TrackID
IDs = df.index.value_counts().loc[lambda x: x >= 10].index.to_numpy()
# 从原 IDs 中无放回随机抽取 100 个
IDs = np.random.default_rng().choice(IDs, size=300, replace=False)
param=df.columns
param=param.drop('cluster')
'''
all_res_emd = []            # 收集每个 ID 的 DataFrame
for uid in tqdm(IDs[0:50], desc='ID progress'):
    # 1. 取单 ID 序列并排序
    touchcellpoint = df.loc[uid].sort_values('Time').drop(columns='cluster')
    # 2. 划分训练 / 预测窗口
    touchcellpoint_pre = touchcellpoint.iloc[:-5, :]   # 训练段

    # 0. 预分配结果表
    lstmdf_mean = pd.DataFrame(index=touchcellpoint.index[-5:], columns=param)

    # 1. 分解 + 记录每个参数的 IMF 列名 & 数量
    max_imf = 0
    imf_cols = {}              # par -> 该参数在 big_imf_df 里的列切片
    big_imf_list = []          # 存放每个参数的 IMFdf（已补零）

    for par in range(len(param)):
        imf_df = decomp_fun(touchcellpoint_pre,par)   # DataFrame，列=IMF0...
        n_imf = imf_df.shape[1]
        max_imf = max(max_imf, n_imf)
        big_imf_list.append(imf_df)

    # 2. 统一补零 -> 横向拼接
    for i, imf_df in enumerate(big_imf_list):
        lack = max_imf - imf_df.shape[1]
        if lack > 0:
            # 用 0 补齐缺失 IMF
            imf_df = pd.concat([imf_df,
                                pd.DataFrame(0, index=imf_df.index,
                                             columns=[f'IMF_fill{j}' for j in range(lack)])],
                               axis=1)
        big_imf_list[i] = imf_df.add_prefix(f'{param[i]}_')   # 列名加前缀防冲突

    big_imf_df = pd.concat(big_imf_list, axis=1)   # 一张大表：5*sum(IMF)

    # 3. 10 次 LSTM 预测（整张表）
    pred_list = []
    for _ in range(10):
        pred_df = LSTM_right(big_imf_df, window_size=3, forecast_horizon=5)  # 5×总IMF
        pred_list.append(pred_df)

    # 4. 拆分 + 重构 -> 每个参数 10 次结果
    for par in param:
        par_prefix = f'{par}_'
        par_imf_cols = [c for c in big_imf_df.columns if c.startswith(par_prefix)]
        # 提取该参数对应 IMF 预测值（10 次）
        par_pred_list = [p[par_imf_cols] for p in pred_list]   # List[DataFrame]
        # 去掉补零列（填充 IMF）
        par_pred_list = [df.loc[:, ~df.columns.str.contains('IMF_fill')]
                         for df in par_pred_list]

        # 重构回时域
        recon_series = []
        for pardf in par_pred_list:
            ts = recon_fun(pardf)          # 返回 Series（长度=5）
            recon_series.append(ts)

        # 5. 求均值 -> 写回
        mean_ts = np.mean(recon_series, axis=0)
        lstmdf_mean[par] = mean_ts
    lstmdf_mean['Time']=touchcellpoint['Time'][-5:]
    # 4. 拼回「训练 + 预测」
    full_pred = pd.concat([touchcellpoint_pre, lstmdf_mean], axis=0)
    # 5. 把 ID 挂回行索引（MultiIndex：ID + Time）
    full_pred = full_pred.assign(ID=uid).set_index('ID', append=True)
    full_pred = full_pred.swaplevel().sort_index()   # 现在索引 -> (ID, Time)
    all_res_emd.append(full_pred)
# 6. 纵向合并所有 ID
final_df_emd = pd.concat(all_res_emd, axis=0)   # 行索引：(ID, Time)
final_df_emd.index = final_df_emd.index.get_level_values(1)
final_df_emd.to_csv(os.path.join(save_dir, f'LSTM-{m}预测效果','随机五十次进行LSTM-EMD预测10X预测.csv'))

metrics_df_emd = quantify_diff(df, final_df_emd, last_k=10, cols=['Area'])
metrics_df_emd.to_csv(os.path.join(save_dir, f'LSTM-{m}预测效果','随机五十次进行LSTM-EMD预测10X预测误差Area.csv'))
##LSTM-EMD
##LSTM-CEEMDAN
##LSTM-TVFEMD

final_df_emd=pd.read_csv(os.path.join(save_dir, 'LSTM-TVFEMD预测效果','随机五十次进行LSTM-TVFEMD预测10X预测.csv'),
                         index_col=0)
# 1. 收集每个 par 的结果
df_list = []
for par in param:
    metrics_df_emd = quantify_diff(df, final_df_emd, last_k=10, cols=[par])
    df_list.append(metrics_df_emd)
# 2. 沿列方向拼接，用参数名作为第二层列索引
metrics_all = pd.concat(df_list, axis=1, keys=param)
metrics_all.to_csv(os.path.join(save_dir, 'LSTM-TVFEMD预测效果','随机五十次进行LSTM-TVFEMD预测10X预测误差_all.csv'))

##############################可视化
metrics_all=pd.read_csv(os.path.join(save_dir, 'LSTM预测效果','随机五十次进行LSTM预测10X预测误差_all.csv'),index_col=0,header=[0, 1])
metrics_all_emd=pd.read_csv(os.path.join(save_dir, 'LSTM-EMD预测效果','随机五十次进行LSTM-EMD预测10X预测误差_all.csv'),index_col=0,header=[0, 1])
metrics_all_ceemdan=pd.read_csv(os.path.join(save_dir, 'LSTM-CEEMDAN预测效果','随机五十次进行LSTM-CEEMDAN预测10X预测误差_all.csv'),index_col=0,header=[0, 1])
metrics_all_tvfemd=pd.read_csv(os.path.join(save_dir, 'LSTM-TVFEMD预测效果','随机五十次进行LSTM-TVFEMD预测10X预测误差_all.csv'),index_col=0,header=[0, 1])

#r_df = metrics_all.xs('r', level=1, axis=1)   # 所有第二层索引为 'r' 的列

# 3. 只看 Area 的 R2
#['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'MARRE', 'TheilU','MaxError', 'Bias', 'R2_shift', 'r', 'NRMSE']
plot_violin_2d(metrics_all, target_param='Area', target_metric='R2')
plot_box_2d(metrics_all, target_param='Area', target_metric='R2')

metrics_all=metrics_all.drop(columns='Time Index')
metrics_all_emd=metrics_all_emd.drop(columns='Time Index')
metrics_all_ceemdan=metrics_all_ceemdan.drop(columns='Time Index')
metrics_all_tvfemd=metrics_all_tvfemd.drop(columns='Time Index')

plot_box_2d(metrics_all,  target_metric='r')
plot_box_2d(metrics_all_emd,  target_metric='r')
plot_box_2d(metrics_all_ceemdan,  target_metric='r')
plot_box_2d(metrics_all_tvfemd,  target_metric='r')



metrics_all = [drop_nan_inf_cols(df) for df in df_list]
U=['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'MARRE', 'TheilU','MaxError', 'Bias', 'R2_shift', 'r', 'NRMSE']
# ---- 1. 先去掉含 NaN/inf 的顶层列 ----
def drop_nan_inf_cols(df):
    ok = df.groupby(level=0, axis=1).apply(lambda g: np.isfinite(g.values).all())
    return df[ok.index[ok]]

metrics_list = [drop_nan_inf_cols(d.copy()) for d in metrics_list]
# ---- 2. 只保留顶层列索引的交集 ----
common_top = metrics_list[0].columns.get_level_values(0).unique()
for df in metrics_list[1:]:
    common_top = common_top.intersection(df.columns.get_level_values(0).unique())
metrics_list = [df[common_top] for df in metrics_list]
for u in U:
    plot_box_2d_pdfmerge(metrics_list,
                         name=['LSTM', 'LSTM-EMD', 'LSTM-CEEMDAN', 'LSTM-TVFEMD'],
                target_metric=u,
                save_path=os.path.join(save_dir,f'LSTM-分解方法随即五十次10X预测评估_{u}.pdf'))


metric_df=pd.DataFrame()
metric='r'
for par in param:
    metric_df=pd.concat([metric_df,metrics_all_emd[par][metric]],axis=0)

metric_df.columns=['lstm']


##################################################################
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
    

def LSTM_right(data,window_size=30,forecast_horizon=10):
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

from sklearn import metrics
def calc_regs_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae   = metrics.mean_absolute_error(y_true, y_pred)
    mse   = metrics.mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    r2    = metrics.r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    marre = np.mean(np.abs(y_pred - y_true) / (y_true.max() - y_true.min())) * 100
    theil_u = np.sqrt(np.mean((y_pred - y_true)**2)) / (
                np.sqrt(np.mean(y_pred**2)) + np.sqrt(np.mean(y_true**2)))
    max_err = np.max(np.abs(y_pred - y_true))
    bias    = np.mean(y_pred - y_true)

    # 直接返回单行 DataFrame
    return pd.DataFrame({
        'MAE'   : [mae],
        'MSE'   : [mse],
        'RMSE'  : [rmse],
        'R2'    : [r2],
        'MAPE'  : [mape],
        'SMAPE' : [smape],
        'MARRE' : [marre],
        'TheilU': [theil_u],
        'MaxError': [max_err],
        'Bias'  : [bias]
    })
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)

def calc_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    合并版：计算全部 13 个常用指标，返回单行 DataFrame
    """
    y_true, y_pred = y_true.dropna(), y_pred.dropna()
    if y_true.shape != y_pred.shape:
        raise ValueError('长度不一致')

    # --- 基础误差 ---
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    marre = np.mean(np.abs(y_pred - y_true) / (y_true.max() - y_true.min())) * 100
    theil_u = np.sqrt(np.mean((y_pred - y_true)**2)) / (
                np.sqrt(np.mean(y_pred**2)) + np.sqrt(np.mean(y_true**2)))
    max_err = np.max(np.abs(y_pred - y_true))
    bias    = np.mean(y_pred - y_true)

    # --- 对齐/形状指标 ---
    pred_shift = y_pred - y_pred.mean() + y_true.mean()
    r2_shift = r2_score(y_true, pred_shift) 
    r = np.corrcoef(y_true, y_pred)[0, 1]
    nrmse = rmse / y_true.std()

    return pd.DataFrame({
        'MAE'      : [mae],
        'MSE'      : [mse],
        'RMSE'     : [rmse],
        'R2'       : [r2],
        'MAPE'     : [mape],
        'SMAPE'    : [smape],
        'MARRE'    : [marre],
        'TheilU'   : [theil_u],
        'MaxError' : [max_err],
        'Bias'     : [bias],
        'R2_shift' : [r2_shift],
        'r'        : [r],
        'NRMSE'    : [nrmse]
    })

def quantify_diff(df_true: pd.DataFrame,
                  df_pred: pd.DataFrame,
                  last_k: int  = 5,
                  cols: list[str]=None):
    """
    对比 df_true 与 df_pred 中同名 ID 的数据。
    参数
    ----
    last_k : int 或 'all'
        取每个 ID 的最后几条；'all' 表示全部数据。
    cols : list 或 None
        指定要评估的列，None 表示所有数值列。
    返回
    ----
    DataFrame，index=ID，columns=13 个指标（指定列的平均）。
    """
    # 0. 列过滤
    if cols is None:
        cols = df_true.select_dtypes(include=np.number).columns
    else:
        cols = pd.Index(cols).intersection(df_true.columns)

    common_ids = df_true.index.intersection(df_pred.index)
    res = []
    for idx in common_ids:
        # ****** 关键：两表都先按 Time 排序再切尾 ******
        y_true = (df_true.loc[idx]
                         .sort_values('Time')[cols]
                         .tail(last_k))
        y_pred = (df_pred.loc[idx]
                         .sort_values('Time')[cols]
                         .tail(last_k))

        id_metrics = []
        for col in cols:
            id_metrics.append(calc_all_metrics(y_true[col], y_pred[col]))
        id_mean = pd.concat(id_metrics, ignore_index=True).mean(axis=0)
        id_mean.name = idx
        res.append(id_mean)

    return pd.DataFrame(res)





'''****************************'''
####################################多次运行检查预测效果：
def run_one_parN(parN: int,touchcellpoint):
    """跑完一个 parN，返回真实序列 + 重建序列"""
    data = touchcellpoint.iloc[:, parN]
    touchcellpoint_pre=touchcellpoint.iloc[:-5,:]
    IMFdf = EMD_Right(touchcellpoint_pre, parN)
    predicted_df = LSTM_right(IMFdf, 3, 5)
    predicted_reconstructed_data = EMD_recue(predicted_df)

    # 拼接
    IMFdfnew = pd.concat([IMFdf, predicted_df], ignore_index=True)
    reconstructed_data = EMD_recue(IMFdfnew)

    return data, reconstructed_data
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import seaborn as sns
def calc_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    计算 6 个指标，返回 dict
    1. R2          : 原始 R²（可能负）
    2. R2_shift    : 均值对齐后 R²（≥0）
    3. r           : 皮尔逊相关系数（-1~1）
    4. MAPE        : 平均绝对百分误差 (%)
    5. RMSE        : 均方根误差
    6. NRMSE       : RMSE / y_true.std()
    """
    y_true, y_pred = y_true.dropna(), y_pred.dropna()
    if y_true.shape != y_pred.shape:
        raise ValueError('长度不一致')

    # 1. 原始 R²
    r2_orig = r2_score(y_true, y_pred)

    # 2. 均值-对齐 R²（消除整体偏移）
    pred_shift = y_pred - y_pred.mean() + y_true.mean()
    r2_shift = r2_score(y_true, pred_shift)

    # 3. 相关系数（只关心形状）
    r = np.corrcoef(y_true, y_pred)[0, 1]

    # 4~6. 误差指标
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nrmse = rmse / y_true.std()

    return {
        'R2'       : r2_orig,
        'R2_shift' : r2_shift,
        'r'        : r,
        'MAPE'     : mape,
        'RMSE'     : rmse,
        'NRMSE'    : nrmse
    }

###################################################不同参数展示or
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
true_c, pred_c = '#1f77b4', '#d62728'
for i in range(9):
    true, pred = run_one_parN(i,touchcellpoint=touchcellpoint)
    m = calc_metrics(true[-5:], pd.Series(pred)[-5:])
    m['Param'] = f'P{i+1}'
    metrics_df = pd.concat([metrics_df, pd.Series(m).to_frame().T], ignore_index=True)
    ax = axes[i]
    ax.plot(pd.array(true), label='True', color=true_c, lw=2.5)
    ax.plot(pred, label='Pred', color=pred_c, lw=2.5, ls='--', dashes=(5, 3))
    ax.set_title(f'Parameter {i+1}', fontsize=14, weight='bold')
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('IMF-LSTM Prediction vs True', fontsize=18, weight='bold', y=0.98)
plt.tight_layout()
plt.show()

#####################################################同参数多次运行or
# 1. 单参数对比图美化
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
# 颜色统一
true_c, pred_c = '#1f77b4', '#d62728'
for _ in range(9):
    true, pred = run_one_parN(parN,touchcellpoint=touchcellpoint)
    m = calc_metrics(true[-5:], pd.Series(pred)[-5:])
    m['Param'] = f'P{i+1}'
    metrics_df = pd.concat([metrics_df, pd.Series(m).to_frame().T], ignore_index=True)
    ax = axes[_]
    ax.plot(pd.array(true), label='True', color=true_c, lw=2.5)
    ax.plot(pred, label='Pred', color=pred_c, lw=2.5, ls='--', dashes=(5, 3))
    ax.set_title(f'Parameter {param[parN]}', fontsize=14, weight='bold')
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('IMF-LSTM Prediction vs True', fontsize=20, weight='bold', y=0.98)
plt.tight_layout()
plt.show()

#######################################可视化预测指标
# 2. 箱线图美化
metrics_long = metrics_df.melt(id_vars='Param', var_name='Metric', value_name='Value')
plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=metrics_long, x='Metric', y='Value', palette='Set2', width=0.6)
sns.stripplot(data=metrics_long, x='Metric', y='Value', color='black', size=4, alpha=0.7)
plt.title('Prediction Error Distribution across 9 Parameters', fontsize=16, weight='bold', pad=20)
plt.ylabel('Score', fontsize=13)
plt.xlabel('')
plt.grid(axis='y', alpha=0.3)
sns.despine(trim=True)
plt.tight_layout()
plt.show()

metrics_df['MAPE'] 




'''****************************'''
##########################################比较LSTM和LSTM+EMD的预测效果：
N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy7, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]


os.makedirs(os.path.join(save_dir, 'LSTM预测效果'), exist_ok=True) 
os.makedirs(os.path.join(save_dir, 'LSTM-EMD预测效果'), exist_ok=True) 
# 出现次数≥5 的 TrackID
IDs = df.index.value_counts().loc[lambda x: x >= 10].index.to_numpy()
# 从原 IDs 中无放回随机抽取 100 个
IDs = np.random.default_rng().choice(IDs, size=10, replace=False)
param=df.columns
param=param.drop('cluster')

for ID in IDs:
    touchcellpoint=df.loc[ID].sort_values(by='Time')
    touchcellpoint=touchcellpoint.drop(columns='cluster')
    touchcellpoint_pre=touchcellpoint.iloc[:-5,:]

    lstmdf=LSTM_right(touchcellpoint,3,5)
    lstmdf=pd.concat([touchcellpoint_pre, lstmdf], axis=0)
    lstmdf.to_csv(os.path.join(save_dir, 'LSTM预测效果',ID+'进行LSTM预测.csv'))
    calc_regs_metrics(lstmdf.iloc[-5:,:], touchcellpoint.iloc[-5:,:])

    lstmdf_value=pd.DataFrame()  
    lstmemd_value=pd.DataFrame()  
    lstmemd_preds = pd.DataFrame()          # 最终汇总表
    #parN=5###################选择参数
    for parN in range(0,len(param)):
        IMFdf=EMD_Right(touchcellpoint_pre,parN)
        if IMFdf.shape[1]==0 :
            continue
        predicted_df=LSTM_right(IMFdf,3,5)
        predicted_reconstructed_data=EMD_recue(predicted_df)
        ###连接前面的原数据和后面的预测数据
        IMFdfnew=pd.concat([IMFdf,predicted_df],ignore_index=True)
        reconstructed_data=EMD_recue(IMFdfnew)
        pred_series = pd.Series(reconstructed_data)
        pred_series.name = param[parN]                  # 列名 = 当前参数值
        lstmemd_preds = pd.concat([lstmemd_preds, pred_series], axis=1)

        u=calc_regs_metrics(lstmdf.iloc[-5:,parN], touchcellpoint.iloc[-5:,parN])
        u.index=[param[parN]]
        lstmdf_value = pd.concat([lstmdf_value, u], axis=0)

        U=calc_regs_metrics(pred_series[-5:], touchcellpoint.iloc[-5:,parN])
        U.index=[param[parN]]
        lstmemd_value=pd.concat([lstmemd_value, U], axis=0)
    lstmemd_preds.to_csv(os.path.join(save_dir, 'LSTM-EMD预测效果',ID+'进行LSTM-EMD预测.csv'))
    
    lstmdf_value.to_csv(os.path.join(save_dir, 'LSTM预测效果',ID+'进行LSTM预测评估.csv'))
    lstmemd_value.to_csv(os.path.join(save_dir, 'LSTM-EMD预测效果',ID+'进行LSTM-EMD预测评估.csv'))


#######可视化误差水平和分布





'''****************************'''
################## ########################展示原始数据分解为EMDimf的过程：
def plot_imf_decomposition(IMFdf, sel_idx, time_ax=None, figsize=(14, 10),save_path=None):
    """
    展示 IMF 分解：原始、剩余合并、被选 IMF 单独
    仅对可视化风格进行美化，其余逻辑不变
    """
    # ---------- 以下代码与原函数完全一致 ----------
    if isinstance(sel_idx, int):
        sel_idx = [sel_idx]
    sel_idx = list(sel_idx)

    all_imf_sum = IMFdf.sum(axis=1)
    sel_imf_sum = IMFdf.iloc[:, sel_idx].sum(axis=1)
    residual = all_imf_sum - sel_imf_sum

    if time_ax is None:
        time_ax = IMFdf.index
    else:
        time_ax = pd.array(time_ax)

    n_sel = len(sel_idx)
    total = 2 + n_sel

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(total, 1, hspace=0.12)  # 加大间隔，避免标题重叠

    # 颜色 & 线型
    orig_c, res_c, imf_c = '#1f77b4', '#ff7f0e', '#2ca02c'

    # 1. 原始 & 剩余 合并图
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_ax, all_imf_sum, color=orig_c, lw=2.5, label='Original')
    ax1.plot(time_ax, residual, color=res_c, lw=2.5, ls='--', dashes=(5, 3), label='Residual')
    ax1.set_title('Original vs Residual', fontsize=14, weight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)
    sns.despine(ax=ax1)

    # 2. 被选 IMF 各自
    for i, imf_no in enumerate(sel_idx, start=1):
        axi = fig.add_subplot(gs[i], sharex=ax1)
        axi.plot(time_ax, IMFdf.iloc[:, imf_no], color=imf_c, lw=2.5)
        axi.set_title(f'IMF {imf_no + 1}', fontsize=14, weight='bold')
        axi.set_ylabel('Amplitude', fontsize=12)
        axi.grid(alpha=0.3)
        sns.despine(ax=axi)
        if i == total - 1:  # 最底图显示横轴
            axi.set_xlabel('Time', fontsize=12)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    plt.show()


N=0
filename  = csv_files[N]          # 取第二个文件（可按需改）
df_path   = os.path.join(directory, filename)
save_dir = os.path.join(workpy7, filename)
os.makedirs(save_dir, exist_ok=True) 
df = pd.read_csv(df_path, index_col=0)
U=inspect_nan(df,'zero')
df=U[0]

param=df.columns#.drop(['Time'])
IDs=df.index.drop_duplicates()
ID=IDs[10]
touchcellpoint=df.loc[ID].sort_values(by='Time')
#t=touchcellpoint['Time'][-5]
touchcellpoint_pre=touchcellpoint.iloc[:-5,:]
parN=int(np.where(param == 'Chemotaxis Index')[0])
IMFdf=EMD_Right(touchcellpoint_pre,parN)
plot_imf_decomposition(IMFdf, sel_idx=[0],save_path=os.path.join(save_dir, ID+'-'+param[parN]+'EMD分解展示_1imf.pdf'))
plot_imf_decomposition(IMFdf, IMFdf.columns,save_path=os.path.join(save_dir, ID+'-'+param[parN]+'EMD分解展示_all.pdf'))

# 预分配空表统计所有ID参数的imf数量
result_df = pd.DataFrame(index=IDs, columns=param, dtype=int)
# 2. 新增：还原质量表（3 指标）
quality_df = pd.DataFrame(index=pd.MultiIndex.from_product([IDs, param],
                                                           names=['ID', 'param']),
                          columns=['MSE', 'NMSE', 'R2'],
                          dtype=float)

# --------------------------------------------------
# 3. 主循环
for i, ID in enumerate(IDs):
    for j, par in enumerate(param):
        touchcellpoint = df.loc[ID].sort_values(by='Time')

        # 3.1 找列号并转数值
        parN = int(np.where(param == par)[0])
        data_ser = pd.to_numeric(touchcellpoint.iloc[:, parN], errors='coerce').dropna()

        # 3.2 长度不足
        if data_ser.size < 3:
            result_df.loc[ID, par] = 0
            quality_df.loc[(ID, par), :] = np.nan
            continue

        # 3.3 EMD 分解
        IMFdf = EMD_Right(pd.DataFrame(data_ser), 0)      # 分解
        imf_count = IMFdf.shape[1]
        result_df.loc[ID, par] = imf_count

        # 3.4 还原
        new_df = EMD_recue(IMFdf)                         # 还原
        reconstructed = new_df   # 假设还原后只有 1 列
        original = data_ser.values

        # 3.5 计算指标
        mse  = mean_squared_error(original, reconstructed)
        nmse = mse / np.var(original) if np.var(original) != 0 else np.nan
        r2   = r2_score(original, reconstructed)

        quality_df.loc[(ID, par), ['MSE', 'NMSE', 'R2']] = [mse, nmse, r2]

from scipy import stats
# 1. 清洗：去掉全 0 的行与列
cleaned = result_df.loc[(result_df != 0).any(axis=1), (result_df != 0).any(axis=0)]
# 2. 描述统计（含众数）
desc = cleaned.describe().T  # 基础统计：count/mean/std/min/25%/50%/75%/max

result_df.to_csv(os.path.join(save_dir, 'EMD分解数量.csv'))
quality_df.to_csv(os.path.join(save_dir, 'EMD质量分数_all.csv'))

###################################提琴图对比
# 1. 长表化（一列值 + 一列 cluster）
df_long = cleaned.stack().reset_index(name='IMF_n')
df_long['cluster'] = df_long['TrackID'].map(id_cluster_map)
# 2. 去全零（保险）
df_plot = df_long[df_long['IMF_n'] > 0]
# 3. 绘图
plt.figure(figsize=(8, 6))
palette = {'green': '#2E8B57', 'red': '#DC143C'}
sns.violinplot(data=df_plot, x='cluster', y='IMF_n',
               palette=palette, inner='quartile', saturation=.9)
# 3.1 箱线图（替换 violinplot）
#sns.boxplot(data=df_plot,x='cluster',y='IMF_n',
#            palette=palette,saturation=.9,width=0.5)         
sns.stripplot(data=df_plot, x='cluster', y='IMF_n',
              palette=palette, size=2.5, alpha=.7, dodge=False)
# 4. 美化
plt.title('IMF Component Count Distribution by Cluster', fontsize=16, weight='bold')
plt.xlabel('')
plt.ylabel('IMF Component Count', fontsize=12)
plt.xticks([0, 1], ['Green', 'Red'])
sns.despine(left=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'EMD分解数量对比.pdf'),
             format='pdf', bbox_inches='tight')
plt.show()

########################################热图对比
# 1. 去重映射
id_cluster_map = df.drop_duplicates(subset='ID')[['ID', 'cluster']].set_index('ID')['cluster']
# 1. 锁定颜色标尺（防止两张图尺度不同）
vmin, vmax = cleaned.min().min(), cleaned.max().max()
# 2. 按 cluster 拆表
green_ids = cleaned.index[cleaned.index.map(id_cluster_map) == 'green']
red_ids   = cleaned.index[cleaned.index.map(id_cluster_map) == 'red']
df_green = cleaned.loc[green_ids]
df_red   = cleaned.loc[red_ids]
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})   # 去掉格线更干净
plt.rcParams['font.size'] = 11
# ---------- 画布 ----------
fig = plt.figure(figsize=(8, 9))                   # 高瘦型
gs = fig.add_gridspec(2, 1,
                       height_ratios=[1, 1],
                       left=0.12, right=0.86,
                       bottom=0.08, top=0.93,
                       hspace=0.03)                # 上下紧贴

ax1 = fig.add_subplot(gs[0])   # 上方 Green
ax2 = fig.add_subplot(gs[1])   # 下方 Red
cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5])  # 右侧 colorbar
# ---------- 上方热图（Green） ----------
sns.heatmap(df_green,
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            cbar=False,
            ax=ax1)
ax1.set_title('Green Cluster', fontsize=14, weight='bold', pad=6)
ax1.set_xlabel('')
ax1.set_ylabel('Track ID', fontsize=12)
ax1.tick_params(axis='x',           # 隐藏横轴刻度+标签
                which='both',
                bottom=False,
                labelbottom=False)
# ---------- 下方热图（Red） ----------
sns.heatmap(df_red,
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            cbar=True,
            cbar_ax=cbar_ax,
            ax=ax2)
ax2.set_title('Red Cluster', fontsize=14, weight='bold', pad=6)
ax2.set_xlabel('Time (or Feature)', fontsize=12)   # 按需改名
ax2.set_ylabel('Track ID', fontsize=12)
# ---------- 统一 colorbar 标签 ----------
cbar_ax.set_ylabel('Intensity', rotation=270, va='bottom', fontsize=12)
# ---------- 全局标题 ----------
fig.suptitle('Intensity Heatmap by Cluster', fontsize=16, weight='bold', y=0.97)
plt.savefig(os.path.join(save_dir,'EMD分解数量对比热图.pdf'),
             format='pdf', bbox_inches='tight')
plt.show()




#####################################################EMD与多种分解方法比较
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter, resample
from sklearn.metrics import mean_squared_error, r2_score
from PyEMD import EEMD, CEEMDAN
from scipy.signal import savgol_filter
# ---------- EMD ----------
def EMD_Right(data, parN=0):
    """
    对 data 的第 parN 列做 EMD 分解。
    常数保护：若整列为常数，返回全 0 的单列 DataFrame，不再分解。
    """
    data_par = np.array(data.iloc[:, parN], dtype=float)

    # ----- 常数保护 -----
    if np.ptp(data_par) == 0:                      # 极差为 0
        return pd.DataFrame([data_par[0]]*data_par.size)

    # 正常分解
    emd = EMD()
    IMFs = emd.emd(data_par, np.arange(1, len(data_par)+1))
    IMFdf = pd.DataFrame(IMFs.T)                   # 转置后 → (样本数, IMF 数)
    return IMFdf


def EMD_recue(IMFdfnew):
    # 还原数据
    reconstructed_data_new = np.zeros_like(IMFdfnew.index, dtype=float)
    # 累加IMFs
    for i in range(IMFdfnew.shape[0]):
        for imfn in IMFdfnew.columns:  # 假设 imfn 是列名
            reconstructed_data_new[i] += IMFdfnew[imfn].iloc[i]
    return reconstructed_data_new

# --------------------------------------------------
# 工具：空分解/常数 双保护
# --------------------------------------------------
def _safe_imf_df(s, imfs, prefix='IMF'):
    """imfs: ndarray (N_imf, N_time) 或空"""
    if imfs.size == 0 or len(imfs) == 0:           # 空分解
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({f'{prefix}1': zero_imf, 'RES': s})
    return pd.DataFrame(imfs.T,
                        columns=[f'{prefix}{i}' for i in range(1, imfs.shape[0]+1)])

# --------------------------------------------------
# 1. EEMD
# --------------------------------------------------
def EEMD_Right(data, parN=0, trials=100, noise_width=0.05):
    s = np.array(data.iloc[:, parN])
    if np.allclose(s, s[0]) or np.std(s) == 0:
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({'IMF1': zero_imf, 'RES': s})
    imfs = EEMD(trials=trials, noise_width=noise_width).eemd(s)
    return _safe_imf_df(s, imfs, 'IMF')
def EEMD_recue(IMFdf): return IMFdf.sum(axis=1).values

# --------------------------------------------------
# 2. FEEMD
# --------------------------------------------------
def FEEMD_Right(data, parN=0, trials=50, noise_width=0.03):
    s = np.array(data.iloc[:, parN])
    if np.allclose(s, s[0]) or np.std(s) == 0:
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({'IMF1': zero_imf, 'RES': s})
    imfs = EEMD(trials=trials, noise_width=noise_width, max_imf=12).eemd(s)
    return _safe_imf_df(s, imfs, 'IMF')
def FEEMD_recue(IMFdf): return IMFdf.sum(axis=1).values

# --------------------------------------------------
# 3. CEEMD
# --------------------------------------------------
def CEEMD_Right(data, parN=0, trials=100):
    s = np.array(data.iloc[:, parN])
    if np.allclose(s, s[0]) or np.std(s) == 0:
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({'IMF1': zero_imf, 'RES': s})
    imfs = EEMD(trials=trials, noise_width=0.05).eemd(s)
    return _safe_imf_df(s, imfs, 'IMF')
def CEEMD_recue(IMFdf): return IMFdf.sum(axis=1).values

# --------------------------------------------------
# 4. CEEMDAN
# --------------------------------------------------
def CEEMDAN_Right(data, parN=0):
    s = np.array(data.iloc[:, parN])
    if np.allclose(s, s[0]) or np.std(s) == 0:
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({'IMF1': zero_imf, 'RES': s})
    imfs = CEEMDAN().ceemdan(s)
    return _safe_imf_df(s, imfs, 'IMF')
def CEEMDAN_recue(IMFdf): return IMFdf.sum(axis=1).values

# --------------------------------------------------
# 5. TVFEMD
# --------------------------------------------------
def TVFEMD_Right(data, parN=0):
    s = np.array(data.iloc[:, parN])
    if np.allclose(s, s[0]) or np.std(s) == 0:
        zero_imf = np.zeros_like(s)
        return pd.DataFrame({'TIMF1': zero_imf, 'RES': s})
    resid = s.copy()
    imfs = []
    for _ in range(12):
        wl = min(51, len(resid) // 2 * 2 - 1)
        if wl < 5:
            break
        trend = savgol_filter(resid, window_length=wl, polyorder=3, mode='nearest')
        imf = resid - trend
        imfs.append(imf)
        resid = trend.copy()
        if np.std(imf) < 1e-4:
            break
    imfs.append(resid)
    return _safe_imf_df(s, np.array(imfs), 'TIMF')
def TVFEMD_recue(IMFdf): return IMFdf.sum(axis=1).values



def plot_metric_violin(quality_df, metric='MSE', methods=None, palette='Set2',
                       save_path=None, trim_extreme=0.02):
    """
    只绘制 quality_df 中指定指标列的提琴图
    metric      : str  'MSE' | 'NMSE' | 'R2'
    methods     : list  默认只画绿色 7 种
    trim_extreme: float 0-1，去掉每组前后极端比例（默认 2%）
    """
    if methods is None:
        methods = ['EMD', 'EEMD', 'FEEMD', 'CEEMD', 'CEEMDAN', 'TVFEMD']

    # 1. 取出要画的数据
    plot_df = (quality_df
               .query("method in @methods")
               .reset_index()
               .loc[:, ['ID', 'param', 'method', metric]]
               .rename(columns={metric: 'value'}))

    # 2. 基于 IQR 去除每组离群点（仅剔除极端异常值）
    if trim_extreme > 0:
        def remove_outliers_iqr(g):
            q1 = g.quantile(0.25)
            q3 = g.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return g.where((g >= lower) & (g <= upper))
        plot_df['value'] = (plot_df
                            .groupby('method')['value']
                            .transform(remove_outliers_iqr))

    # 3. 绘图
    plt.figure(figsize=(8, 4))
    ax = sns.violinplot(data=plot_df,
                        x='method',
                        y='value',
                        inner='quartile',
                        palette=palette,
                        saturation=0.85)

    # 4. 美化
    ax.set_title(f'Comparison of {metric} indicators', fontsize=14, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=12)
    sns.despine(left=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    plt.show()
    return ax




#7种分解-还原函数对字典
method_dict = {
    'EMD':      (EMD_Right,      EMD_recue),
    'EEMD':     (EEMD_Right,     EEMD_recue),
    'FEEMD':    (FEEMD_Right,    FEEMD_recue),
    'CEEMD':    (CEEMD_Right,    CEEMD_recue),
    'CEEMDAN':  (CEEMDAN_Right,  CEEMDAN_recue),
    'TVFEMD':   (TVFEMD_Right,   TVFEMD_recue),
}

METHODS = ['EMD', 'EEMD','FEEMD','CEEMD','CEEMDAN','TVFEMD']  
# ==========================================================
# 1. 预分配结果表
#    用三级索引：(ID, param, method)
# ==========================================================
# 出现次数≥5 的 TrackID
IDs = df.index.value_counts().loc[lambda x: x >= 5].index.to_numpy()
# 从原 IDs 中无放回随机抽取 100 个
IDs = np.random.default_rng().choice(IDs, size=10, replace=False)
param=df.columns
param=param.drop('cluster')
result_df = pd.DataFrame(index=pd.MultiIndex.from_product([IDs, param, METHODS],
                                                          names=['ID', 'param', 'method']),
                         columns=['imf_count'], dtype=int)
quality_df = pd.DataFrame(index=pd.MultiIndex.from_product([IDs, param, METHODS],
                                                            names=['ID', 'param', 'method']),
                          columns=['MSE', 'NMSE', 'R2'], dtype=float)

# ==========================================================
# 2. 主循环
# ==========================================================
for ID in IDs:
    for par in param:
        # 2.1 取序列
        touchcellpoint = df.loc[ID].sort_values(by='Time')
        touchcellpoint=touchcellpoint.drop(columns='cluster')


        parN = int(np.where(param == par)[0])
        data_ser = pd.to_numeric(touchcellpoint.iloc[:, parN], errors='coerce').dropna()

        if data_ser.size < 3:                       # 跳过太短
            for m in METHODS:
                result_df.loc[(ID, par, m), 'imf_count'] = 0
                quality_df.loc[(ID, par, m), ['MSE', 'NMSE', 'R2']] = np.nan
            continue

        # 2.2 对每个方法分别计算
        for m in METHODS:
            decomp_fun, recon_fun = method_dict[m]

            # ---- 分解 ----
            IMFdf = decomp_fun(pd.DataFrame(data_ser), 0)
            imf_count = IMFdf.shape[1]
            result_df.loc[(ID, par, m), 'imf_count'] = imf_count

            # ---- 还原 ----
            reconstructed = recon_fun(IMFdf)
            original = data_ser.values

            # ---- 指标 ----
            mse  = mean_squared_error(original, reconstructed)
            nmse = mse / np.var(original) if np.var(original) != 0 else np.nan
            r2   = r2_score(original, reconstructed)
            quality_df.loc[(ID, par, m), ['MSE', 'NMSE', 'R2']] = [mse, nmse, r2]

quality_df.to_csv(os.path.join(save_dir, 'EMD与其他分解质量分数.csv'))
quality_clean = quality_df.dropna()

# 画 MSE
plot_metric_violin(quality_clean, metric='MSE',
                   save_path=os.path.join(save_dir, 'EMD与其他分解质量分数_MSE.pdf'))

# 画 R²
plot_metric_violin(quality_clean, metric='R2',
                   save_path=os.path.join(save_dir, 'EMD与其他分解质量分数_R2.pdf'))

# 只挑 3 种方法画 NMSE
#plot_metric_violin(quality_clean, metric='NMSE', methods=['EMD', 'CEEMDAN', 'TVFEMD'])












import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_box_2d(metrics_all: pd.DataFrame,
                target_param: str = None,
                target_metric: str = None,
                save_path: str = None,
                figsize: tuple = (6, 4),
                show_n: bool = False,   # 默认不展示样本量
                palette: str = 'Set2'):
    """
    箱线图 2D 美化版
    用法同前，show_n=False 时不展示数据量
    """
    # 1. 字体与负号
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='whitegrid', font='DejaVu Sans')

    # 2. 数据筛选（同前）
    if target_param is not None and target_metric is not None:
        plot_df = metrics_all.loc[:, (target_param, target_metric)]
        x_col, y_col = 'group', 'value'
        long = (plot_df.to_frame(name=y_col)
                       .assign(group=f'{target_param}-{target_metric}'))
        title = f'{target_param} - {target_metric}'
    elif target_param is not None:
        plot_df = metrics_all[target_param]
        x_col, y_col = 'metric', 'value'
        long = (plot_df.stack().rename(y_col).reset_index()
                       .rename(columns={'level_1': x_col}))
        title = f'Metrics Distribution for Parameter {target_param}'
    elif target_metric is not None:
        plot_df = metrics_all.xs(target_metric, level=1, axis=1)
        x_col, y_col = 'param', 'value'
        long = (plot_df.stack().rename(y_col).reset_index()
                       .rename(columns={'level_1': x_col}))
        title = f'Parameter Distribution for Metric {target_metric}'
    else:
        raise ValueError('必须指定 target_param 和/或 target_metric')

    # 3. IQR 去异常
    def iqr_filter(g):
        q1, q3 = g[y_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return g[(g[y_col] >= low) & (g[y_col] <= high)]
    long = long.groupby(x_col, group_keys=False).apply(iqr_filter)

    # 4. 绘图
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=long,
                     x=x_col,
                     y=y_col,
                     palette=palette,
                     width=0.55,
                     linewidth=1.2,
                     fliersize=3,
                     saturation=0.85)

    # 5. 轴标签美化
    ax.set_title(title, fontsize=12, weight='bold', pad=12)
    ax.set_xlabel('', fontsize=0)   # 去掉默认 xlabel
    ax.set_ylabel(y_col, fontsize=12)
    # 旋转+对齐，防止重叠
    plt.xticks(rotation=30, ha='right', fontsize=3)
    sns.despine(left=True)          # 去掉左边脊线

    # 6. 保存 & 展示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return ax


def plot_box_2d_pdfmerge(metrics_all,
                target_param: str = None,
                target_metric: str = None,
                save_path: str = None,
                figsize: tuple = (6, 4),
                show_n: bool = False,
                palette: str = 'Set2',
                name=['LSTM', 'LSTM-EMD', 'LSTM-CEEMDAN', 'LSTM-TVFEMD']):
    """
    箱线图 2D 美化版
    支持单 DataFrame 或 DataFrame 列表（一页多图）
    传入列表时：save_path 必须给 *.pdf，否则只显示不保存
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages

    # ---- 统一美化 ----
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='whitegrid', font='DejaVu Sans')

    # ---- 自动区分单/多套数据 ----
    df_list = metrics_all if isinstance(metrics_all, list) else [metrics_all]

    n_plot = len(df_list)
    n_col = min(2, n_plot)
    n_row = (n_plot + n_col - 1) // n_col

    # ---- 子图布局 ----
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(figsize[0] * n_col, figsize[1] * n_row),
                             squeeze=False)

    # ---- 通用数据准备函数 ----
    def prep_df(df, idx):
        if target_param is not None and target_metric is not None:
            plot_df = df.loc[:, (target_param, target_metric)]
            x_col, y_col = 'group', 'value'
            long = (plot_df.to_frame(name=y_col)
                           .assign(group=f'{target_param}-{target_metric}'))
            title = f'{target_param} - {target_metric}'
        elif target_param is not None:
            plot_df = df[target_param]
            x_col, y_col = 'metric', 'value'
            long = (plot_df.stack().rename(y_col).reset_index()
                           .rename(columns={'level_1': x_col}))
            title = f'Metrics Distribution for Parameter {target_param}'
        elif target_metric is not None:
            plot_df = df.xs(target_metric, level=1, axis=1)
            x_col, y_col = 'param', 'value'
            long = (plot_df.stack().rename(y_col).reset_index()
                           .rename(columns={'level_1': x_col}))
            title = f'Parameter Distribution for Metric {target_metric}'
        else:
            raise ValueError('必须指定 target_param 和/或 target_metric')

        # IQR 去异常
        def iqr_filter(g):
            q1, q3 = g[y_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return g[(g[y_col] >= low) & (g[y_col] <= high)]
        long = long.groupby(x_col, group_keys=False).apply(iqr_filter)
        return long, x_col, y_col, title

    # ---- 绘制每个 DataFrame ----  新可视化开始  ----
    for i, df in enumerate(df_list):
        row, col = divmod(i, n_col)
        ax = axes[row, col]
        long, x_col, y_col, title = prep_df(df, i)

        # 1. 自动计算箱体宽度：组别越多，箱体越窄
        n_group = long[x_col].nunique()
        box_width = max(0.35, 0.8 - n_group * 0.04)

        # 2. 画箱线图
        sns.boxplot(
            data=long,
            x=x_col,
            y=y_col,
            palette=palette,
            width=box_width,
            linewidth=0.2,
            fliersize=0,          # 先把异常点关掉，后面单独画
            saturation=0.75,
            ax=ax,
        )

        # 3. 重画中位数 + 细化箱体边框
        for patch in ax.artists:
            # 边框变细 + 同色深色
            patch.set_linewidth(0.5)
            darker = patch.get_facecolor()[:3] * 0.75        # 深 25%
            patch.set_edgecolor(darker)
            patch.set_facecolor(patch.get_facecolor()[:3] + (0.55,))
            patch.set_joinstyle('round')
        # 4. 单独画异常点，样式更精致
        for k, (group, sub) in enumerate(long.groupby(x_col)):
            q1, q3 = sub[y_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flier = sub[(sub[y_col] < low) | (sub[y_col] > high)]
            if not flier.empty:
                color = sns.color_palette(palette, n_group)[k]
                ax.scatter(
                    flier[x_col],
                    flier[y_col],
                    color='white',
                    edgecolors=color,
                    linewidths=0.3,
                    s=10,
                    marker='o',
                    zorder=10,
                    label='',
                )

        # 5. 标题与坐标轴
        ax.set_title(
            f'{title} — {name[i]}',
            fontsize=12,
            weight='bold',
            pad=10,
            color='#2b2b2b',
        )
        # 给标题加一条浅灰色下划线
        ax.title.set_position((0.5, 1.02))
        fig.canvas.draw()
        p = ax.title.get_position()
        line = plt.Line2D(
            (p[0] - 0.15, p[0] + 0.15),
            (p[1] - 0.015, p[1] - 0.015),
            transform=ax.transAxes,
            color='#d0d0d0',
            linewidth=1,
        )
        ax.add_artist(line)

        ax.set_xlabel('')
        ax.set_ylabel(y_col, fontsize=11)
        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha='right',
            rotation_mode='anchor',
            fontsize=2,
        )

        # 6. 网格与脊柱
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', alpha=0.15)
        sns.despine(ax=ax, left=False, right=True, top=True, bottom=True)
        ax.spines['left'].set_linewidth(1.1)

        # 加粗中位线
        for l in ax.lines[4::6]:      # seaborn 0.12+ 每个 box 对应第 4 条线为中位线
            l.set_linewidth(1.8)
            l.set_color('white')
            l.set_solid_capstyle('round')

    # ---- 删除多余子图 ----
    for j in range(n_plot, n_row * n_col):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()

    # ---- 保存 / 显示 ----
    if save_path and n_plot > 1:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'📦 多图已保存 → {save_path}')
    elif save_path and n_plot == 1:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'💾 单图已保存 → {save_path}')
    else:
        plt.show()
    return fig



def plot_violin_2d(metrics_all: pd.DataFrame,
                   target_param: str  = None,
                   target_metric: str  = None,
                   save_path: str = None,
                   figsize: tuple = (6, 4),
                   show_n: bool = True):


    """
    对双重列索引 (param, metric) 的指标表画提琴图
    用法：
    1. 只看 Area 参数下全部指标：
       plot_violin_2d(metrics_all, target_param='Area')
    2. 只看 R2 指标在所有参数下分布：
       plot_violin_2d(metrics_all, target_metric='R2')
    3. 只看 Area 参数的 R2 分布：
       plot_violin_2d(metrics_all, target_param='Area', target_metric='R2')
    """
    # 1. 数据筛选
    if target_param is not None and target_metric is not None:  # 单格
        plot_df = metrics_all.loc[:, (target_param, target_metric)]
        x_col, y_col = 'dummy', 'value'   # 仅一组， dummy
        long = plot_df.to_frame(name=y_col).assign(dummy=target_param)
        title = f'{target_param} - {target_metric}'
    elif target_param is not None:                              # param 视角
        plot_df = metrics_all[target_param]
        x_col, y_col = 'metric', 'value'
        long = plot_df.stack().rename(y_col).reset_index().rename(columns={'level_1': x_col})
        title = f'参数 {target_param} 各指标分布'
    elif target_metric is not None:                             # metric 视角
        plot_df = metrics_all.xs(target_metric, level=1, axis=1)
        x_col, y_col = 'param', 'value'
        long = plot_df.stack().rename(y_col).reset_index().rename(columns={'level_1': x_col})
        title = f'指标 {target_metric} 各参数分布'
    else:
        raise ValueError('必须指定 target_param 和/或 target_metric')

    # 2. 按组剔除异常值（IQR 1.5）
    def iqr_filter(g):
        q1, q3 = g[y_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        return g[(g[y_col] >= low) & (g[y_col] <= high)]
    long = long.groupby(x_col, group_keys=False).apply(iqr_filter)

    # 3. 画图
    sns.set(style='whitegrid', font='SimHei')
    plt.figure(figsize=figsize)
    ax = sns.violinplot(data=long, x=x_col, y=y_col, palette='Set2', inner='box')
    if show_n:
        n_counts = long[x_col].value_counts().sort_index()
        for i, key in enumerate(n_counts.index):
            ax.text(i, ax.get_ylim()[1]*0.95, f'n={n_counts[key]}',
                    ha='center', va='top', fontsize=9)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return ax








################################################预测值与真实值比较——美化
def plot_lstm_vs_true(df_true, df_pred,
                      parN=5,
                      time_col='Time',
                      value_col=None,
                      figsize=(6.8, 2.8),
                      save_path=None,
                      ext='pdf',
                      dpi=600):
    """
    df_true : 完整真实数据（含未来段）
    df_pred : 拼接了【训练段+预测段】，列与 df_true 相同
    parN    : 要绘制的列序号（默认第 5 列）
    """
    if value_col is None:
        value_col = df_true.columns[parN]

    # 1. 时间轴
    t_true = df_true[time_col]
    t_pred = df_pred[time_col]

    # 2. 分段：训练 / 预测
    last_train_t = df_pred[df_pred.index < len(df_pred) - 5][time_col].max()
    mask_train = t_pred <= last_train_t
    mask_fore  = t_pred >  last_train_t

    # 3. 绘图
    plt.rcParams.update({'font.family': 'Times New Roman',
                         'axes.labelsize': 9,
                         'xtick.labelsize': 8,
                         'ytick.labelsize': 8,
                         'legend.fontsize': 8,
                         'axes.linewidth': 0.5})
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 3.1 真实全程（灰，细线）
    ax.plot(t_true, df_true[value_col],
            color='lightgray', lw=1.2, label='True', zorder=2)

    # 3.2 训练段（雾蓝，实线）
    ax.plot(t_pred[mask_train], df_pred.loc[mask_train, value_col],
            color='#5B8EA7', lw=2, label='Train', zorder=3)

    # 3.3 预测段（酒红，长虚线）
    ax.plot(t_pred[mask_fore], df_pred.loc[mask_fore, value_col],
            color='#D62728', lw=1.8, dashes=(6, 3),
            label='LSTM forecast', zorder=4)

    # 3.4 预测区间阴影
    ax.axvspan(t_pred[mask_fore].min(), t_pred[mask_fore].max(),
               color='#D62728', alpha=0.08, zorder=1)

    # 3.5 稀疏符号（仅预测段）
    step = max(1, mask_fore.sum() // 8)
    ax.scatter(t_pred[mask_fore][::step],
               df_pred.loc[mask_fore, value_col][::step],
               color='#D62728', s=20, zorder=4,
               marker='o', linewidths=0.4, edgecolors='white')

    # 4.  cosmetic
    ax.set_title(f'{value_col}  LSTM Forecast vs True', fontsize=10, weight='bold', pad=6)
    ax.set_xlabel(time_col, fontsize=9, labelpad=2)
    ax.set_ylabel(value_col, fontsize=9, labelpad=2)
    ax.legend(frameon=False, loc='best', ncol=3)
    ax.grid(alpha=0.1)
    sns.despine(ax=ax)

    # 5. 保存
    if save_path:
        out_file = f'{save_path}.{ext}'
        plt.savefig(out_file, dpi=dpi if ext == 'png' else None,
                    bbox_inches='tight', transparent=True)
        print(f'图已保存 → {out_file}')
    plt.show()



