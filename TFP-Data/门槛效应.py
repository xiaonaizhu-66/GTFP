"""
超简化门槛效应分析 - 无需statsmodels
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("超简化门槛效应分析")
print("="*60)

# 1. 创建或加载数据
print("\n1. 创建示例数据...")
np.random.seed(42)
n = 300  # 300个观测值

# 创建示例数据（请替换为你的真实数据）
data = {
    '公司代码': [f'C{i:03d}' for i in range(1, n+1)],
    '年份': np.random.choice([2019, 2020, 2021, 2022, 2023], n),
    # 数字化转型指数
    '数字化指数': np.random.normal(50, 15, n).clip(0, 100),
    # GTFP值
    'GTFP': np.random.uniform(0.4, 0.9, n),
    # 绿色创新
    '绿色创新': np.random.exponential(5, n),
    # 公司规模
    '公司规模': np.random.lognormal(10, 1, n),
}

df = pd.DataFrame(data)

# 模拟门槛效应：数字化指数>55时，绿色创新对GTFP影响更大
high_digital_mask = df['数字化指数'] > 55
df.loc[high_digital_mask, 'GTFP'] += df.loc[high_digital_mask, '绿色创新'] * 0.08
df.loc[~high_digital_mask, 'GTFP'] += df.loc[~high_digital_mask, '绿色创新'] * 0.02

print(f"数据形状: {df.shape}")
print(f"前5行数据:")
print(df.head())

# 2. 基本统计分析
print("\n2. 基本统计分析:")
print("-"*40)
print(df[['数字化指数', 'GTFP', '绿色创新', '公司规模']].describe())

# 3. 相关系数分析
print("\n3. 相关系数分析:")
print("-"*40)
corr_matrix = df[['数字化指数', 'GTFP', '绿色创新', '公司规模']].corr()
print(corr_matrix.round(3))

# 4. 简单门槛分析（中位数分组）
print("\n4. 简单门槛分析:")
print("-"*40)

threshold = df['数字化指数'].median()
print(f"数字化指数中位数: {threshold:.2f}")

low_group = df[df['数字化指数'] <= threshold]
high_group = df[df['数字化指数'] > threshold]

print(f"低数字化组: {len(low_group)} 个样本")
print(f"高数字化组: {len(high_group)} 个样本")

# 组间GTFP差异
print(f"\nGTFP比较:")
print(f"低组GTFP均值: {low_group['GTFP'].mean():.4f}")
print(f"高组GTFP均值: {high_group['GTFP'].mean():.4f}")
print(f"差异: {high_group['GTFP'].mean() - low_group['GTFP'].mean():.4f}")

# t检验
t_stat, p_value = stats.ttest_ind(high_group['GTFP'], low_group['GTFP'], equal_var=False)
print(f"t检验: t={t_stat:.4f}, p={p_value:.4f}")

if p_value < 0.05:
    print("✅ 组间GTFP差异显著!")
else:
    print("❌ 组间GTFP差异不显著")

# 5. 绿色创新与GTFP关系分析（按组）
print("\n5. 绿色创新与GTFP关系:")
print("-"*40)

def analyze_relationship(group_df, group_name):
    """分析绿色创新与GTFP的关系"""
    # 简单线性回归：y = a + b*x
    x = group_df['绿色创新']
    y = group_df['GTFP']
    
    # 计算回归系数（最小二乘法）
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    # 斜率 b = Σ[(xi - x_mean)(yi - y_mean)] / Σ[(xi - x_mean)^2]
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator > 0:
        b = numerator / denominator
        a = y_mean - b * x_mean
        
        # 计算R平方
        y_pred = a + b * x
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算标准误
        se = np.sqrt(ss_res / (n - 2) / denominator)
        
        # t检验
        t_stat = b / se if se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        print(f"{group_name}:")
        print(f"  样本数: {n}")
        print(f"  系数b: {b:.6f}")
        print(f"  标准误: {se:.6f}")
        print(f"  t值: {t_stat:.4f}")
        print(f"  p值: {p_val:.4f}")
        print(f"  R平方: {r_squared:.4f}")
        
        return b, se, p_val
    else:
        print(f"{group_name}: 无法计算回归")
        return np.nan, np.nan, np.nan

print("\n低数字化组分析:")
low_b, low_se, low_p = analyze_relationship(low_group, "低数字化组")

print("\n高数字化组分析:")
high_b, high_se, high_p = analyze_relationship(high_group, "高数字化组")

# 6. 系数差异检验
print("\n6. 系数差异检验:")
print("-"*40)

if not np.isnan(low_b) and not np.isnan(high_b):
    coef_diff = high_b - low_b
    se_pooled = np.sqrt(low_se**2 + high_se**2)
    
    if se_pooled > 0:
        t_stat_diff = coef_diff / se_pooled
        df_total = len(low_group) + len(high_group) - 4  # 两个参数
        p_value_diff = 2 * (1 - stats.t.cdf(abs(t_stat_diff), df_total))
        
        print(f"绿色创新系数差异: {coef_diff:.6f}")
        print(f"t统计量: {t_stat_diff:.4f}")
        print(f"p值: {p_value_diff:.4f}")
        
        if p_value_diff < 0.01:
            print("✅✅ 存在非常显著的门槛效应!")
        elif p_value_diff < 0.05:
            print("✅ 存在显著的门槛效应!")
        elif p_value_diff < 0.1:
            print("⚠️ 存在边缘显著的门槛效应")
        else:
            print("❌ 未发现显著的门槛效应")
    else:
        print("无法计算系数差异")
else:
    print("无法进行系数差异检验")

# 7. 寻找最优门槛值
print("\n7. 寻找最优门槛值:")
print("-"*40)

# 在30-70分位数范围内搜索
percentiles = range(30, 71, 5)
candidates = np.percentile(df['数字化指数'], percentiles)

best_threshold = None
best_f_stat = -np.inf

print("搜索候选门槛值:")
for p, candidate in zip(percentiles, candidates):
    low_mask = df['数字化指数'] <= candidate
    high_mask = df['数字化指数'] > candidate
    
    if sum(low_mask) < 20 or sum(high_mask) < 20:
        continue
    
    # 计算各组方差
    var_low = df[low_mask]['GTFP'].var()
    var_high = df[high_mask]['GTFP'].var()
    
    # 计算F统计量（方差比检验）
    if var_low > 0 and var_high > 0:
        f_stat = var_high / var_low if var_high >= var_low else var_low / var_high
        
        if f_stat > best_f_stat:
            best_f_stat = f_stat
            best_threshold = candidate
        
        print(f"  第{p}百分位 ({candidate:.1f}): F={f_stat:.3f}")

if best_threshold is not None:
    print(f"\n最优门槛值: {best_threshold:.2f}")
    print(f"最大F统计量: {best_f_stat:.4f}")

# 8. 结果可视化
print("\n8. 生成可视化图表...")
print("-"*40)

plt.figure(figsize=(15, 10))

# 图1: 数字化与GTFP关系
plt.subplot(2, 3, 1)
plt.scatter(df['数字化指数'], df['GTFP'], alpha=0.6, s=30, c='blue')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'中位数: {threshold:.1f}')
if best_threshold:
    plt.axvline(x=best_threshold, color='green', linestyle=':', linewidth=2, label=f'最优: {best_threshold:.1f}')
plt.xlabel('数字化指数')
plt.ylabel('GTFP')
plt.title('数字化转型与GTFP关系')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2: 分组箱线图
plt.subplot(2, 3, 2)
box_data = [low_group['GTFP'], high_group['GTFP']]
box_labels = [f'低数字化\n(n={len(low_group)})', f'高数字化\n(n={len(high_group)})']
bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
# 设置颜色
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.ylabel('GTFP')
plt.title('不同数字化水平的GTFP分布')

# 图3: 回归线对比
plt.subplot(2, 3, 3)
# 低组回归线
if not np.isnan(low_b):
    x_low = np.linspace(low_group['绿色创新'].min(), low_group['绿色创新'].max(), 100)
    y_low = np.mean(low_group['GTFP']) + low_b * (x_low - np.mean(low_group['绿色创新']))
    plt.plot(x_low, y_low, 'b-', linewidth=2, label='低数字化组')

# 高组回归线
if not np.isnan(high_b):
    x_high = np.linspace(high_group['绿色创新'].min(), high_group['绿色创新'].max(), 100)
    y_high = np.mean(high_group['GTFP']) + high_b * (x_high - np.mean(high_group['绿色创新']))
    plt.plot(x_high, y_high, 'r-', linewidth=2, label='高数字化组')

plt.xlabel('绿色创新')
plt.ylabel('GTFP')
plt.title('绿色创新与GTFP关系对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 图4: 数字化分布
plt.subplot(2, 3, 4)
plt.hist(df['数字化指数'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
plt.xlabel('数字化指数')
plt.ylabel('频数')
plt.title('数字化转型指数分布')
plt.grid(True, alpha=0.3)

# 图5: 相关系数热力图
plt.subplot(2, 3, 5)
corr_vars = ['数字化指数', 'GTFP', '绿色创新', '公司规模']
corr_values = df[corr_vars].corr().values
plt.imshow(corr_values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='相关系数')
plt.xticks(range(len(corr_vars)), corr_vars, rotation=45)
plt.yticks(range(len(corr_vars)), corr_vars)
plt.title('变量相关性')

# 添加数值
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        plt.text(j, i, f'{corr_values[i, j]:.2f}', 
                ha='center', va='center', color='black', fontsize=10)

# 图6: 年度趋势（如果有多个年份）
plt.subplot(2, 3, 6)
if df['年份'].nunique() > 1:
    yearly_stats = df.groupby('年份').agg({
        'GTFP': 'mean',
        '数字化指数': 'mean'
    }).reset_index()
    
    fig6, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(yearly_stats['年份'], yearly_stats['GTFP'], 'b-o', linewidth=2, markersize=8, label='GTFP')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('平均GTFP', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(yearly_stats['年份'])
    
    ax2 = ax1.twinx()
    ax2.plot(yearly_stats['年份'], yearly_stats['数字化指数'], 'r-s', linewidth=2, markersize=8, label='数字化指数')
    ax2.set_ylabel('平均数字化指数', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('年度趋势分析')
    plt.grid(True, alpha=0.3)
else:
    # 如果没有多个年份，显示密度图
    from scipy.stats import gaussian_kde
    
    x = df['数字化指数']
    y = df['GTFP']
    
    # 计算二维密度
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    plt.scatter(x, y, c=z, s=30, alpha=0.6, cmap='viridis')
    plt.xlabel('数字化指数')
    plt.ylabel('GTFP')
    plt.title('数字化与GTFP密度图')
    plt.colorbar(label='密度')
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图表
output_path = r"C:\Users\15535\Desktop\TFP-Data\超简门槛效应分析.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存: {output_path}")
plt.show()

# 9. 保存分析结果
print("\n9. 保存分析结果...")
print("-"*40)

results = {
    '分析时间': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    '总样本数': len(df),
    '数字化指数均值': f"{df['数字化指数'].mean():.2f}",
    '数字化指数标准差': f"{df['数字化指数'].std():.2f}",
    'GTFP均值': f"{df['GTFP'].mean():.4f}",
    '中位数门槛值': f"{threshold:.2f}",
    '低组样本数': len(low_group),
    '高组样本数': len(high_group),
    '低组GTFP均值': f"{low_group['GTFP'].mean():.4f}",
    '高组GTFP均值': f"{high_group['GTFP'].mean():.4f}",
    'GTFP差异': f"{high_group['GTFP'].mean() - low_group['GTFP'].mean():.4f}",
    'GTFP差异显著性': '显著' if p_value < 0.05 else '不显著',
    '低组绿色创新系数': f"{low_b:.6f}" if not np.isnan(low_b) else 'NaN',
    '高组绿色创新系数': f"{high_b:.6f}" if not np.isnan(high_b) else 'NaN',
    '系数差异': f"{coef_diff:.6f}" if 'coef_diff' in locals() else 'NaN',
    '系数差异显著性': '显著' if 'p_value_diff' in locals() and p_value_diff < 0.05 else '不显著',
    '最优门槛值': f"{best_threshold:.2f}" if best_threshold else '未找到',
    '是否存在门槛效应': '是' if ('p_value_diff' in locals() and p_value_diff < 0.05) else '否'
}

# 转换为DataFrame并保存
results_df = pd.DataFrame([results])
results_path = r"C:\Users\15535\Desktop\TFP-Data\超简门槛效应结果.csv"
results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
print(f"✅ 分析结果已保存: {results_path}")

print("\n" + "="*60)
print("🎉 分析完成!")
print("="*60)

print("\n📋 主要发现:")
print(f"1. 数字化指数中位数: {threshold:.2f}")
print(f"2. 高数字化组GTFP比低组高: {high_group['GTFP'].mean() - low_group['GTFP'].mean():.4f}")
print(f"   显著性: {'显著' if p_value < 0.05 else '不显著'}")

if 'coef_diff' in locals() and not np.isnan(coef_diff):
    print(f"3. 绿色创新系数差异: {coef_diff:.6f}")
    if 'p_value_diff' in locals():
        print(f"   显著性: {'显著' if p_value_diff < 0.05 else '不显著'}")
    
    if p_value_diff < 0.05:
        print(f"   ✅ 发现门槛效应: 数字化改变了绿色创新对GTFP的影响")
        print(f"   低数字化组系数: {low_b:.6f}")
        print(f"   高数字化组系数: {high_b:.6f}")
        if best_threshold:
            print(f"   最优门槛值: {best_threshold:.2f}")
    else:
        print(f"   ❌ 未发现显著的门槛效应")

print(f"\n📊 分析图表: {output_path}")
print(f"📄 详细结果: {results_path}")