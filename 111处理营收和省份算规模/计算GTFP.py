"""
修复版GTFP快速计算脚本
处理重复值问题
"""
import pandas as pd
import numpy as np
import warnings
import os
import time
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')

print("="*60)
print("GTFP快速计算脚本 v3.0 - 修复重复值问题")
print("="*60)

# ==================== 1. 设置路径 ====================
input_file = r"C:\Users\15535\Desktop\111处理营收和省份算规模\SBM_DEA.csv"
output_dir = r"C:\Users\15535\Desktop\111处理营收和省份算规模"

print(f"输入文件: {input_file}")
print(f"输出目录: {output_dir}")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# ==================== 2. 加载数据 ====================
print("\n" + "-"*40)
print("步骤1: 加载数据")

def safe_load_csv(file_path):
    """安全加载CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ 成功使用 {encoding} 编码加载文件")
            return df, encoding
        except:
            continue
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        return df, 'utf-8(ignore)'
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        sys.exit(1)

df, file_encoding = safe_load_csv(input_file)
print(f"数据形状: {df.shape}")

# 清理列名
df.columns = df.columns.str.strip()

# 重命名列（修正可能的列名错误）
column_mapping = {
    '非期望_NOX': '非期望_NOx',  # 修正大小写不一致
}
for old, new in column_mapping.items():
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})

# ==================== 3. 检查必要列 ====================
print("\n" + "-"*40)
print("步骤2: 检查数据列")

required_cols = ['DMU', '投入_资本', '投入_劳动', '投入_能源', 
                 '产出_营收', '非期望_SO2', '非期望_NOx']

missing_cols = []
for col in required_cols:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"❌ 缺少列: {missing_cols}")
    print("可用的列:", list(df.columns))
    
    # 尝试自动匹配
    for col in missing_cols:
        for actual_col in df.columns:
            if col in actual_col or actual_col in col:
                print(f"  可能 '{actual_col}' 对应 '{col}'")
                df = df.rename(columns={actual_col: col})
                break
    
    # 重新检查
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 仍然缺少列: {missing_cols}")
        sys.exit(1)

print("✅ 所有必要列都存在")

# ==================== 4. 数据清洗 ====================
print("\n" + "-"*40)
print("步骤3: 数据清洗")

# 处理缺失值
numeric_cols = ['投入_资本', '投入_劳动', '投入_能源', 
                '产出_营收', '非期望_SO2', '非期望_NOx']

print("缺失值处理:")
for col in numeric_cols:
    if col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: 用中位数 {median_val:.2f} 填充 {missing} 个缺失值")

# 确保数据类型
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 移除完全相同的行（如果有）
initial_rows = len(df)
df = df.drop_duplicates(subset=numeric_cols)
if len(df) < initial_rows:
    print(f"  移除 {initial_rows - len(df)} 个完全重复的行")

print("✅ 数据清洗完成")

# ==================== 5. 计算GTFP ====================
print("\n" + "-"*40)
print("步骤4: 计算GTFP")

start_calc = time.time()

# 1. 计算基本效率指标
print("计算基础效率指标...")
epsilon = 1e-10  # 防止除零

# 添加小随机噪声，避免完全相同的值
np.random.seed(42)
noise_scale = 1e-12

for col in numeric_cols:
    if col in df.columns:
        # 添加微小随机噪声
        noise = np.random.normal(0, noise_scale, len(df))
        df[col] = df[col] + noise

# 计算效率
df['资本效率'] = df['产出_营收'] / (df['投入_资本'] + epsilon)
df['劳动效率'] = df['产出_营收'] / (df['投入_劳动'] + epsilon)
df['能源效率'] = df['产出_营收'] / (df['投入_能源'] + epsilon)

# 2. 计算综合指标
df['总投入'] = df['投入_资本'] + df['投入_劳动'] + df['投入_能源']
df['总污染'] = df['非期望_SO2'] + df['非期望_NOx']

# 3. 计算传统TFP
df['传统TFP'] = df['产出_营收'] / (df['总投入'] + epsilon)

# 4. 计算绿色TFP（考虑污染）
# 方法1：简单比率
df['绿色TFP_简单'] = df['产出_营收'] / ((df['总投入'] + 1) * (df['总污染'] + 1))

# 方法2：使用对数形式（更稳定）
df['ln营收'] = np.log(df['产出_营收'] + 1)
df['ln总投入'] = np.log(df['总投入'] + 1)
df['ln总污染'] = np.log(df['总污染'] + 1)
df['绿色TFP_对数'] = df['ln营收'] - 0.7 * df['ln总投入'] - 0.3 * df['ln总污染']

# 5. 综合GTFP（多种方法的平均）
df['综合GTFP'] = (df['绿色TFP_简单'] + df['绿色TFP_对数']) / 2

# 6. 标准化到0-1范围（添加微小差异）
print("标准化处理...")
# 先排序，确保有差异
sorted_gtfp = np.sort(df['综合GTFP'].values)

# 创建新的标准化值，确保有足够差异
min_val = sorted_gtfp[0]
max_val = sorted_gtfp[-1]
range_val = max_val - min_val

if range_val < epsilon * 100:  # 如果范围太小
    print("⚠️  GTFP值差异太小，使用排名标准化")
    # 使用排名标准化
    df['标准化GTFP'] = df['综合GTFP'].rank(method='first') / len(df)
else:
    df['标准化GTFP'] = (df['综合GTFP'] - min_val) / (range_val + epsilon)

# 7. 确保标准化值在0-1之间
df['标准化GTFP'] = df['标准化GTFP'].clip(0, 1)

# 8. 添加微小随机差异避免完全相同
df['标准化GTFP'] = df['标准化GTFP'] + np.random.uniform(-1e-10, 1e-10, len(df))

# 9. 排名
df['GTFP排名'] = df['标准化GTFP'].rank(ascending=False, method='first').astype(int)
df['百分位排名'] = (df['GTFP排名'] / len(df) * 100).round(2)

# 10. 分级 - 使用自定义分位数避免重复边界问题
print("计算效率等级...")
try:
    # 尝试使用qcut
    df['效率等级'] = pd.qcut(df['标准化GTFP'], q=5, labels=['E级', 'D级', 'C级', 'B级', 'A级'], duplicates='drop')
except:
    print("⚠️  qcut失败，使用等距分箱")
    # 如果qcut失败，使用等距分箱
    bins = np.linspace(df['标准化GTFP'].min(), df['标准化GTFP'].max(), 6)
    labels = ['E级', 'D级', 'C级', 'B级', 'A级']
    df['效率等级'] = pd.cut(df['标准化GTFP'], bins=bins, labels=labels, include_lowest=True)

# 11. 计算松弛变量
print("计算松弛变量...")
# 计算行业平均值（按效率等级分组）
if '效率等级' in df.columns:
    group_means = df.groupby('效率等级')[numeric_cols].transform('mean')
else:
    group_means = df[numeric_cols].mean()

# 计算松弛（与组平均的差距）
for col in ['投入_资本', '投入_劳动', '投入_能源']:
    if col in df.columns:
        mean_col = f'{col}_组平均'
        if mean_col in group_means.columns:
            df[f'{col}_松弛'] = df[col] - group_means[mean_col]
            df[f'{col}_改进%'] = np.where(df[col] > epsilon, 
                                         df[f'{col}_松弛'] / df[col] * 100, 0)

for col in ['产出_营收']:
    if col in df.columns:
        mean_col = f'{col}_组平均'
        if mean_col in group_means.columns:
            df[f'{col}_不足'] = group_means[mean_col] - df[col]
            df[f'{col}_提升%'] = np.where(df[col] > epsilon, 
                                        df[f'{col}_不足'] / df[col] * 100, 0)

for col in ['非期望_SO2', '非期望_NOx']:
    if col in df.columns:
        mean_col = f'{col}_组平均'
        if mean_col in group_means.columns:
            df[f'{col}_过剩'] = df[col] - group_means[mean_col]
            df[f'{col}_削减%'] = np.where(df[col] > epsilon, 
                                        df[f'{col}_过剩'] / df[col] * 100, 0)

calc_time = time.time() - start_calc
print(f"✅ GTFP计算完成，耗时: {calc_time:.2f}秒")
print(f"  标准化GTFP范围: {df['标准化GTFP'].min():.6f} 到 {df['标准化GTFP'].max():.6f}")
print(f"  唯一值数量: {df['标准化GTFP'].nunique()}")
print(f"  效率等级分布: {df['效率等级'].value_counts().to_dict()}")

# ==================== 6. 保存CSV结果 ====================
print("\n" + "-"*40)
print("步骤5: 保存CSV结果")

# 选择重要列
output_cols = [
    'DMU', 
    '标准化GTFP', 
    '效率等级', 
    'GTFP排名', 
    '百分位排名',
    '传统TFP', 
    '绿色TFP_简单', 
    '绿色TFP_对数',
    '综合GTFP',
    '资本效率', 
    '劳动效率', 
    '能源效率',
    '总投入',
    '总污染'
]

# 添加松弛变量
slack_cols = [col for col in df.columns if any(x in col for x in ['松弛', '不足', '过剩', '改进%', '提升%', '削减%'])]

# 合并所有列
all_output_cols = [col for col in output_cols if col in df.columns] + slack_cols[:8]

# 保存完整结果
csv_path = os.path.join(output_dir, "GTFP_完整结果.csv")
df[all_output_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV文件已保存: {csv_path}")
print(f"  包含 {len(all_output_cols)} 列，{len(df)} 行")

# 保存前1000名样本（便于查看）
sample_path = os.path.join(output_dir, "GTFP_前1000名.csv")
df.nlargest(1000, '标准化GTFP')[all_output_cols].to_csv(sample_path, index=False, encoding='utf-8-sig')
print(f"✅ 样本文件已保存: GTFP_前1000名.csv")

# ==================== 7. 生成可视化图表 ====================
print("\n" + "-"*40)
print("步骤6: 生成可视化图表")

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图表1: GTFP分布
    plt.figure(figsize=(10, 6))
    n_bins = min(100, df['标准化GTFP'].nunique())
    plt.hist(df['标准化GTFP'], bins=n_bins, edgecolor='black', alpha=0.7)
    plt.axvline(df['标准化GTFP'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df["标准化GTFP"].mean():.4f}')
    plt.xlabel('标准化GTFP')
    plt.ylabel('频数')
    plt.title(f'GTFP分布直方图 (n={len(df):,})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart1_path = os.path.join(output_dir, "GTFP_分布图.png")
    plt.savefig(chart1_path, dpi=150)
    print(f"✅ 图表1: GTFP_分布图.png")
    plt.close()
    
    # 图表2: 效率等级
    plt.figure(figsize=(8, 6))
    grade_counts = df['效率等级'].value_counts().sort_index()
    colors = ['#ff4444', '#ff8844', '#ffcc44', '#44cc44', '#4488ff']
    bars = plt.bar(range(len(grade_counts)), grade_counts.values, color=colors)
    plt.xticks(range(len(grade_counts)), grade_counts.index)
    plt.xlabel('效率等级')
    plt.ylabel('数量')
    plt.title('效率等级分布')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(grade_counts.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "GTFP_等级分布.png")
    plt.savefig(chart2_path, dpi=150)
    print(f"✅ 图表2: GTFP_等级分布.png")
    plt.close()
    
    # 图表3: 前20名
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(df))
    top_df = df.nlargest(top_n, '标准化GTFP').sort_values('标准化GTFP', ascending=True)
    
    bars = plt.barh(range(top_n), top_df['标准化GTFP'])
    plt.yticks(range(top_n), top_df['DMU'], fontsize=9)
    plt.xlabel('标准化GTFP')
    plt.title(f'GTFP排名前{top_n}名')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "GTFP_前20名.png")
    plt.savefig(chart3_path, dpi=150)
    print(f"✅ 图表3: GTFP_前20名.png")
    plt.close()
    
    # 图表4: 箱线图
    plt.figure(figsize=(10, 6))
    if '效率等级' in df.columns:
        # 按等级分组绘制箱线图
        data_to_plot = [df[df['效率等级']==level]['标准化GTFP'].values 
                       for level in df['效率等级'].cat.categories]
        
        box = plt.boxplot(data_to_plot, labels=df['效率等级'].cat.categories,
                         patch_artist=True)
        
        # 设置颜色
        colors = ['#ffcccc', '#ffe6cc', '#ffffcc', '#ccffcc', '#cce6ff']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xlabel('效率等级')
        plt.ylabel('标准化GTFP')
        plt.title('各效率等级GTFP分布箱线图')
        plt.grid(True, alpha=0.3, axis='y')
    else:
        # 简单箱线图
        plt.boxplot(df['标准化GTFP'].values)
        plt.ylabel('标准化GTFP')
        plt.title('GTFP箱线图')
    
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "GTFP_箱线图.png")
    plt.savefig(chart4_path, dpi=150)
    print(f"✅ 图表4: GTFP_箱线图.png")
    plt.close()
    
    print(f"✅ 所有4张图表已生成")
    
except Exception as e:
    print(f"⚠️ 图表生成出错: {e}")
    import traceback
    traceback.print_exc()

# ==================== 8. 生成详细报告 ====================
print("\n" + "-"*40)
print("步骤7: 生成详细报告")

report_path = os.path.join(output_dir, "GTFP_详细报告.txt")

try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("绿色全要素生产率(GTFP)分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件: {os.path.basename(input_file)}\n")
        f.write(f"总样本数: {len(df):,}\n")
        f.write(f"计算耗时: {calc_time:.2f}秒\n\n")
        
        f.write("一、数据质量\n")
        f.write("-"*40 + "\n")
        f.write(f"原始数据行数: {initial_rows:,}\n")
        f.write(f"处理后行数: {len(df):,}\n")
        f.write(f"GTFP唯一值数量: {df['标准化GTFP'].nunique():,}\n\n")
        
        f.write("二、GTFP统计\n")
        f.write("-"*40 + "\n")
        stats = df['标准化GTFP'].describe()
        f.write(f"平均值: {stats['mean']:.6f}\n")
        f.write(f"标准差: {stats['std']:.6f}\n")
        f.write(f"最小值: {stats['min']:.6f}\n")
        f.write(f"25%分位: {stats['25%']:.6f}\n")
        f.write(f"中位数: {stats['50%']:.6f}\n")
        f.write(f"75%分位: {stats['75%']:.6f}\n")
        f.write(f"最大值: {stats['max']:.6f}\n\n")
        
        f.write("三、效率等级分布\n")
        f.write("-"*40 + "\n")
        if '效率等级' in df.columns:
            grade_counts = df['效率等级'].value_counts().sort_index()
            for grade, count in grade_counts.items():
                percent = count / len(df) * 100
                f.write(f"{grade}: {count:,} ({percent:.1f}%)\n")
        f.write("\n")
        
        f.write("四、表现最佳单位\n")
        f.write("-"*40 + "\n")
        top10 = df.nlargest(10, '标准化GTFP')[['DMU', '标准化GTFP', '效率等级', 'GTFP排名']]
        for _, row in top10.iterrows():
            f.write(f"第{row['GTFP排名']:4}名: {row['DMU']:20} "
                   f"GTFP={row['标准化GTFP']:.6f} 等级={row['效率等级']}\n")
        f.write("\n")
        
        f.write("五、表现最差单位\n")
        f.write("-"*40 + "\n")
        bottom10 = df.nsmallest(10, '标准化GTFP')[['DMU', '标准化GTFP', '效率等级', 'GTFP排名']]
        for _, row in bottom10.iterrows():
            f.write(f"第{row['GTFP排名']:4}名: {row['DMU']:20} "
                   f"GTFP={row['标准化GTFP']:.6f} 等级={row['效率等级']}\n")
        f.write("\n")
        
        f.write("六、改进方向\n")
        f.write("-"*40 + "\n")
        improvement_metrics = [
            ('投入_资本_改进%', '资本投入', '减少'),
            ('投入_劳动_改进%', '劳动投入', '减少'),
            ('投入_能源_改进%', '能源投入', '减少'),
            ('产出_营收_提升%', '营业收入', '增加'),
            ('非期望_SO2_削减%', 'SO2排放', '减少'),
            ('非期望_NOx_削减%', 'NOx排放', '减少')
        ]
        
        for col, name, action in improvement_metrics:
            if col in df.columns:
                avg_val = df[col].mean()
                if abs(avg_val) > 0.1:  # 只显示有意义的改进
                    f.write(f"{name}: 平均需要{action} {abs(avg_val):.1f}%\n")
        f.write("\n")
        
        f.write("七、输出文件\n")
        f.write("-"*40 + "\n")
        output_files = [
            "GTFP_完整结果.csv",
            "GTFP_前1000名.csv",
            "GTFP_分布图.png",
            "GTFP_等级分布.png",
            "GTFP_前20名.png",
            "GTFP_箱线图.png",
            "GTFP_详细报告.txt"
        ]
        
        for file_name in output_files:
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                f.write(f"✓ {file_name} ({size_kb:.1f} KB)\n")
            else:
                f.write(f"✗ {file_name} (未生成)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告结束\n")
        f.write("="*70 + "\n")
    
    print(f"✅ 详细报告: GTFP_详细报告.txt")
    
except Exception as e:
    print(f"❌ 报告生成失败: {e}")

# ==================== 9. 显示最终结果 ====================
print("\n" + "="*60)
print("🎉 GTFP计算完成!")
print("="*60)

print(f"\n📊 主要统计:")
print(f"  样本数量: {len(df):,}")
print(f"  平均GTFP: {df['标准化GTFP'].mean():.6f}")
print(f"  中位数GTFP: {df['标准化GTFP'].median():.6f}")
print(f"  GTFP范围: [{df['标准化GTFP'].min():.6f}, {df['标准化GTFP'].max():.6f}]")
print(f"  唯一值: {df['标准化GTFP'].nunique():,}")

print(f"\n📈 效率等级:")
if '效率等级' in df.columns:
    for level, count in df['效率等级'].value_counts().sort_index().items():
        percent = count / len(df) * 100
        print(f"  {level}: {count:,} ({percent:.1f}%)")

print(f"\n📁 生成文件:")
files_to_check = [
    "GTFP_完整结果.csv",
    "GTFP_前1000名.csv",
    "GTFP_分布图.png",
    "GTFP_等级分布.png",
    "GTFP_前20名.png",
    "GTFP_箱线图.png",
    "GTFP_详细报告.txt"
]

for file_name in files_to_check:
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  ✓ {file_name} ({size_kb:.1f} KB)")
    else:
        print(f"  ✗ {file_name}")

print("\n" + "="*60)
print("所有计算已完成!")
print("按Enter键退出...")
input()
