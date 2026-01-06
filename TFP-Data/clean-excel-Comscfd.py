import pandas as pd
import os

# 设置文件路径
file_path = r'C:\Users\15535\Desktop\TFP-Data\FS_Comscfd.xlsx'

# 读取文件（根据实际文件格式选择）
try:
    # 尝试读取CSV格式（如果是制表符分隔）
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
except:
    try:
        # 尝试读取Excel格式
        df = pd.read_excel(file_path)
    except:
        # 如果是CSV逗号分隔
        df = pd.read_csv(file_path, encoding='utf-8')

print("原始数据形状:", df.shape)
print("原始列名:", df.columns.tolist())

# 确保日期列是字符串类型
df['Accper'] = df['Accper'].astype(str)

# 筛选12-31日的数据
df_filtered = df[df['Accper'].str.contains('12-31')].copy()

print(f"筛选12-31日后的数据量: {df_filtered.shape[0]}")

# 筛选报表类型为A类的数据
df_filtered = df_filtered[df_filtered['Typrep'].astype(str).str.contains('A', na=False)]

print(f"筛选A类报表后的数据量: {df_filtered.shape[0]}")

# 筛选公司，去除ST和*ST公司
# 假设ShortName列包含公司简称
df_filtered = df_filtered[~df_filtered['ShortName'].astype(str).str.contains('ST')]
df_filtered = df_filtered[~df_filtered['ShortName'].astype(str).str.contains('\*ST')]

print(f"去除ST/*ST公司后的数据量: {df_filtered.shape[0]}")

# 提取年份信息（可选）
df_filtered['Year'] = df_filtered['Accper'].str.extract(r'(\d{4})')

# 转换数值类型
df_filtered['Stkcd'] = df_filtered['Stkcd'].astype(str).str.strip()
df_filtered['C001014000'] = pd.to_numeric(df_filtered['C001014000'], errors='coerce')

# 重置索引
df_filtered = df_filtered.reset_index(drop=True)

# 显示前几行数据
print("\n处理后的数据预览:")
print(df_filtered[['Stkcd', 'ShortName', 'Accper', 'Typrep', 'C001014000']].head())

# 保存处理后的数据
output_path = r'C:\Users\15535\Desktop\TFP-Data\FS_Comscfd_processed.csv'
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n处理完成！数据已保存到: {output_path}")
print(f"最终数据形状: {df_filtered.shape}")

# 显示每年的数据分布
if 'Year' in df_filtered.columns:
    year_dist = df_filtered['Year'].value_counts().sort_index()
    print("\n每年数据分布:")
    for year, count in year_dist.items():
        print(f"{year}年: {count}条记录")