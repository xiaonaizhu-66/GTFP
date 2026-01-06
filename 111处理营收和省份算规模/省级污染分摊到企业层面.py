import pandas as pd
import numpy as np

# ==========================================
# 1. 读取数据
# ==========================================
# 请修改为你的实际文件路径
file_firm_path = 'firm_year_with_province_scale_share.xlsx'  # 你的公司占比表
file_prov_path = 'province_so2_nox_2019_2024.xlsx'          # 你的省份污染表

# 读取 Excel
# 注意：根据你的截图，公司表中省份列名是 'province'，年份是 'year'
# 省份表中列名是 'province', 'year', 'so2', 'nox'
df_firm = pd.read_excel(file_firm_path)
df_prov = pd.read_excel(file_prov_path)

print("公司数据行数:", len(df_firm))
print("省份数据行数:", len(df_prov))

# ==========================================
# 2. 数据清洗与格式统一
# ==========================================

# 确保年份都是整数类型，防止一个是 '2019' (str) 一个是 2019 (int) 导致匹配失败
df_firm['year'] = df_firm['year'].astype(int)
df_prov['year'] = df_prov['year'].astype(int)

# 去除省份名称中可能存在的空格
df_firm['province'] = df_firm['province'].astype(str).str.strip()
df_prov['province'] = df_prov['province'].astype(str).str.strip()

# ==========================================
# 3. 数据合并 (Merge)
# ==========================================
# 使用 left join，保留所有公司行，把省份数据贴过去
df_merge = pd.merge(df_firm, df_prov, on=['province', 'year'], how='left')

# 检查匹配情况：如果匹配后污染数据是 NaN，说明该省份/年份在宏观表中没找到
missing_data = df_merge[df_merge['so2'].isna()]
if len(missing_data) > 0:
    print(f"警告：有 {len(missing_data)} 行数据未匹配到省份排放量。")
    print("可能是省份名称不一致（如'北京' vs '北京市'）或年份缺失。")
    # 如果想看是哪些没匹配上，取消下面这行的注释
    # print(missing_data[['stkcd', 'province', 'year']].head())
else:
    print("完美！所有公司都匹配到了对应的省份污染数据。")

# ==========================================
# 4. 核心计算：推算企业级污染排放
# ==========================================
# 公式：企业排放 = 省份总排放 * (企业营收 / 省份总营收)
# 你的表中已有 share = province_scale_share

df_merge['Company_SO2'] = df_merge['so2'] * df_merge['province_scale_share']
df_merge['Company_NOx'] = df_merge['nox'] * df_merge['province_scale_share']

# ==========================================
# 5. 整合最终用于 GTFP 计算的宽表
# ==========================================

# 注意：计算 GTFP 还需要 资本(K) 和 劳动(L)
# 你的截图中 firm 表只有“营业收入”和“研发费用”。
# 如果 K (固定资产) 和 L (员工人数) 不在这个表里，你需要在这里把它们合进来。
# 假设你之前有一个 cleaned_data 包含 K and L，你需要 merge 一下。
# 这里我先假设你稍后会处理 K 和 L，先保存目前的污染计算结果。

# 选取需要的列
output_columns = [
    'stkcd', 'year', 'province', 
    '营业收入',               # 期望产出 (Y)
    'province_scale_share', 
    'Company_SO2',            # 非期望产出 1 (Bad Output)
    'Company_NOx'             # 非期望产出 2 (Bad Output)
]

# 如果你的表里其实已经有固定资产和人数了，记得把列名加进上面的 list

df_final = df_merge[output_columns].copy()

# 处理缺失值和无穷大（DEA模型对数据质量要求很高）
df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()

print(f"最终用于计算的数据行数: {len(df_final)}")

# ==========================================
# 6. 保存结果
# ==========================================
output_file = 'GTFP_Ready_Data_Step1.xlsx'
df_final.to_excel(output_file, index=False)
print(f"文件已保存为: {output_file}")
print("下一步：请检查该文件中是否包含 资本(K) 和 劳动(L)，若不包含，需进行第二次合并。")