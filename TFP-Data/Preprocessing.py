import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import os

# ===============================
# 0. 设置工作目录（防止找不到文件）
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ===============================
# 1. 缩尾函数
# ===============================
def winsorize_series(s, limits=(0.01, 0.01)):
    return pd.Series(winsorize(s, limits=limits), index=s.index)

# ===============================
# 2. 读取数字化指数
# ===============================
print("读取数字化指数...")
df_digi = pd.read_excel("Digital_Index.xlsx")

df_digi = df_digi[[
    'symbol', 'sgnyear', 'digitaltransindex',
    'equitynatureid', 'province'
]]

df_digi.rename(columns={
    'symbol': 'Stkcd',
    'sgnyear': 'Year',
    'digitaltransindex': 'Digi',
    'equitynatureid': 'Ownership',
    'province': 'Province'
}, inplace=True)

df_digi['Stkcd'] = pd.to_numeric(df_digi['Stkcd'], errors='coerce')
df_digi['Year'] = pd.to_numeric(df_digi['Year'], errors='coerce')

# ===============================
# 3. 读取资产负债表
# ===============================
print("读取资产负债表...")
df_bal = pd.read_excel("FS_Combas.xlsx")

df_bal = df_bal[['Stkcd', 'Accper', 'A001212000', 'A001000000']]
df_bal.rename(columns={
    'Accper': 'Date',
    'A001212000': 'NetFixedAsset',
    'A001000000': 'Asset'
}, inplace=True)

# ===============================
# 4. 读取利润表
# ===============================
print("读取利润表...")
df_inc = pd.read_excel("FS_Comins.xlsx")

df_inc = df_inc[['Stkcd', 'Accper', 'B001101000']]
df_inc.rename(columns={
    'Accper': 'Date',
    'B001101000': 'Revenue'
}, inplace=True)

# ===============================
# 5. 读取员工表
# ===============================
print("读取员工表...")
df_staff = pd.read_excel("STK_CompanyStaff.xlsx")

df_staff = df_staff[
    (df_staff['EmployStructure'] == '总人数') &
    (df_staff['EmployDetail'].str.contains('在职', na=False))
]

df_staff = df_staff[['Symbol', 'EndDate', 'Amount']]
df_staff.rename(columns={
    'Symbol': 'Stkcd',
    'EndDate': 'Date',
    'Amount': 'Staff'
}, inplace=True)

# ===============================
# 6. 清洗日期，提取 Year
# ===============================
def clean_df(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].dt.month == 12]
    df['Year'] = df['Date'].dt.year
    df.drop(columns=['Date'], inplace=True)
    return df

df_bal = clean_df(df_bal)
df_inc = clean_df(df_inc)
df_staff = clean_df(df_staff)

# ===============================
# 7. 合并数据
# ===============================
print("合并数据...")

df = df_bal.merge(df_inc, on=['Stkcd', 'Year'], how='inner')
print("合并资产负债表 + 利润表：", len(df))

df = df.merge(df_staff, on=['Stkcd', 'Year'], how='inner')
print("加入员工数据后：", len(df))

df = df.merge(df_digi, on=['Stkcd', 'Year'], how='inner')
print("加入数字化指数后：", len(df))

# ===============================
# 8. 构造变量
# ===============================
print("构造变量...")

# 强制转数值
for col in ['Revenue', 'NetFixedAsset', 'Staff', 'Asset', 'Digi']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['lnY'] = np.log(df['Revenue'] + 1)
df['lnK'] = np.log(df['NetFixedAsset'] + 1)
df['lnL'] = np.log(df['Staff'] + 1)
df['Size'] = np.log(df['Asset'] + 1)

df['SOE'] = np.where(df['Ownership'] == 1, 1, 0)

# ===============================
# 9. 清洗 + 缩尾
# ===============================
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(subset=['lnY', 'lnK', 'lnL', 'Digi', 'Size'], inplace=True)

for col in ['lnY', 'lnK', 'lnL', 'Digi', 'Size']:
    df[col] = winsorize_series(df[col])

# ===============================
# 10. 保存结果
# ===============================
print("保存数据...")

# Excel：完整版（含中文）
df.to_excel("Final_Regression_Data.xlsx", index=False)

# Stata：只保留回归需要的【纯数值变量】
stata_vars = [
    'Stkcd', 'Year',
    'lnY', 'lnK', 'lnL',
    'Digi', 'Size', 'SOE'
]

df[stata_vars].to_stata(
    "Final_Regression_Data.dta",
    write_index=False,
    version=117
)

print("✅ 全部完成！")
print("生成文件：")
print(" - Final_Regression_Data.xlsx")
print(" - Final_Regression_Data.dta")
