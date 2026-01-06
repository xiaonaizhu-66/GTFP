import pandas as pd
# 读取表1：公司-省份映射表
df_province = pd.read_excel("省份表.xlsx")

# 只保留关键变量
df_province = df_province[['symbol', 'province']]

# 统一证券代码命名
df_province = df_province.rename(columns={'symbol': 'stkcd'})

# 证券代码统一为6位字符串
df_province['stkcd'] = df_province['stkcd'].astype(str).str.zfill(6)

# 去重（一家公司只保留一个省份）
df_province = df_province.drop_duplicates(subset='stkcd')

print("【公司-省份表】前5行：")
print(df_province.head())
print("公司数量：", df_province['stkcd'].nunique())
# 读取表2：公司-年度数据
df_firm_year = pd.read_excel("处理后的营业收入.xlsx")

# 重命名证券代码列（按你的表头）
df_firm_year = df_firm_year.rename(columns={'证券代码': 'stkcd'})

# 证券代码统一格式
df_firm_year['stkcd'] = df_firm_year['stkcd'].astype(str).str.zfill(6)

print("【公司-年度表】前5行：")
print(df_firm_year.head())
print("公司数量：", df_firm_year['stkcd'].nunique())
df_panel = df_firm_year.merge(
    df_province,
    on='stkcd',
    how='left'   # 左连接：不丢任何公司-年度观测
)
# 检查省份是否成功匹配
missing_province = df_panel[df_panel['province'].isna()]['stkcd'].unique()

print("未匹配到省份的公司数量：", len(missing_province))

if len(missing_province) > 0:
    print("示例未匹配公司（前10个）：")
    print(missing_province[:10])
else:
    print("✅ 所有公司均成功匹配省份")
df_panel.to_excel("firm_year_with_province.xlsx", index=False)

print("🎉 数据处理完成，文件已保存：firm_year_with_province.xlsx")
