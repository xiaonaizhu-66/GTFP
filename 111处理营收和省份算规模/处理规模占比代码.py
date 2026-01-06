import pandas as pd

# 读取你刚刚生成的面板数据
df = pd.read_excel("firm_year_with_province.xlsx")

print(df.head())
# 如果是日期，提取年份
df['year'] = pd.to_datetime(df['统计截止日期']).dt.year
# 计算每个 省份-年份 的总营业收入
province_year_total = (
    df
    .groupby(['province', 'year'])['营业收入']
    .sum()
    .reset_index()
    .rename(columns={'营业收入': 'province_year_total_revenue'})
)

print(province_year_total.head())
df = df.merge(
    province_year_total,
    on=['province', 'year'],
    how='left'
)
df['province_scale_share'] = (
    df['营业收入'] / df['province_year_total_revenue']
)
check = (
    df.groupby(['province', 'year'])['province_scale_share']
      .sum()
      .reset_index()
)

print(check.head())
df.to_excel(
    "firm_year_with_province_scale_share.xlsx",
    index=False
)

print("✅ 已生成：firm_year_with_province_scale_share.xlsx")
