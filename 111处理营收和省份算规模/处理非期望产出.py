import pandas as pd

# 企业层面数据（含 scale_share）
firm = pd.read_excel(
    "firm_year_with_province.xlsx"
)

# 省级污染数据
pollution = pd.read_excel(
    "2020年工业废水排放量和二氧化硫和氮氧化物（全省）.xlsx"
)
# 去空格
firm['province'] = firm['province'].str.strip()
pollution['province'] = pollution['province'].str.strip()

# 确保年份是 int
firm['year'] = firm['year'].astype(int)
pollution['year'] = pollution['year'].astype(int)
df = firm.merge(
    pollution,
    on=['province', 'year'],
    how='left'
)
df['so2_firm'] = df['SO2'] * df['scale_share']
df['wastewater_firm'] = df['工业废水'] * df['scale_share']
df['dust_firm'] = df['烟粉尘'] * df['scale_share']
import numpy as np

df['ln_so2'] = np.log(df['so2_firm'] + 1)
df['ln_wastewater'] = np.log(df['wastewater_firm'] + 1)
df['ln_dust'] = np.log(df['dust_firm'] + 1)
df.to_excel(
    "/mnt/data/firm_undesirable_output.xlsx",
    index=False
)
