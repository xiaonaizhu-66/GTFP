import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\15535\Desktop\111处理营收和省份算规模")

files = {
    2019: "2019年各省直辖市自治区废水二氧化硫氮氧化物排放量汇总.xlsx",
    2020: "2020年工业废水排放量和二氧化硫和氮氧化物.xlsx",
    2021: "2021年工业颗粒物排放量和二氧化硫和氮氧化物.xlsx",
    2022: "2022年工业颗粒物排放量和二氧化硫和氮氧化物.xlsx",
    2023: "2023年各省直辖市自治区工业颗粒物二氧化硫氮氧化物排放量汇总.xlsx",
    2024: "2024年工业颗粒物排放量和二氧化硫和氮氧化物.xlsx",
}

all_years = []

for year, fname in files.items():

    file_path = data_dir / fname
    if not file_path.exists():
        print(f"❌ 文件不存在：{file_path}")
        continue

    # ✅ ① 这里才创建 df
    df = pd.read_excel(file_path)

    # ✅ ② 列名清洗【必须在这里】
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('（', '(')
        .str.replace('）', ')')
    )

    # ✅ ③ 重命名
    df = df.rename(columns={
        '省份': 'province',
        '工业二氧化硫排放量(吨)': 'so2',
        '工业氮氧化物排放量(吨)': 'nox'
    })

    # ✅ ④ 加年份
    df['year'] = year

    # ✅ ⑤ 只保留需要的列
    df = df[['province', 'year', 'so2', 'nox']]

    all_years.append(df)

# ✅ ⑥ 循环结束后再 concat
pollution = pd.concat(all_years, ignore_index=True)

pollution.replace(['-', '—', '…', ''], pd.NA, inplace=True)
pollution['so2'] = pd.to_numeric(pollution['so2'], errors='coerce')
pollution['nox'] = pd.to_numeric(pollution['nox'], errors='coerce')

pollution = pollution[~pollution['province'].str.contains('全国', na=False)]

pollution.to_excel("province_so2_nox_2019_2024.xlsx", index=False)

print("✅ 处理完成")
