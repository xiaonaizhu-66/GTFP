import pandas as pd

# 1. 写对文件的完整路径（根据你的文件位置）
# 比如文件在“Desktop\TFP-Data”里，完整路径是：
file_path = r"C:\Users\15535\Desktop\TFP-Data\FS_Combas.xlsx"

# 读取Excel（指定第2行为列名）
df = pd.read_excel(
    file_path,  # 这里用完整路径
    engine="openpyxl",
    header=1
)

# 2. 转换日期格式
df["统计截止日期"] = pd.to_datetime(df["统计截止日期"])

# 3. 筛选12月31日
df_filtered = df[
    (df["统计截止日期"].dt.month == 12) & 
    (df["统计截止日期"].dt.day == 31)
]

# 4. 去空行
df_filtered = df_filtered.dropna(how="all")

# 5. 保存（指定完整路径，避免找不到文件）
save_path = r"C:\Users\15535\Desktop\TFP-Data\FS_Combas_仅1231.xlsx"
df_filtered.to_excel(
    save_path,
    index=False,
    engine="openpyxl"
)
print("筛选后的数据行数：", len(df_filtered))  # 如果输出0，说明没有12-31的数据