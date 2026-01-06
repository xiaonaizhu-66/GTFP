import pandas as pd

# 1. 读取所有表格
print("正在读取资产负债表...")
balance = pd.read_excel('FS_Combas_仅1231.xlsx')  # 资产负债表
print(f"资产负债表读取完成，共 {len(balance)} 行")

print("正在读取利润表...")
income = pd.read_excel('FS_Comins.xlsx')          # 利润表
income = income.rename(columns={'Stkcd': '证券代码', 'Accper': '统计截止日期', 'Typrep': '报表类型', 'B001101000': '营业收入'})
print(f"利润表读取完成，共 {len(income)} 行")

print("正在读取现金流量表...")
cashflow = pd.read_excel('FS_Comscfd.xlsx')       # 现金流量表
cashflow = cashflow.rename(columns={'Stkcd': '证券代码', 'Accper': '统计截止日期', 'Typrep': '报表类型', 'C001014000': '购买商品、接受劳务支付的现金'})
print(f"现金流量表读取完成，共 {len(cashflow)} 行")

print("正在读取员工表...")
# staff = pd.read_excel('STK_CompanyStaff.xlsx')         # 员工表
# print(f"员工表读取完成，共 {len(staff)} 行")
staff = None  # 暂时跳过员工表

# 2. 统一预处理函数
def preprocess_table(df, table_name):
    """统一预处理：筛选年报，提取年份"""
    df = df[df['报表类型'] == 'A']  # 只保留年报
    df['年份'] = pd.to_datetime(df['统计截止日期']).dt.year
    df = df.drop(columns=['报表类型', '统计截止日期'])  # 删除不需要的列
    return df

balance = preprocess_table(balance, 'balance')
income = preprocess_table(income, 'income')
cashflow = preprocess_table(cashflow, 'cashflow')
# staff = preprocess_table(staff, 'staff')

# 3. 员工表特殊处理（按证券代码+年份汇总员工数）
# staff_agg = staff.groupby(['证券代码', '年份'])['数量'].sum().reset_index()
# staff_agg = staff_agg.rename(columns={'数量': '员工总数'})
staff_agg = None

# 4. 逐步合并（左连接，以资产负债表为基准）
merged_df = balance.merge(
    income[['证券代码', '年份', 'B001101000']],  # 营业收入
    on=['证券代码', '年份'],
    how='left'
).merge(
    cashflow[['证券代码', '年份', 'C001014000']],  # 购买商品、接受劳务支付的现金
    on=['证券代码', '年份'],
    how='left'
)# .merge(
#     staff_agg,
#     on=['证券代码', '年份'],
#     how='left'
# )

# 5. 保存合并结果
print(f"合并后的数据行数: {len(merged_df)}")
print(f"合并后的数据列数: {len(merged_df.columns)}")
print("列名:", list(merged_df.columns))
merged_df.to_excel('制造业上市公司整合数据.xlsx', index=False)
print("数据整合完成，已保存为 '制造业上市公司整合数据.xlsx'")