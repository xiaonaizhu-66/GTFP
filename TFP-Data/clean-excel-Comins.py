import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置文件路径
desktop_path = r"C:\Users\15535\Desktop\TFP-Data"
comins_file = os.path.join(desktop_path, "FS_Comins.xlsx")  # 利润表文件
combas_file = os.path.join(desktop_path, "FS_Combas_仅1231.xlsx")  # 资产负债表文件

# 1. 读取并处理Comins表（利润表）
def process_comins(file_path):
    """
    处理利润表数据
    1. 只保留A类报表
    2. 去除ST、*ST公司
    3. 只保留12月31日报表
    4. 转换单位为万元
    """
    print("正在读取利润表数据...")
    
    # 读取Excel文件，跳过前2行表头
    df = pd.read_excel(file_path, skiprows=2)
    
    # 重命名列（使用CSMAR标准列名）
    df.columns = ['证券代码', '证券简称', '统计截止日期', '报表类型', 'B001101000', 'B001216000']
    
    # 只保留年报数据（A类）
    df = df[df['报表类型'] == 'A'].copy()
    print(f"保留A类报表后，剩余{len(df)}条记录")
    
    # 去除ST、*ST公司
    st_keywords = ['ST', '*ST', 'SST', 'S*ST']
    mask = df['证券简称'].apply(lambda x: any(keyword in str(x) for keyword in st_keywords))
    df = df[~mask].copy()
    print(f"去除ST/*ST公司后，剩余{len(df)}条记录")
    
    # 将统计截止日期转换为datetime格式
    df['统计截止日期'] = pd.to_datetime(df['统计截止日期'])
    
    # 只保留12月31日的报表
    df = df[df['统计截止日期'].dt.month == 12].copy()
    df = df[df['统计截止日期'].dt.day == 31].copy()
    print(f"保留12月31日报表后，剩余{len(df)}条记录")
    
    # 提取年份
    df['Year'] = df['统计截止日期'].dt.year
    
    # 重命名指标列（更直观的名称）
    df = df.rename(columns={
        'B001101000': '营业收入',  # 营业收入
        'B001216000': '研发费用'  # 研发费用
    })
    
    # 转换单位：元 -> 万元（保留2位小数）
    for col in ['营业收入', '研发费用']:
        df[col] = df[col] / 10000
        df[col] = df[col].round(2)
    
    # 处理缺失值
    df['研发费用'] = df['研发费用'].fillna(0)  # 研发费用缺失值填充为0
    
    # 选择需要的列并重命名以与Combas表一致
    df = df.rename(columns={
        '证券代码': 'Stkcd',
        '证券简称': 'ShortName'
    })
    
    df = df[['Stkcd', 'ShortName', 'Year', '营业收入', '研发费用']]
    
    # 按证券代码和年份排序
    df = df.sort_values(['Stkcd', 'Year']).reset_index(drop=True)
    
    # 检查数据质量
    print(f"\n利润表数据质量检查:")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['Year'].min()}年 - {df['Year'].max()}年")
    print(f"公司数量: {df['Stkcd'].nunique()}")
    print(f"营业收入统计 - 均值: {df['营业收入'].mean():.2f}万元, 最大值: {df['营业收入'].max():.2f}万元, 最小值: {df['营业收入'].min():.2f}万元")
    print(f"研发费用统计 - 均值: {df['研发费用'].mean():.2f}万元, 最大值: {df['研发费用'].max():.2f}万元, 最小值: {df['研发费用'].min():.2f}万元")
    
    return df

# 2. 读取并处理Combas表（资产负债表）
def process_combas(file_path):
    """
    处理资产负债表数据
    """
    print("\n正在读取资产负债表数据...")
    
    try:
        # 读取完整数据
        df = pd.read_excel(file_path)
        print(f"读取到{len(df)}条资产负债表记录")
        
        # 重命名列以保持一致性
        df = df.rename(columns={
            '证券代码': 'Stkcd',
            '证券简称': 'ShortName',
            '统计截止日期': 'Accper',
            '报表类型': 'Typrep'
        })
        
        # 只保留年报数据（A类）
        df = df[df['Typrep'] == 'A'].copy()
        print(f"保留A类报表后，剩余{len(df)}条记录")
        
        # 处理日期
        df['Accper'] = pd.to_datetime(df['Accper'])
        df['Year'] = df['Accper'].dt.year
        
        # 只保留12月31日的报表
        df = df[df['Accper'].dt.month == 12].copy()
        df = df[df['Accper'].dt.day == 31].copy()
        print(f"保留12月31日报表后，剩余{len(df)}条记录")
        
        # 转换单位：元 -> 万元（保留2位小数）
        for col in ['固定资产净额', '资产总计']:
            if col in df.columns:
                df[col] = df[col] / 10000
                df[col] = df[col].round(2)
                print(f"已将 '{col}' 列单位转换为万元")
        
        # 去除ST、*ST公司（与利润表保持一致）
        st_keywords = ['ST', '*ST', 'SST', 'S*ST']
        mask = df['ShortName'].apply(lambda x: any(keyword in str(x) for keyword in st_keywords))
        df = df[~mask].copy()
        print(f"去除ST/*ST公司后，剩余{len(df)}条记录")
        
        # 重命名指标列
        df = df.rename(columns={
            '固定资产净额': '固定资产',
            '资产总计': '总资产'
        })
        
        # 选择需要的列
        df = df[['Stkcd', 'ShortName', 'Year', '固定资产', '总资产']]
        
        # 按证券代码和年份排序
        df = df.sort_values(['Stkcd', 'Year']).reset_index(drop=True)
        
        print(f"\n资产负债表数据质量检查:")
        print(f"总记录数: {len(df)}")
        print(f"时间范围: {df['Year'].min()}年 - {df['Year'].max()}年")
        print(f"公司数量: {df['Stkcd'].nunique()}")
        print(f"固定资产统计 - 均值: {df['固定资产'].mean():.2f}万元, 最大值: {df['固定资产'].max():.2f}万元, 最小值: {df['固定资产'].min():.2f}万元")
        print(f"总资产统计 - 均值: {df['总资产'].mean():.2f}万元, 最大值: {df['总资产'].max():.2f}万元, 最小值: {df['总资产'].min():.2f}万元")
        
        return df
        
    except Exception as e:
        print(f"读取资产负债表出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 3. 合并两个表格
def merge_comins_combas(df_comins, df_combas):
    """
    将利润表数据与资产负债表数据进行合并
    """
    print("\n正在合并利润表和资产负债表数据...")
    
    if df_combas is None:
        print("资产负债表数据为空，仅返回利润表数据")
        return df_comins
    
    # 检查两个数据集的列
    print(f"利润表列名: {list(df_comins.columns)}")
    print(f"资产负债表列名: {list(df_combas.columns)}")
    
    # 合并数据（内连接，只保留两个表都有的记录）
    merged_df = pd.merge(
        df_comins,
        df_combas,
        on=['Stkcd', 'Year', 'ShortName'],
        how='inner'
    )
    
    print(f"合并后记录数: {len(merged_df)}")
    print(f"合并成功率: {len(merged_df)/min(len(df_comins), len(df_combas))*100:.2f}%")
    
    # 检查是否有重复
    duplicates = merged_df.duplicated(subset=['Stkcd', 'Year']).sum()
    if duplicates > 0:
        print(f"警告: 发现{duplicates}条重复记录")
        merged_df = merged_df.drop_duplicates(subset=['Stkcd', 'Year'], keep='first')
        print(f"去重后记录数: {len(merged_df)}")
    
    return merged_df

# 4. 保存处理后的数据
def save_processed_data(df_comins, df_combas, merged_df, output_path):
    """
    保存处理后的数据到Excel文件
    """
    print(f"\n正在保存处理后的数据...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存到Excel，包含多个工作表
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 保存处理后的利润表
        df_comins.to_excel(writer, sheet_name='Processed_Comins', index=False)
        print(f"  √ 已保存利润表到 'Processed_Comins' 工作表")
        
        # 保存资产负债表（如果存在）
        if df_combas is not None:
            df_combas.to_excel(writer, sheet_name='Processed_Combas', index=False)
            print(f"  √ 已保存资产负债表到 'Processed_Combas' 工作表")
        
        # 保存合并后的完整数据
        merged_df.to_excel(writer, sheet_name='Merged_Data', index=False)
        print(f"  √ 已保存合并数据到 'Merged_Data' 工作表")
        
        # 添加数据摘要
        summary_data = {
            '统计项': ['总记录数', '公司数量', '时间范围', '数据年份数', 
                     '营业收入均值(万元)', '研发费用均值(万元)', 
                     '固定资产均值(万元)', '总资产均值(万元)'],
            '数值': [
                len(merged_df),
                merged_df['Stkcd'].nunique(),
                f"{merged_df['Year'].min()}-{merged_df['Year'].max()}",
                merged_df['Year'].nunique(),
                f"{merged_df['营业收入'].mean():.2f}",
                f"{merged_df['研发费用'].mean():.2f}",
                f"{merged_df['固定资产'].mean():.2f}",
                f"{merged_df['总资产'].mean():.2f}"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Data_Summary', index=False)
        print(f"  √ 已保存数据摘要到 'Data_Summary' 工作表")
        
        # 添加列说明
        col_descriptions = [
            ['Stkcd', '证券代码'],
            ['ShortName', '证券简称'],
            ['Year', '年份'],
            ['营业收入', '营业收入(万元)'],
            ['研发费用', '研发费用(万元)'],
            ['固定资产', '固定资产净额(万元)'],
            ['总资产', '资产总计(万元)']
        ]
        pd.DataFrame(col_descriptions, columns=['列名', '说明']).to_excel(writer, sheet_name='Column_Description', index=False)
        print(f"  √ 已保存列说明到 'Column_Description' 工作表")
        
        # 添加各年份数据分布
        year_dist = merged_df['Year'].value_counts().sort_index().reset_index()
        year_dist.columns = ['年份', '公司数量']
        year_dist.to_excel(writer, sheet_name='Year_Distribution', index=False)
        print(f"  √ 已保存年份分布到 'Year_Distribution' 工作表")
    
    return output_path

# 5. 主函数
def main():
    print("=" * 60)
    print("利润表与资产负债表数据处理与合并程序")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(comins_file):
        print(f"错误: 未找到利润表文件: {comins_file}")
        return
    
    if not os.path.exists(combas_file):
        print(f"错误: 未找到资产负债表文件: {combas_file}")
        return
    
    # 处理利润表
    print("\n" + "=" * 60)
    df_comins = process_comins(comins_file)
    
    # 处理资产负债表
    print("\n" + "=" * 60)
    df_combas = process_combas(combas_file)
    
    # 合并两个表格
    print("\n" + "=" * 60)
    merged_df = merge_comins_combas(df_comins, df_combas)
    
    # 保存处理后的数据
    print("\n" + "=" * 60)
    output_file = os.path.join(desktop_path, "Processed_Merged_Data.xlsx")
    save_processed_data(df_comins, df_combas, merged_df, output_file)
    
    # 显示最终结果摘要
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"输出文件: {output_file}")
    print(f"\n最终数据集信息:")
    print(f"  总记录数: {len(merged_df)}")
    print(f"  公司数量: {merged_df['Stkcd'].nunique()}")
    print(f"  时间范围: {merged_df['Year'].min()}年 - {merged_df['Year'].max()}年")
    print(f"  数据列数: {len(merged_df.columns)}")
    
    # 显示前5行数据预览
    print(f"\n数据预览（前5行）:")
    print(merged_df.head().to_string())
    
    # 显示各年份数据量
    print(f"\n各年份数据分布:")
    year_counts = merged_df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}年: {count}条记录")
    
    # 显示数据列信息
    print(f"\n数据列信息:")
    for col in merged_df.columns:
        dtype = merged_df[col].dtype
        non_null = merged_df[col].notna().sum()
        unique_count = merged_df[col].nunique() if dtype == 'object' else 'N/A'
        print(f"  {col:10s} | 类型: {dtype:10s} | 非空值: {non_null:6d}/{len(merged_df):6d} | 唯一值: {unique_count}")

# 6. 运行主程序
if __name__ == "__main__":
    main()