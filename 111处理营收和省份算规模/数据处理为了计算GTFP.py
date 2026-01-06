import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 设置文件路径（与之前保持一致）
desktop_path = r"C:\Users\15535\Desktop\111处理营收和省份算规模"
merged_file = os.path.join(desktop_path, "Final_Data_with_Environment.xlsx")  # 你的综合表

def prepare_sbm_dea_so2_nox_only():
    """
    使用SO₂和NOx两个非期望产出准备SBM-DEA输入数据
    """
    print("=" * 60)
    print("SBM-DEA输入准备（SO₂ + NOx 两个非期望产出）")
    print("=" * 60)
    
    # 1. 读取数据
    print("正在读取数据...")
    try:
        df = pd.read_excel(merged_file)
        print(f"成功读取数据，形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取失败: {e}")
        return
    
    # 2. 检查必要列
    print("\n检查必要数据列...")
    
    # 识别关键列
    column_mapping = {}
    
    # 证券代码
    code_cols = [col for col in df.columns if '代码' in str(col) or 'stkcd' in str(col).lower() or 'symbol' in str(col).lower()]
    if code_cols:
        column_mapping[code_cols[0]] = 'Stkcd'
    
    # 年份
    year_cols = [col for col in df.columns if 'year' in str(col).lower() or 'Year' in df.columns]
    if year_cols:
        column_mapping[year_cols[0]] = 'Year'
    elif '统计截止日期' in df.columns:
        df['Year'] = pd.to_datetime(df['统计截止日期']).dt.year
    
    # 营业收入
    revenue_cols = [col for col in df.columns if '营业' in str(col) or 'revenue' in str(col).lower()]
    if revenue_cols:
        column_mapping[revenue_cols[0]] = '营业收入'
    
    # 固定资产
    fixed_cols = [col for col in df.columns if '固定' in str(col) or 'fixed' in str(col).lower()]
    if fixed_cols:
        column_mapping[fixed_cols[0]] = '固定资产'
    
    # 员工数
    emp_cols = [col for col in df.columns if '员工' in str(col) or 'employ' in str(col).lower()]
    if emp_cols:
        column_mapping[emp_cols[0]] = '员工总数'
    
    # 购买商品劳务现金
    cash_cols = [col for col in df.columns if '劳务' in str(col) or '商品' in str(col) or 'cash' in str(col).lower()]
    if cash_cols:
        column_mapping[cash_cols[0]] = '购买商品劳务现金'
    
    # 环境数据
    if 'so2' in df.columns:
        column_mapping['so2'] = 'so2'
    if 'nox' in df.columns:
        column_mapping['nox'] = 'nox'
    if 'province_scale_share' in df.columns:
        column_mapping['province_scale_share'] = 'province_scale_share'
    
    # 应用列名映射
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"已标准化列名: {column_mapping}")
    
    # 3. 确保单位正确（转换为万元）
    print("\n统一数据单位...")
    
    # 检查营收单位（如果数值很大可能是元）
    if '营业收入' in df.columns and df['营业收入'].max() > 1000000:
        df['营业收入'] = df['营业收入'] / 10000  # 元转万元
        print("  营业收入: 元 → 万元")
    
    if '固定资产' in df.columns and df['固定资产'].max() > 1000000:
        df['固定资产'] = df['固定资产'] / 10000  # 元转万元
        print("  固定资产: 元 → 万元")
    
    if '购买商品劳务现金' in df.columns and df['购买商品劳务现金'].max() > 1000000:
        df['购买商品劳务现金'] = df['购买商品劳务现金'] / 10000  # 元转万元
        print("  购买商品劳务现金: 元 → 万元")
    
    # 4. 计算企业级排放量
    print("\n计算企业级污染物排放量...")
    
    if 'so2' in df.columns and 'province_scale_share' in df.columns:
        df['企业SO2排放量'] = df['so2'] * df['province_scale_share']
        print(f"  计算SO2排放量完成")
    elif '企业SO2排放量' in df.columns:
        print(f"  使用已有的SO2排放量")
    else:
        print(f"  ⚠️ 无法计算SO2排放量")
        df['企业SO2排放量'] = 0
    
    if 'nox' in df.columns and 'province_scale_share' in df.columns:
        df['企业NOx排放量'] = df['nox'] * df['province_scale_share']
        print(f"  计算NOx排放量完成")
    elif '企业NOx排放量' in df.columns:
        print(f"  使用已有的NOx排放量")
    else:
        print(f"  ⚠️ 无法计算NOx排放量")
        df['企业NOx排放量'] = 0
    
    # 5. 创建SBM-DEA输入数据
    print("\n创建SBM-DEA输入数据结构...")
    
    sbm_data = pd.DataFrame()
    
    # DMU标识
    if 'Stkcd' in df.columns and 'Year' in df.columns:
        sbm_data['DMU'] = df['Stkcd'].astype(str) + '_' + df['Year'].astype(str)
        print(f"  DMU标识: 公司代码_年份")
    else:
        print(f"  ⚠️ 无法创建DMU标识，使用序号")
        sbm_data['DMU'] = ['DMU_' + str(i) for i in range(len(df))]
    
    # 投入指标（3个）
    if '固定资产' in df.columns:
        sbm_data['投入_资本'] = df['固定资产']
        print(f"  投入1: 资本（固定资产）")
    else:
        sbm_data['投入_资本'] = 0
        print(f"  ⚠️ 缺少资本投入数据")
    
    if '员工总数' in df.columns:
        sbm_data['投入_劳动'] = df['员工总数']
        print(f"  投入2: 劳动（员工总数）")
    else:
        sbm_data['投入_劳动'] = 0
        print(f"  ⚠️ 缺少劳动投入数据")
    
    if '购买商品劳务现金' in df.columns:
        sbm_data['投入_能源'] = df['购买商品劳务现金']
        print(f"  投入3: 能源（购买商品劳务现金代理）")
    else:
        sbm_data['投入_能源'] = 0
        print(f"  ⚠️ 缺少能源投入数据")
    
    # 期望产出（1个）
    if '营业收入' in df.columns:
        sbm_data['产出_营收'] = df['营业收入']
        print(f"  期望产出: 营业收入")
    else:
        sbm_data['产出_营收'] = 0
        print(f"  ⚠️ 缺少期望产出数据")
    
    # 非期望产出（2个）- 修改的关键部分！
    # SO₂排放
    if '企业SO2排放量' in df.columns:
        sbm_data['非期望_SO2'] = df['企业SO2排放量']
        print(f"  非期望产出1: SO₂排放")
    else:
        sbm_data['非期望_SO2'] = 0
        print(f"  ⚠️ 缺少SO₂排放数据")
    
    # NOx排放
    if '企业NOx排放量' in df.columns:
        sbm_data['非期望_NOx'] = df['企业NOx排放量']
        print(f"  非期望产出2: NOx排放")
    else:
        sbm_data['非期望_NOx'] = 0
        print(f"  ⚠️ 缺少NOx排放数据")
    
    # 6. 数据清洗和质量检查
    print("\n数据清洗和质量检查...")
    
    # 处理缺失值
    for col in sbm_data.columns:
        if col != 'DMU':
            non_null = sbm_data[col].notna().sum()
            if non_null < len(sbm_data):
                median_val = sbm_data[col].median()
                sbm_data[col] = sbm_data[col].fillna(median_val)
                print(f"  {col}: 填充 {len(sbm_data)-non_null} 个缺失值")
    
    # 处理零值和负值
    for col in sbm_data.columns:
        if col != 'DMU':
            zero_count = (sbm_data[col] == 0).sum()
            if zero_count > 0:
                col_min = sbm_data[col][sbm_data[col] > 0].min() if (sbm_data[col] > 0).any() else 0.001
                sbm_data[col] = sbm_data[col].replace(0, col_min / 10)
                print(f"  {col}: 替换 {zero_count} 个零值为小正数")
            
            neg_count = (sbm_data[col] < 0).sum()
            if neg_count > 0:
                sbm_data[col] = sbm_data[col].abs()
                print(f"  {col}: 取绝对值 {neg_count} 个负值")
    
    # 7. 保存数据
    print("\n保存SBM-DEA输入文件...")
    
    # CSV格式（MaxDEA推荐）
    csv_file = os.path.join(desktop_path, "SBM_DEA_Input_SO2_NOx_Only.csv")
    sbm_data.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV文件: {csv_file}")
    
    # Excel格式（方便查看）
    excel_file = os.path.join(desktop_path, "SBM_DEA_Input_SO2_NOx_Only.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 数据表
        sbm_data.to_excel(writer, sheet_name='DEA_Input_Data', index=False)
        
        # 变量说明表
        var_desc = pd.DataFrame({
            '变量类型': ['标识变量', '投入指标', '投入指标', '投入指标', 
                      '期望产出', '非期望产出', '非期望产出'],
            '变量名称': ['DMU', '投入_资本', '投入_劳动', '投入_能源',
                      '产出_营收', '非期望_SO2', '非期望_NOx'],
            '数据单位': ['文本', '万元', '人', '万元', '万元', '吨', '吨'],
            'MaxDEA设置': ['不选', 'Input', 'Input', 'Input', 
                        'Desirable Output', 'Undesirable Output', 'Undesirable Output'],
            '数据来源': ['公司代码+年份', '固定资产', '员工总数', '购买商品劳务现金',
                      '营业收入', 'SO₂排放量×比例', 'NOx排放量×比例']
        })
        var_desc.to_excel(writer, sheet_name='变量说明', index=False)
        
        # 数据统计表
        stats_data = []
        for col in sbm_data.columns:
            if col != 'DMU':
                stats_data.append({
                    '变量': col,
                    '均值': sbm_data[col].mean(),
                    '标准差': sbm_data[col].std(),
                    '最小值': sbm_data[col].min(),
                    '中位数': sbm_data[col].median(),
                    '最大值': sbm_data[col].max(),
                    '非空值': sbm_data[col].notna().sum(),
                    '零值数': (sbm_data[col] <= 0.001).sum()
                })
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='数据统计', index=False)
    
    print(f"  ✅ Excel文件: {excel_file}")
    
    # 8. 显示结果摘要
    print("\n" + "=" * 60)
    print("数据准备完成摘要")
    print("=" * 60)
    print(f"总DMU数量: {len(sbm_data)}")
    print(f"投入指标: 3个（资本、劳动、能源）")
    print(f"期望产出: 1个（营业收入）")
    print(f"非期望产出: 2个（SO₂、NOx）")
    
    print("\n前5行数据预览:")
    print(sbm_data.head().to_string())
    
    print("\n" + "=" * 60)
    print("MaxDEA使用说明")
    print("=" * 60)
    print("1. 打开MaxDEA软件")
    print(f"2. File → New Project → Import from File → 选择: {csv_file}")
    print("3. 变量设置:")
    print("   - DMU: 不选择")
    print("   - 投入_资本: Input")
    print("   - 投入_劳动: Input")
    print("   - 投入_能源: Input")
    print("   - 产出_营收: Desirable Output")
    print("   - 非期望_SO2: Undesirable Output")
    print("   - 非期望_NOx: Undesirable Output")
    print("4. 模型设置:")
    print("   Model: SBM")
    print("   Orientation: Non-oriented")
    print("   RTS: Variable (VRS)")
    print("   Undesirable Outputs: Include")
    print("5. 点击'Solve'运行")
    
    return sbm_data

def main():
    """
    主函数
    """
    print("开始处理SBM-DEA输入数据（SO₂ + NOx版本）...")
    
    # 检查文件是否存在
    if not os.path.exists(merged_file):
        print(f"错误: 找不到数据文件: {merged_file}")
        print("请确保已经运行了环境数据处理程序")
        return
    
    # 运行处理
    sbm_data = prepare_sbm_dea_so2_nox_only()
    
    if sbm_data is not None:
        print("\n" + "=" * 60)
        print("✅ 处理完成！")
        print("=" * 60)
        print("\n输出文件位于: C:\\Users\\15535\\Desktop\\TFP-Data")
        print("1. SBM_DEA_Input_SO2_NOx_Only.csv - MaxDEA输入文件")
        print("2. SBM_DEA_Input_SO2_NOx_Only.xlsx - 详细数据文件")
        print("\n现在可以使用MaxDEA计算绿色全要素生产率(GTFP)了！")

if __name__ == "__main__":
    main()