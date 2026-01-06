import pandas as pd
import os

def prepare_sbm_dea_two_pollutants(df):
    """
    准备SBM-DEA输入数据（2个非期望产出：SO₂和NOx）
    """
    print("准备SBM-DEA输入数据（2个非期望产出）...")
    
    # 创建SBM-DEA输入
    sbm_data = pd.DataFrame()
    
    # 1. DMU标识（公司_年份）
    if 'Stkcd' in df.columns and 'Year' in df.columns:
        sbm_data['DMU'] = df['Stkcd'].astype(str) + '_' + df['Year'].astype(str)
    elif '证券代码' in df.columns and 'year' in df.columns:
        sbm_data['DMU'] = df['证券代码'].astype(str) + '_' + df['year'].astype(str)
    else:
        # 创建临时标识
        sbm_data['DMU'] = ['DMU_' + str(i) for i in range(len(df))]
    
    # 2. 投入指标（3个）
    # 资本投入
    capital_cols = ['固定资产', '固定资产净额', 'fixed_assets']
    for col in capital_cols:
        if col in df.columns:
            sbm_data['投入_资本'] = df[col]
            break
    
    # 劳动投入
    labor_cols = ['员工总数', '员工数', 'employees', '在职员工总数']
    for col in labor_cols:
        if col in df.columns:
            sbm_data['投入_劳动'] = df[col]
            break
    
    # 能源投入（代理变量）
    energy_cols = ['接受劳务支付的现金', '购买商品劳务现金', 'material_cost']
    for col in energy_cols:
        if col in df.columns:
            sbm_data['投入_能源'] = df[col]
            break
    
    # 3. 期望产出（1个）
    revenue_cols = ['营业收入', 'revenue', '营收']
    for col in revenue_cols:
        if col in df.columns:
            sbm_data['产出_营收'] = df[col]
            break
    
    # 4. 非期望产出（2个）- 关键修改！
    # SO₂排放
    so2_cols = ['企业SO2排放量', 'SO2_排放', 'so2_emission']
    for col in so2_cols:
        if col in df.columns:
            sbm_data['非期望_SO2'] = df[col]
            break
    else:
        # 如果没找到，尝试计算
        if 'so2' in df.columns and 'province_scale_share' in df.columns:
            sbm_data['非期望_SO2'] = df['so2'] * df['province_scale_share']
    
    # NOx排放
    nox_cols = ['企业NOx排放量', 'NOx_排放', 'nox_emission']
    for col in nox_cols:
        if col in df.columns:
            sbm_data['非期望_NOx'] = df[col]
            break
    else:
        # 如果没找到，尝试计算
        if 'nox' in df.columns and 'province_scale_share' in df.columns:
            sbm_data['非期望_NOx'] = df['nox'] * df['province_scale_share']
    
    # 5. 数据清洗
    # 处理缺失值
    for col in sbm_data.columns:
        if col != 'DMU':
            # 用列中位数填充缺失
            median_val = sbm_data[col].median()
            sbm_data[col] = sbm_data[col].fillna(median_val)
            
            # 将零值替换为小正数（避免DEA除零错误）
            sbm_data[col] = sbm_data[col].replace(0, 0.001)
            
            # 如果有负值，取绝对值（污染物不应为负）
            if (sbm_data[col] < 0).any():
                sbm_data[col] = sbm_data[col].abs()
    
    print(f"\n✅ SBM-DEA数据准备完成!")
    print(f"DMU数量: {len(sbm_data)}")
    print(f"投入指标 (3个): {[col for col in sbm_data.columns if '投入' in col]}")
    print(f"期望产出 (1个): {[col for col in sbm_data.columns if '产出' in col]}")
    print(f"非期望产出 (2个): {[col for col in sbm_data.columns if '非期望' in col]}")
    
    # 显示统计信息
    print(f"\n数据统计:")
    for col in sbm_data.columns:
        if col != 'DMU':
            mean_val = sbm_data[col].mean()
            std_val = sbm_data[col].std()
            min_val = sbm_data[col].min()
            max_val = sbm_data[col].max()
            print(f"  {col:15s}: 均值={mean_val:10.2f}, 标准差={std_val:10.2f}, 范围=[{min_val:.2f}, {max_val:.2f}]")
    
    return sbm_data

def save_for_maxdea(sbm_data, output_dir):
    """
    保存为MaxDEA可读取的格式
    """
    # 保存为CSV（MaxDEA推荐格式）
    csv_path = os.path.join(output_dir, "SBM_DEA_Input_SO2_NOx.csv")
    sbm_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 保存为Excel
    excel_path = os.path.join(output_dir, "SBM_DEA_Input_SO2_NOx.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        sbm_data.to_excel(writer, sheet_name='DEA_Input', index=False)
        
        # 添加说明sheet
        instructions = pd.DataFrame({
            '变量类型': ['标识变量', '投入指标', '投入指标', '投入指标', 
                      '期望产出', '非期望产出', '非期望产出'],
            '变量名': ['DMU', '投入_资本', '投入_劳动', '投入_能源', 
                    '产出_营收', '非期望_SO2', '非期望_NOx'],
            '说明': ['决策单元标识（公司_年份）', 
                   '固定资产（万元）', 
                   '员工总数（人）', 
                   '购买商品劳务现金（万元）- 能源代理', 
                   '营业收入（万元）', 
                   'SO₂排放量（吨）', 
                   'NOx排放量（吨）'],
            'MaxDEA设置': ['不选', 'Input', 'Input', 'Input', 
                        'Desirable Output', 'Undesirable Output', 'Undesirable Output']
        })
        instructions.to_excel(writer, sheet_name='变量说明', index=False)
    
    print(f"\n✅ 数据已保存:")
    print(f"  CSV文件: {csv_path}")
    print(f"  Excel文件: {excel_path}")
    
    return csv_path, excel_path

def main():
    print("=" * 60)
    print("SBM-DEA输入数据准备（SO₂ + NOx 两个非期望产出）")
    print("=" * 60)
    
    # 读取你的综合数据
    data_file = os.path.join(desktop_path, "Final_Data_with_Environment.xlsx")
    
    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")
        print("请先运行环境数据处理程序")
        return
    
    print(f"读取数据文件: {data_file}")
    df = pd.read_excel(data_file)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)[:15]}...")
    
    # 检查必要数据
    required = ['营业收入', '固定资产', '员工总数', '接受劳务支付的现金', 
                'so2', 'nox', 'province_scale_share']
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ 缺少以下列: {missing}")
        print("请检查数据文件")
    
    # 准备SBM-DEA输入
    sbm_data = prepare_sbm_dea_two_pollutants(df)
    
    # 保存文件
    csv_file, excel_file = save_for_maxdea(sbm_data, desktop_path)
    
    # 显示MaxDEA设置指南
    print("\n" + "=" * 60)
    print("MaxDEA 设置指南（SO₂ + NOx 版本）")
    print("=" * 60)
    print("\n1. 打开 MaxDEA 软件")
    print("2. File → New Project")
    print("3. Data → Import from File")
    print(f"   选择: {csv_file}")
    print("\n4. 变量设置:")
    print("   - DMU: 不选择（仅标识）")
    print("   - 投入_资本: Input")
    print("   - 投入_劳动: Input")
    print("   - 投入_能源: Input")
    print("   - 产出_营收: Desirable Output")
    print("   - 非期望_SO2: Undesirable Output")
    print("   - 非期望_NOx: Undesirable Output")
    print("\n5. 模型设置:")
    print("   Model Type: SBM")
    print("   Orientation: Non-oriented")
    print("   RTS: Variable (VRS)")
    print("   Undesirable Outputs: Include")
    print("\n6. 点击 'Solve' 运行计算")
    
    print("\n" + "=" * 60)
    print("✅ 准备完成！现在可以运行MaxDEA计算GTFP")
    print("=" * 60)

if __name__ == "__main__":
    main()