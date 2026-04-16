import pandas as pd
import os
import re # Import regex module for keyword checking
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Cache for index component counts to avoid redundant API calls
index_details_cache = {}

def is_valid_index_identifier(identifier, fund_short_name=None, fund_family=None):
    """Check if the identifier string looks like a valid index name or ticker."""
    if not identifier or identifier == "N/A":
        return False
    
    identifier_lower = identifier.lower()
    
    # Avoid using the fund's own name or family as the index
    if fund_short_name and identifier_lower == fund_short_name.lower():
        return False
    if fund_family and identifier_lower == fund_family.lower():
        return False
    
    # Check for common index keywords or ticker format
    index_keywords = ["index", "s&p", "nasdaq", "russell", "dow", "msci", "ftse"]
    if identifier.startswith("^") or any(keyword in identifier_lower for keyword in index_keywords):
        return True
    
    # Add more specific checks if needed
    return False

def get_index_constituent_count(index_identifier_str):
    """Helper function to get constituent count for a given VALID index identifier."""
    # Now assumes index_identifier_str is already validated or is None/N/A
    if not YFINANCE_AVAILABLE or not index_identifier_str or index_identifier_str == "N/A":
        return "N/A"

    normalized_identifier_key = str(index_identifier_str).lower().strip()
    if not normalized_identifier_key:
        return "N/A"

    if normalized_identifier_key in index_details_cache: # Check cache
        return index_details_cache[normalized_identifier_key]

    common_indices_map = {
        "s&p 500 index": {"ticker": "^GSPC", "typical_count": 503},
        "s&p 500": {"ticker": "^GSPC", "typical_count": 503},
        "spdr s&p 500 etf trust": {"ticker": "^GSPC", "typical_count": 503},
        "nasdaq-100 index": {"ticker": "^NDX", "typical_count": 101},
        "nasdaq 100 index": {"ticker": "^NDX", "typical_count": 101},
        "nasdaq-100": {"ticker": "^NDX", "typical_count": 101},
        "nasdaq 100": {"ticker": "^NDX", "typical_count": 101},
        "invesco qqq trust": {"ticker": "^NDX", "typical_count": 101},
        "russell 2000 index": {"ticker": "^RUT", "typical_count": 2000},
        "russell 2000": {"ticker": "^RUT", "typical_count": 2000},
        "dow jones industrial average": {"ticker": "^DJI", "typical_count": 30},
        "msci world": {"ticker": "^MIWO00000PUS", "typical_count": "Varies (~1500)"}, # Example, ticker might vary
        "msci em": {"ticker": "^MSCIEF", "typical_count": "Varies (~1400)"} # Example, ticker might vary
    }

    index_ticker_to_query = None
    count = "N/A" # Default count

    if normalized_identifier_key.startswith("^"): # Already a ticker
        index_ticker_to_query = normalized_identifier_key
        # Attempt to find typical count even for tickers for fallback
        for name_key, data in common_indices_map.items():
            if data["ticker"] == index_ticker_to_query:
                count = data["typical_count"]
                break
    else: # It's a name, try to map it
        if normalized_identifier_key in common_indices_map:
            data = common_indices_map[normalized_identifier_key]
            index_ticker_to_query = data["ticker"]
            count = data["typical_count"]
        else:
            for name_key, data in common_indices_map.items():
                if name_key in normalized_identifier_key:
                    index_ticker_to_query = data["ticker"]
                    count = data["typical_count"]
                    break
    
    # Only query components if we have a ticker derived from a VALID identifier
    if index_ticker_to_query:
        try:
            index_obj = yf.Ticker(index_ticker_to_query)
            components_df = index_obj.components
            if components_df is not None and not components_df.empty:
                count = len(components_df)
            # If .components is empty or None, we keep the 'typical_count' or 'N/A'
        except Exception as e:
            # print(f"Failed to get .components for index {index_ticker_to_query}: {e}")
            pass 
    
    index_details_cache[normalized_identifier_key] = count # Cache result
    return count

def get_ticker_details(ticker_symbol):
    """获取 ticker 的类型、追踪的指数（如果适用）和该指数的成分股数量。"""
    if not YFINANCE_AVAILABLE:
        return "yfinance not installed", "N/A", "N/A"
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        quote_type_raw = info.get('quoteType', 'Unknown')
        fund_short_name = info.get('shortName')
        fund_family = info.get('fundFamily')
        
        product_type = "Unknown"
        tracked_index_name = "N/A"
        index_constituent_count = "N/A"

        if quote_type_raw == 'EQUITY':
            product_type = 'Stock'
        elif quote_type_raw == 'ETF':
            product_type = 'ETF'
            potential_index_id = "N/A"
            # 1. Check 'benchmark'
            benchmark_val = info.get('benchmark')
            if is_valid_index_identifier(benchmark_val, fund_short_name, fund_family):
                potential_index_id = benchmark_val
            else:
                # 2. If benchmark is invalid/missing, check 'category'
                category_val = info.get('category')
                if is_valid_index_identifier(category_val, fund_short_name, fund_family):
                    potential_index_id = category_val
                # We explicitly AVOID using fundFamily as the index identifier here.

            if potential_index_id != "N/A":
                tracked_index_name = potential_index_id
                # Get count ONLY if we found a valid identifier
                index_constituent_count = get_index_constituent_count(tracked_index_name)
            # else: tracked_index_name and count remain "N/A"

        elif quote_type_raw == 'MUTUALFUND':
            product_type = 'Mutual Fund'
            # Optionally, try to get benchmark for mutual funds too
            benchmark_val = info.get('benchmark')
            if is_valid_index_identifier(benchmark_val, fund_short_name, fund_family):
                tracked_index_name = benchmark_val
                # No count for mutual fund benchmarks typically via this method
                index_constituent_count = "N/A (MF)"
            else:
                category_val = info.get('category')
                if category_val: # Just record category if no valid benchmark
                    tracked_index_name = category_val
                    index_constituent_count = "N/A (MF)"

        else:
            product_type = quote_type_raw # Could be 'INDEX', 'CURRENCY', 'FUTURE', etc.
        
        return product_type, tracked_index_name, index_constituent_count

    except Exception as e:
        # print(f"Could not get details for {ticker_symbol}: {e}")
        return "Unknown/Error", "N/A", "N/A"

def analyze_stocks(folder_path):
    """
    对指定文件夹中所有CSV文件的 'close' 列进行描述性统计分析，并查询产品类型、追踪指数及成分股数量。
    """
    all_stock_data = []
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return all_stock_data

    print("\n正在分析文件并查询产品详细信息（这可能需要一些时间）...")
    # Filter for CSV files only before counting for progress
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    total_files = len(file_list)

    for i, filename in enumerate(file_list):
        # if filename.endswith(".csv"): # This check is now done when creating file_list
        file_path = os.path.join(folder_path, filename)
        product_code = filename.replace(".csv", "")
        print(f"处理中: {filename} ({i+1}/{total_files})")

        stats_data = {'Product Code': product_code}
        product_type, tracked_index, constituent_count = get_ticker_details(product_code)
        stats_data['Type'] = product_type
        stats_data['Tracked Index'] = tracked_index
        stats_data['Index Constituent Count'] = constituent_count
        
        try:
            df = pd.read_csv(file_path, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df.dropna(subset=['close'], inplace=True)

            if not df.empty:
                stats = df['close'].describe().to_dict()
                stats_data.update(stats)
            else:
                stats_data['error_stats'] = "文件为空或 'close' 列无可分析数据。"
        except ValueError as ve:
            stats_data['error_stats'] = f"统计分析时发生值错误: {ve}。"
        except Exception as e:
            stats_data['error_stats'] = f"统计分析时发生未知错误: {e}"
        all_stock_data.append(stats_data)
    print("产品详细信息查询和文件分析完成。")
    return all_stock_data

# --- 使用示例 ---
history_folder_path = 'History'
excel_output_path = 'stock_analysis_summary.xlsx'

if not YFINANCE_AVAILABLE:
    print("警告: yfinance 库未安装。产品类型、指数信息及成分股数量将无法获取。")
    print("请运行 'pip install yfinance' 来安装它。")

stock_statistics_list = analyze_stocks(history_folder_path)

if not stock_statistics_list:
    print(f"在 '{history_folder_path}' 文件夹中没有找到CSV文件或无法分析。")
else:
    df_results = pd.DataFrame(stock_statistics_list)
    
    # Define column order, making sure new columns are included
    desired_cols = ['Product Code', 'Type', 'Tracked Index', 'Index Constituent Count']
    # Add existing stats columns, excluding any error columns for now
    stats_cols = [col for col in df_results.columns if col not in desired_cols and col != 'error_stats']
    final_cols = desired_cols + stats_cols
    if 'error_stats' in df_results.columns: # Append error column at the end if it exists
        final_cols.append('error_stats')
    
    # Reorder DataFrame according to final_cols, handling missing columns gracefully
    df_results = df_results.reindex(columns=final_cols)
        
    try:
        df_results.to_excel(excel_output_path, index=False, engine='openpyxl')
        print(f"\n分析结果已成功保存到: {os.path.abspath(excel_output_path)}")
        if not YFINANCE_AVAILABLE:
             print("\n提示：由于yfinance未安装，Excel文件中'Type', 'Tracked Index'和'Index Constituent Count'列可能为空或显示错误。")
    except Exception as e:
        print(f"\n保存Excel文件失败: {e}")
        print("请确保您已安装 'openpyxl' 库 (pip install openpyxl)")

# --- 关于选取建议的思考 ---
# 拿到统计结果后，可以根据以下一些角度考虑选取：
# 1. 波动性 (标准差 std):
#    - 高标准差通常意味着价格波动较大，可能带来更高风险和更高潜在回报。
#    - 低标准差则价格相对稳定。
# 2. 价格区间 (min, max, 25%, 50%, 75%):
#    - 了解资产的历史价格范围。
#    - 结合当前价格与历史分位点，判断是处于高位还是低位。
# 3. 数据量 (count):
#    - 数据点越多，统计结果通常越可靠，历史模式也可能更明显。
#    - 如果某些资产数据量过少，可能不适合进行某些依赖长期数据的策略。
# 4. 平均价格 (mean):
#    - 可以作为资产价值的一个参考，但需结合其他指标。
#
# 选取策略的建议需要结合您的投资目标和风险偏好。例如：
# - 如果您寻求高增长潜力且能承受高风险，可能会关注标准差较大，且近期有上涨趋势（例如当前价格高于75%分位数）的股票。
# - 如果您偏好稳健，可能会选择标准差较小，数据量充足，且价格处于历史相对合理区间的ETF或股票。
# - 还可以考虑结合不同类型的资产进行组合，例如一些高波动股票配合一些稳健的ETF。
#
# 拿到具体统计数据后，您可以告诉我您的偏好，我可以提供更具体的建议。 
