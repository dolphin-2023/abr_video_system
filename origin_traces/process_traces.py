import os
import glob
import pandas as pd
import numpy as np

def clean_and_resample_trace(input_file, output_txt, is_mb_s=False):
    """
    通用 Trace 清洗引擎
    将原始杂乱的网络日志，清洗为每秒 1 行的 Kbps 标准序列。
    """
    try:
        # 1. 读取原始数据：通常是时间戳和吞吐量两列，以制表符或空格分隔
        # 使用正则表达式 '\s+' 处理不同数量的空格或制表符
        df = pd.read_csv(input_file, sep=r'\s+', header=None, 
                        names=['timestamp', 'throughput'], 
                        dtype={'timestamp': float, 'throughput': float})
        df = df.dropna()
        
        # 2. 统一量纲：目标是 Mbps (Megabits per second)
        # Puffer 数据集的吞吐量单位是 Byte/s 或 MB/s，需要乘以 8 转换为 bit
        if is_mb_s:
            df['mbps'] = df['throughput'] * 8.0
        else:
            df['mbps'] = df['throughput']
        
        # 3. 时间归零与重采样：将第一行的时间视作第 0 秒
        start_time = df['timestamp'].min()
        df['relative_time'] = df['timestamp'] - start_time
        # 向下取整，划定数据所属的“秒”区间
        df['sec_index'] = np.floor(df['relative_time']).astype(int)
        
        # 4. 按秒聚合：求出每一秒内的平均吞吐量
        grouped = df.groupby('sec_index')['mbps'].mean().reset_index()
        
        # 5. 空洞填补：构建一个连续的时间轴，缺失的秒数用 0 Mbps 填充，注意这个逻辑有问题，网络容量维持上一时刻的观测状态更合适
        max_sec = grouped['sec_index'].max()
        all_secs = pd.DataFrame({'sec_index': range(int(max_sec) + 1)})
        merged = pd.merge(all_secs, grouped, on='sec_index', how='left').fillna(0)
        
        # 6. 单位下探：转换为 Kbps（限速代理的基准单位）
        merged['kbps'] = merged['mbps'] * 1000.0
        
        # 7. 落盘保存：确保最小带宽为 10 Kbps，防止完全断网导致播放器内部逻辑崩溃
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("# 标准 1Hz Kbps 清洗轨迹\n")
            for kbps in merged['kbps']:
                safe_kbps = max(10.0, kbps)
                f.write(f"{safe_kbps:.2f}\n")
                
        print(f"清洗完成 | 有效时长: {len(merged)} 秒 | 输出: {os.path.basename(output_txt)}")
        
    except Exception as e:
        print(f"处理失败 [{input_file}]: {e}")

def batch_process():
    # 数据集根目录 (假设你克隆的仓库放在此目录下)
    base_dir = "Real-world-bandwidth-traces-master"
    if not os.path.exists(base_dir):
        print(f"未找到原始数据集目录: {base_dir}，请确保路径正确。")
        return

    # 输出目录直接指向模拟器的 traces 文件夹，实现数据无缝对接
    out_dir = "../simulator/traces/real_world"
    os.makedirs(out_dir, exist_ok=True)
    '''
    print("=== 开始清洗 HSDPA (3G) 数据集 ===")
    # 移动网络场景，波动极其剧烈
    hsdpa_files = glob.glob(f"{base_dir}/traces_3gp/*.log")
    for file in hsdpa_files:
        name = os.path.basename(file).replace('.log', '.txt')
        clean_and_resample_trace(file, f"{out_dir}/hsdpa_{name}", is_mb_s=False)
    print("=== 开始清洗 cooked_3gp 数据集 ===")
    cooked_hsdpa_files = glob.glob(f"{base_dir}/cooked_3gp/*")   # 无后缀，用 *
    for file in cooked_hsdpa_files:
        if os.path.isfile(file):  # 过滤掉可能的子目录
            name = os.path.basename(file)
            clean_and_resample_trace(file, f"{out_dir}/cooked_{name}.txt", is_mb_s=False)

    print("\n=== 开始清洗 Oboe 数据集 ===")
    # 包含车载、步行等多种日常移动场景
    oboe_files = glob.glob(f"{base_dir}/traces_oboe/*.txt")
    for file in oboe_files:
        name = os.path.basename(file)
        clean_and_resample_trace(file, f"{out_dir}/oboe_{name}", is_mb_s=False)
    '''
    print("\n=== 开始清洗 Puffer 数据集 ===")
    # 斯坦福真实直播平台数据，需特殊处理字节单位
    puffer_files = glob.glob(f"{base_dir}/puffer_211017/**/*") + glob.glob(f"{base_dir}/puffer_220218/**/*")  # 两批数据，均无后缀
    for file in puffer_files:
        name = os.path.basename(file)
        clean_and_resample_trace(file, f"{out_dir}/puffer_{name}", is_mb_s=True)
    '''
    print("=== 开始清洗 FCC 数据集 ===")
    fcc_files = glob.glob(f"{base_dir}/fcc_ori/**/*.log", recursive=True)
    for file in fcc_files:
        name = os.path.basename(file).replace('.log', '.txt')
        clean_and_resample_trace(file, f"{out_dir}/fcc_{name}", is_mb_s=False)

    print("\n=== 开始清洗 FCC + HSDPA 混合数据集 ===")
    mixed_files = glob.glob(f"{base_dir}/fcc_and_hsdpa/**/*.log", recursive=True)
    for file in mixed_files:
        name = os.path.basename(file).replace('.log', '.txt')
        clean_and_resample_trace(file, f"{out_dir}/mixed_{name}", is_mb_s=False)
    '''

    print(f"\n全部清洗结束！可用轨迹已存入 {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    batch_process()