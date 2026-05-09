import os
import glob
import pandas as pd
import numpy as np

def clean_and_resample_trace(input_file, output_txt, is_mb_s=False):
    """
    Puffer 轨迹专用清洗引擎。
    将非连续的采样日志对齐为 1Hz 的 Kbps 标准序列，并采用前向填充修复数据稀疏空洞。
    """
    try:
        # 1. 挂载并解析稀疏的时序日志
        df = pd.read_csv(input_file, sep=r'\s+', header=None, 
                        names=['timestamp', 'throughput'], 
                        dtype={'timestamp': float, 'throughput': float})
        df = df.dropna()
        
        if df.empty:
            return

        # 2. 统一量纲为 Mbps
        if is_mb_s:
            df['mbps'] = df['throughput'] * 8.0
        else:
            df['mbps'] = df['throughput']
        
        # 3. 时间轴对齐与离散化
        start_time = df['timestamp'].min()
        df['relative_time'] = df['timestamp'] - start_time
        df['sec_index'] = np.floor(df['relative_time']).astype(int)
        
        # 4. 秒级聚合计算
        grouped = df.groupby('sec_index')['mbps'].mean().reset_index()
        
        # 5. 构建连续时间轴
        max_sec = grouped['sec_index'].max()
        all_secs = pd.DataFrame({'sec_index': range(int(max_sec) + 1)})
        merged = pd.merge(all_secs, grouped, on='sec_index', how='left')
        
        # 【核心修正】：使用前向填充 (ffill) 替代 fillna(0)
        # 假设在未采样的空闲周期内，网络容量维持上一时刻的观测状态
        merged['mbps'] = merged['mbps'].ffill().bfill()
        
        # 6. 精度下探至 Kbps
        merged['kbps'] = merged['mbps'] * 1000.0
        
        # 7. 约束边界并落盘 (补充 .txt 后缀保障文件可读性)
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("# 标准 1Hz Kbps 清洗轨迹 (Puffer Dataset)\n")
            for kbps in merged['kbps']:
                safe_kbps = max(10.0, kbps)
                f.write(f"{safe_kbps:.2f}\n")
                
        print(f"清洗完成 | 有效时长: {len(merged)} 秒 | 输出: {os.path.basename(output_txt)}")
        
    except Exception as e:
        print(f"处理失败 [{input_file}]: {e}")

def batch_process():
    base_dir = "Real-world-bandwidth-traces-master"
    if not os.path.exists(base_dir):
        print(f"未找到原始数据集目录: {base_dir}，请确保路径正确。")
        return

    # 指向外部模拟器的标准拓扑路径
    out_dir = "../simulator/traces/real_world"
    os.makedirs(out_dir, exist_ok=True)

    print("=== 开始清洗 Puffer 数据集 ===")
    
    # 启用递归查找 (recursive=True)，穿透 cooked_traces 等所有层级的子目录
    puffer_files = glob.glob(f"{base_dir}/puffer_211017/**/*", recursive=True) + \
                   glob.glob(f"{base_dir}/puffer_220218/**/*", recursive=True)
    
    for file in puffer_files:
        if os.path.isfile(file):
            name = os.path.basename(file)
            # 过滤系统隐藏文件或非日志文件
            if not name.startswith('.'):
                # 强制为无后缀的原始文件添加 .txt 标识
                clean_and_resample_trace(file, f"{out_dir}/puffer_{name}.txt", is_mb_s=True)

    print(f"\n全部清洗结束！可用轨迹已存入 {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    batch_process()