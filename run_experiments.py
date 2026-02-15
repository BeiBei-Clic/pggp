import numpy as np
import json
import os
from main import main

# 指定的6个数据集
datasets = ["6.2","6.2b","12.4","14.3","14.4","25.13"]

# 随机种子0-9
seeds = list(range(10))

def run_experiments():
    for dataset in datasets:
        print(f"开始处理数据集: {dataset}")
        
        # 存储每次运行的结果
        results = []
        
        # 运行10次实验
        for seed in seeds:
            print(f"  运行种子 {seed}", end=" ... ")
            result = main(filename=dataset, seed=seed, data_count=100, generations=200, population_size=300)
            result['seed'] = seed
            results.append(result)
            print(f"完成 (RMSE: {result['test_rmse']:.6f})")

        if len(results) == 0:
            print(f"数据集 {dataset} 所有运行都失败")
            continue
            
        # 保存所有运行的完整结果
        dataset_results = {
            'dataset': dataset,
            'total_runs': len(results),
            'all_experiments': results
        }
        
        # 保存到文件
        output_file = f"results/{dataset}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
        print(f"数据集 {dataset} 结果已保存到 {output_file}")
        
        # 显示所有运行结果的摘要
        test_rmses = [r['test_rmse'] for r in results]
        print(f"  所有运行的RMSE结果:")
        for i, rmse in enumerate(test_rmses):
            print(f"    种子 {i}: {rmse:.6f}")
        print()

if __name__ == "__main__":
    run_experiments()