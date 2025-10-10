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
            
        # 计算测试集RMSE的分位数
        test_mses = [r['test_rmse'] for r in results]
        q25 = np.percentile(test_mses, 25)
        q50 = np.percentile(test_mses, 50)  # 中位数
        q75 = np.percentile(test_mses, 75)
        
        # 找到最接近分位数的实验
        def find_closest(target):
            distances = [abs(mse - target) for mse in test_mses]
            return distances.index(min(distances))
        
        idx_q25 = find_closest(q25)
        idx_q50 = find_closest(q50)
        idx_q75 = find_closest(q75)
        
        # 保存结果
        dataset_results = {
            'dataset': dataset,
            'total_runs': len(results),
            'quantiles': {
                'q25': q25,
                'q50': q50,
                'q75': q75
            },
            'selected_experiments': {
                'q25': {
                    'seed': results[idx_q25]['seed'],
                    'test_rmse': results[idx_q25]['test_rmse'],
                    'fitness_trend': results[idx_q25]['fitness_trend'],
                    'training_time': results[idx_q25]['training_time'],
                    'best_individual': results[idx_q25]['best_individual']
                },
                'q50': {
                    'seed': results[idx_q50]['seed'],
                    'test_rmse': results[idx_q50]['test_rmse'],
                    'fitness_trend': results[idx_q50]['fitness_trend'],
                    'training_time': results[idx_q50]['training_time'],
                    'best_individual': results[idx_q50]['best_individual']
                },
                'q75': {
                    'seed': results[idx_q75]['seed'],
                    'test_rmse': results[idx_q75]['test_rmse'],
                    'fitness_trend': results[idx_q75]['fitness_trend'],
                    'training_time': results[idx_q75]['training_time'],
                    'best_individual': results[idx_q75]['best_individual']
                }
            }
        }
        
        # 保存到文件
        output_file = f"results/{dataset}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
        print(f"数据集 {dataset} 结果已保存到 {output_file}")
        print(f"  Q25 RMSE: {q25:.6f} (种子 {results[idx_q25]['seed']})")
        print(f"  Q50 RMSE: {q50:.6f} (种子 {results[idx_q50]['seed']})")
        print(f"  Q75 RMSE: {q75:.6f} (种子 {results[idx_q75]['seed']})")
        print()

if __name__ == "__main__":
    run_experiments()