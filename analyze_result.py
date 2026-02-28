import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# --- 配置区 ---
# 请将这里的 'your_csv_file.csv' 替换为你实际生成的文件名
CSV_FILENAME = 'logic_fallacy_experiment_results_20250627-170006.csv'  # 示例文件名，请替换

# 为了在图表中正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


# --- 数据加载与预处理 ---

def load_and_preprocess_data(filename):
    """加载CSV数据并进行预处理"""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'。请确保文件名正确且文件与脚本在同一目录下。")
        return None

    # 定义每种探针的“正确答案”(Ground Truth)
    # is_valid_reasoning 的正确值：True代表逻辑有效，False代表存在谬误/偏差
    ground_truth = {
        'Modus Ponens (Control)': True,
        'Affirming the Consequent (Fallacy)': False,
        'Modus Tollens (Control)': True,
        'Denying the Antecedent (Fallacy)': False,
        'Quantifier Scope (Fallacy)': False,
        'Modal Fallacy (De Re/De Dicto)': False,
        'Composition Fallacy': False,
        'Linda Problem (Conjunction Fallacy)': False,
        'Gambler\'s Fallacy': False,
        'Base Rate Fallacy': False
        # 悖论类型不包含在内，因为它们没有简单的对错
    }

    # 筛选出需要进行准确率评估的探针
    eval_probes_df = df[df['probe_type'].isin(ground_truth.keys())].copy()

    # 计算每个回答是否正确
    eval_probes_df['ground_truth'] = eval_probes_df['probe_type'].map(ground_truth)
    eval_probes_df['is_correct'] = (eval_probes_df['is_valid_reasoning'] == eval_probes_df['ground_truth'])

    print("数据加载和预处理完成！")
    return eval_probes_df, df


# --- 分析与可视化函数 ---

def analyze_and_visualize(df):
    """执行所有分析和可视化任务"""
    if df is None or df.empty:
        print("数据为空，无法进行分析。")
        return

    print("\n--- 1. 宏观定量统计分析 ---")
    # 1.1 总体准确率
    overall_accuracy = df['is_correct'].mean()
    print(f"[*] 总体准确率 (非悖论题): {overall_accuracy:.2%}")

    # 1.2 分模块准确率
    accuracy_by_module = df.groupby('module')['is_correct'].mean().sort_index()
    print("\n[*] 分模块准确率:")
    print(accuracy_by_module)

    # 1.3 分项准确率
    accuracy_by_probe = df.groupby('probe_type')['is_correct'].mean().sort_values(ascending=False)
    print("\n[*] 分探针类型准确率:")
    print(accuracy_by_probe)

    # 1.4 置信度分析
    print("\n[*] 置信度分析:")
    confidence_stats = df.groupby('is_correct')['confidence_score'].describe()
    print(confidence_stats)

    # 1.5 谬误识别准确率 (混淆矩阵)
    fallacy_df = df[(df['ground_truth'] == False) & (df['is_correct'] == True)]
    # 理论上正确的谬误类型
    true_fallacy_types = {
        'Affirming the Consequent (Fallacy)': 'AFFIRMING_THE_CONSEQUENT',
        'Denying the Antecedent (Fallacy)': 'DENYING_THE_ANTECEDENT',
        'Quantifier Scope (Fallacy)': 'QUANTIFIER_SCOPE_FALLACY',
        'Modal Fallacy (De Re/De Dicto)': 'MODAL_FALLACY',
        'Composition Fallacy': 'COMPOSITION_FALLACY',
        'Linda Problem (Conjunction Fallacy)': 'CONJUNCTION_FALLACY',
        'Gambler\'s Fallacy': 'GAMBLER_S_FALLACY',
        'Base Rate Fallacy': 'BASE_RATE_FALLACY'
    }
    fallacy_df['true_fallacy_type'] = fallacy_df['probe_type'].map(true_fallacy_types)

    if not fallacy_df.empty:
        confusion_matrix = pd.crosstab(fallacy_df['true_fallacy_type'], fallacy_df['fallacy_type'])
        print("\n[*] 谬误识别混淆矩阵 (行: 真实谬误, 列: 模型判断):")
        print(confusion_matrix)

    # --- 2. 可视化 ---
    print("\n--- 正在生成可视化图表... ---")

    # 图1: 分模块准确率 (条形图)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracy_by_module.index, y=accuracy_by_module.values, palette='viridis')
    plt.title('大语言模型在不同逻辑模块上的准确率', fontsize=16)
    plt.xlabel('逻辑模块', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.xticks(ticks=np.arange(len(accuracy_by_module)), labels=[f"模块 {i}" for i in accuracy_by_module.index])
    plt.ylim(0, 1)
    # 在条形图上显示百分比
    for index, value in enumerate(accuracy_by_module):
        plt.text(index, value + 0.02, f'{value:.1%}', ha='center', fontsize=11)
    plt.savefig('accuracy_by_module.png', dpi=300)
    print("[*] 图表 'accuracy_by_module.png' 已保存。")

    # 图2: 置信度分布对比 (箱形图)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_correct', y='confidence_score', data=df, palette='Set2')
    plt.title('模型判断正确与错误时的置信度分布', fontsize=16)
    plt.xlabel('判断是否正确', fontsize=12)
    plt.ylabel('置信度分数', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['错误', '正确'])
    plt.savefig('confidence_distribution.png', dpi=300)
    print("[*] 图表 'confidence_distribution.png' 已保存。")

    # 图3: 谬误识别混淆矩阵 (热力图)
    if not fallacy_df.empty and not confusion_matrix.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
        plt.title('谬误类型识别混淆矩阵', fontsize=16)
        plt.xlabel('模型识别的谬误类型', fontsize=12)
        plt.ylabel('实际的谬误类型', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()  # 调整布局防止标签重叠
        plt.savefig('fallacy_confusion_matrix.png', dpi=300)
        print("[*] 图表 'fallacy_confusion_matrix.png' 已保存。")

    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    eval_df, full_df = load_and_preprocess_data(CSV_FILENAME)
    analyze_and_visualize(eval_df)

    # 额外提示进行定性分析
    print("\n--- 3. 定性分析建议 ---")
    print("定量分析已完成。接下来，请手动检查CSV文件，重点关注以下内容以进行定性分析：")
    print("1. 悖论题 (Liar Paradox, Curry's Paradox) 的 'reasoning_chain' 和 'final_explanation'，分析模型的应对策略。")
    print("2. 寻找判断错误但置信度高 (如 >0.9) 的案例，分析其 'reasoning_chain'，看模型如何“自信地犯错”。")
    print("3. 寻找“歪打正着”的案例（'is_correct'为True，但'reasoning_chain'逻辑混乱）。")
    print("4. 将这些典型案例的分析写入你的论文，作为定量数据的有力补充。")