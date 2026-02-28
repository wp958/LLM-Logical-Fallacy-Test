# 🤖 大语言模型能抵抗逻辑谬误吗？——基于 Python 的批量化测试与统计分析
> **LLM Evaluation Project**: Can Large Language Models Resist Logical Fallacies?
> **作者**：王潘 | **专业**：哲学

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![API](https://img.shields.io/badge/API-Spark_X1-orange.svg)
![Prompt_Engineering](https://img.shields.io/badge/Prompt-Structured_JSON-green.svg)
![Data_Analysis](https://img.shields.io/badge/Data_Analysis-Pandas-lightgrey.svg)

## 📖 项目简介
本项目是对当前生成式大语言模型（以星火 Spark X1 为例）逻辑推理能力边界的实证研究。
通过将传统的逻辑学/认知心理学经典问题转化为自动化测试探针，探讨大语言模型究竟是真正具备了“逻辑理解”能力，还是仅仅作为“随机鹦鹉（Stochastic Parrots）”在进行深度的“模式匹配”。本项目结合了**哲学逻辑思辨**与 **Python 自动化 API 测试**，揭示了 LLM 在认知偏差与逻辑悖论面前的真实表现。

## 🎯 核心技术亮点 (Engineering Highlights)
- **自动化评测管线构建**：使用 Python `requests` 库直连大模型 API，实现批量化提问与数据回收。
- **结构化 Prompt 工程**：设计了严格的 Prompt 模板，强制大模型以 `JSON` 格式输出布尔值（判断真假）、浮点数（自信度得分）与思维链（CoT），实现了评测数据的结构化提取。
- **系统化“逻辑探针”设计**：涵盖 4 大模块（基础形式逻辑、概念语义理解、极限悖论挑战、认知陷阱测试）、8 种经典题型（如库里悖论、琳达问题等）。

---

## 📊 核心测试结果与数据可视化

### 1. 模块准确率分层：完美的“形式”，脆弱的“语义”
> 实验表明，模型在基础形式逻辑（模块1）和概率认知陷阱（模块4）上表现出 100% 的准确率，但在需要深层语境和概念理解的“语义模块（模块2）”上，准确率降至 77.8%。它擅长处理“形式”，却难以把握“内容”。


<img width="3000" height="1800" alt="accuracy_by_module" src="https://github.com/user-attachments/assets/4cc83eb8-2db9-4883-8b7c-7284b19c3ad2" />


### 2. 诊断能力矩阵：模型知道错在哪里吗？
> 混淆矩阵显示，模型内部已形成有效的谬误诊断机制。它不仅能判断有效性，还能在绝大多数情况下精准归类谬误类型（如区分“肯定后件”与“否定前件”），唯一的混淆发生在语义复杂的“模态谬误”中。


<img width="3600" height="2400" alt="fallacy_confusion_matrix" src="https://github.com/user-attachments/assets/73dc85e4-e9a0-4b9c-8bf2-d685f628bf2b" />


### 3. 最危险的发现：高自信的“幻觉”（The Confidence Gap）
> 这是本项目最核心的发现。对于错误的判断（左侧箱体），模型的自信度中位数依然高达 0.95。这意味着它“不知道自己不知道”。一个在 99% 情况下可靠的模型，在 1% 的错误上表现得理直气壮，这在实际应用中是巨大的安全隐患。


<img width="3000" height="1800" alt="confidence_distribution" src="https://github.com/user-attachments/assets/a04b4a11-4995-472a-a06c-f71ca2ba9a0b" />


### 4. 悖论极限测试：复现系统 vs 推理系统
通过对“说谎者悖论”与“库里悖论”的对比测试发现，模型能完美背诵教科书里常见的“说谎者悖论”归谬过程，却在面对结构更精妙的“库里悖论”时陷入死循环与混淆。证明当前 LLM 本质上是一个强大的**“知识复述系统”**，而非统一规则的通用逻辑推理器。

---

## 💡 结论与展望
大语言模型具有明显的“能力分层”：是一个高分的“逻辑考生”，但远非真正的“逻辑学家”。
为实现真正的人工智能，未来亟需：
1. **拥抱神经符号系统（Neuro-symbolic AI）**：将 LLM 的语感与传统符号 AI 的严谨推理相结合。
2. **构建内在世界因果模型**：超越纯粹基于统计概率的文本拼接。
3. **引入“自我认知”与置信度评估机制**：打破模型“过度自信”的黑盒，提升人机信任度。

> *完整的自动化测试 Python 脚本与详细统计数据分析，请参见本仓库文件。*
