# -*- coding: utf-8 -*-
"""
Financial Deep Research Faithful Grader for Factuality Evaluation

This module provides a grader for evaluating the factuality and faithfulness
of financial research agent reports based on provided knowledge and context.
"""

import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderError
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# Chinese Prompt
FINANCIAL_FAITHFUL_PROMPT_ZH = """# 任务
你是一位专业的金融数据分析专家，具备深厚的金融信息提取和审查能力。你需要经过严密的分析思考，提取出报告中的四元组信息（主体，指标，数值，时间），并判断与搜索结果中是否一致。

# 评估流程（Chain-of-Thought）
请按照以下步骤进行思考和分析：

## 步骤1：仔细阅读搜索结果
- 逐个来源阅读，理解每个搜索结果的内容和时间背景
- 识别搜索结果中包含的所有金融数据点
- 注意不同来源之间的数据对应关系和时间差异

## 步骤2：提取报告中的四元组
从报告中提取所有包含具体数值的陈述，形成四元组：
1. **主体**：指标的主体，可以是公司名称、板块名称、产品名称、资产类别等金融实体
2. **指标**：金融指标名称，详见金融指标字典
3. **数值**：指标的数值描述，完整保留精度（如：3.14、-0.5%、454.03亿元、2,000手等）
4. **时间**：数值对应的时间信息（如：2025年6月30日、2025年第三季度、截至2025年11月30日等）

### 时间提取细则
4.1 优先关联到特定时间节点：XX年X月X日、截止/截至XX年X月X日
4.2 或关联到特定时间段：XXX年初至今、X年X月X日-XX月之间
4.3 模糊时间可作为补充：近X年/月、XX季度、这周/上周
4.4 注意整体时间描述：如"截至2025年7月7日收盘，贵州茅台的财务数据显示：..."
4.5 缺少明确时间时，根据上下文推断或标注"未明确"

## 步骤3：逐个验证四元组
对每个提取的四元组，按以下顺序进行验证：

### 3.1 定位对应的搜索结果来源
- 在搜索结果中找到与该四元组相关的内容
- 记录来源编号（如：来源1、来源2等）
- 如果多个来源包含相关信息，选择最匹配的主要来源

### 3.2 主体匹配验证
- 比对报告中的主体与搜索结果中的主体是否完全一致
- 注意主体范围：如"直销渠道"≠"总体"，"某产品"≠"某公司整体"

### 3.3 指标匹配验证（详见指标匹配规则）
- 按照指标匹配的四级标准进行判断
- 记录匹配类型：完全匹配/语义等价/包含关系/不兼容

### 3.4 数值对比验证（详见数值匹配规则）
- 精确对比数值，包括单位、精度
- 应用容差规则判断是否可接受

### 3.5 时间对比验证
- 确认时间点或时间段是否一致
- 注意语义等价的时间表达（如：2025Q3 = 2025年7-9月）

### 3.6 综合判断
- 如果所有要素都匹配，标记为无错误（error_type: null）
- 如果存在不匹配，标记错误类型和详细原因
- 在error_reason中引用具体的搜索结果来源和内容

## 金融指标字典
1. **通用指标**：[净利润、营业收入、每股收益、每股净资产、净资产收益率(ROE)、毛利率、净利率、市盈率、市净率、资产负债率、现金流、收益率、持仓占比、持仓时长、涨跌幅、成交量、成交额、换手率等]

2. **前缀**：[平均、预期、建议、XXX滚动、近XXX年、近XXX月、近XX周、近XXX日、持有XXX年、XXX占XXX比例、同比、环比、年度、季度、其他可拼接的前缀XX等]
   * 可拼接在通用指标前面，形成新指标（如："近一周收益率"、"同比增长"）

3. **后缀**：[同比上升/上涨/增长XXX、环比上升/上涨/增长XXX、xx天移动平均、历史分位数、增速、中位数、排名、XXX同类排名]
   * 可拼接在通用指标后面，形成新指标
   * 注意占比类指标：如"直销渠道占比31.9%"

## 指标匹配规则（显式指导）

### 匹配级别1：完全匹配 ✓
指标名称完全一致，无需额外判断
- 示例："净利润" = "净利润" → 无错误

### 匹配级别2：语义等价 ✓
指标名称不同但含义完全相同
- 示例："净利润增长率" = "净利润同比增长" → 无错误
- 示例："营收" = "营业收入" → 无错误
- 示例："ROE" = "净资产收益率" → 无错误

### 匹配级别3：包含关系 ⚠️
需要明确标注主体范围是否一致
- 示例："主营业务收入"是"营业收入"的子集 → 如果混用则为subject_error
- 示例："i茅台营收"是"总营收"的一部分 → 如果混用则为subject_error

### 匹配级别4：不兼容 ✗
指标含义完全不同
- 示例："净利润" ≠ "营业收入" → indicator_error
- 示例："收益" ≠ "收益率" → indicator_error
- 示例："近一周收益率" ≠ "今日收益率" → indicator_error（时间范围不同）

### 判断流程
1. 首先判断是否完全匹配或语义等价 → 如果是，继续验证其他要素
2. 然后判断是否存在包含关系 → 如果是，检查主体范围是否一致
3. 最后判断是否完全不兼容 → 如果是，标记为indicator_error

## 数值匹配规则（边界条件处理）

### 规则1：精确匹配
数值完全一致（末尾的0可以省略）
- ✓ "454.03亿元" = "454.03亿元"
- ✓ "3019.20元" = "3019.2元" （末尾0可省略）

### 规则2：单位换算等价
不同单位但数值等价
- ✓ "454.03亿元" = "45403百万元" = "4540.3千万元"
- ✓ "15%" = "0.15" （百分比与小数）
- 判断方法：换算后数值差异 < 0.01%

### 规则3：容差范围内的近似
对于"约"、"大约"等修饰词，允许小范围误差
- ✓ "约500亿" vs "498.5亿" （差异1.5% < 5%容差）
- ✗ "约500亿" vs "454亿" （差异10.1% > 5%容差）
- 规则：差异率 = |报告值 - 搜索值| / 搜索值
  - 无"约"等词：差异率必须 = 0%
  - 有"约"等词：差异率 ≤ 5%

### 规则4：负数语义等价
负数的不同表达方式
- ✓ "净利润下降15%" = "净利润增长-15%" = "净利润同比-15%"
- ✓ "亏损15亿" = "净利润-15亿元"

### 规则5：范围表达
- ✓ "15%-20%之间" 包含 "17.5%"
- ✗ "超过15%" 不等于 "15%"

### 规则6：四舍五入禁止（重要）
任何精度损失都视为value_error（除非有"约"等修饰词）
- ✗ "3019.20元" → "3千多元" （value_error：精度损失过大）
- ✗ "454.03亿元" → "454亿元" （value_error：省略小数）
- ✗ "15.8%" → "16%" （value_error：四舍五入）

## 错误类型定义

### time_error（时间错误）
时效混淆错误，包括：
- 年份错误：2024年 vs 2025年
- 季度/月份错误：Q2 vs Q3
- 时间混淆：将不同时间点的数据混淆
- 时间归属错误：将发布时间误作为数据时间

### value_error（数值错误）
数值不准确，包括：
- 数值不存在：搜索结果中未出现该数值
- 精度损失：四舍五入、省略小数
- 单位错误：未正确换算单位
- 数值混淆：将不同实体的数值混淆

### indicator_error（指标错误）
指标名称不一致，包括：
- 指标名称混淆："收益" vs "收益率"
- 指标类型错误："净利润" vs "营业收入"
- 时间范围不同："近一周" vs "今日"
- 计算方式不同："平均" vs "累计"

### subject_error（主体错误）
主体归属错误，包括：
- 实体混淆：将A公司数据归因于B公司
- 范围错误：将部分数据归因于整体（如：渠道收入 vs 总收入）
- 产品混淆：将产品A的数据归因于产品B

# 输出格式
请直接输出JSON格式，包含以下结构：
{{
  "tuples": [
    {{
      "subject": "指标的主体",
      "indicator": "金融指标名称",
      "value": "指标的数值",
      "time": "时间信息",
      "error_type": "错误类型(time_error/value_error/indicator_error/subject_error)，如果没有错误填写null",
      "error_reason": "错误原因（如果没有错误填写空字符串），必须引用搜索结果来源编号和具体内容"
    }}
  ]
}}

# Few-Shot Reasoning Examples（推理示例）

## 示例1：完全正确的四元组（思维链展示）

**报告陈述**：贵州茅台2025年上半年净利润为454.03亿元
**搜索结果**：【来源1】截至2025年6月30日，贵州茅台实现净利润454.03亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：贵州茅台 ✓ 完全匹配
3. 指标验证：净利润 ✓ 完全匹配（级别1：完全匹配）
4. 时间验证：2025年上半年 = 截至2025年6月30日 ✓ 语义等价
5. 数值验证：454.03亿元 ✓ 精确匹配（规则1）
6. 结论：所有要素均正确

**输出**：
{{"subject": "贵州茅台", "indicator": "净利润", "value": "454.03亿元", "time": "2025年上半年", "error_type": null, "error_reason": ""}}

---

## 示例2：单位换算等价（思维链展示）

**报告陈述**：招商银行截至2025年9月末资产总额为45403百万元
**搜索结果**：【来源2】2025年9月30日，招商银行资产总额454.03亿元

**推理过程**：
1. 定位来源：在【来源2】中找到相关数据
2. 主体验证：招商银行 ✓
3. 指标验证：资产总额 = 资产总额 ✓
4. 时间验证：截至2025年9月末 = 2025年9月30日 ✓
5. 数值验证：45403百万元 vs 454.03亿元
   - 换算：45403百万 = 45403/1000亿 = 45.403亿？ ✗
   - 重新计算：45403百万 = 454.03亿 ✓（45403÷100=454.03）
   - 应用规则2：单位换算等价 ✓
6. 结论：所有要素均正确，数值经单位换算后一致

**输出**：
{{"subject": "招商银行", "indicator": "资产总额", "value": "45403百万元", "time": "截至2025年9月末", "error_type": null, "error_reason": ""}}

---

## 示例3：负数语义等价（思维链展示）

**报告陈述**：宁德时代2025年第三季度净利润同比下降8.5%
**搜索结果**：【来源3】宁德时代Q3净利润同比增长-8.5%

**推理过程**：
1. 定位来源：在【来源3】中找到相关数据
2. 主体验证：宁德时代 ✓
3. 指标验证：净利润同比 ✓
4. 时间验证：2025年第三季度 = Q3 ✓
5. 数值验证：下降8.5% vs 增长-8.5%
   - 应用规则4：负数语义等价
   - "下降8.5%" = "增长-8.5%" ✓
6. 结论：所有要素均正确，负数表达方式不同但含义相同

**输出**：
{{"subject": "宁德时代", "indicator": "净利润同比", "value": "下降8.5%", "time": "2025年第三季度", "error_type": null, "error_reason": ""}}

---

## 示例4：时间语义等价（思维链展示）

**报告陈述**：比亚迪2025年前三季度营业收入为4500亿元
**搜索结果**：【来源1】比亚迪2025年1-9月实现营业收入4500亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：比亚迪 ✓
3. 指标验证：营业收入 ✓
4. 时间验证：前三季度 = 1-9月 ✓ 语义等价
5. 数值验证：4500亿元 ✓
6. 结论：所有要素均正确

**输出**：
{{"subject": "比亚迪", "indicator": "营业收入", "value": "4500亿元", "time": "2025年前三季度", "error_type": null, "error_reason": ""}}

---

## 示例5：精度保留正确（思维链展示）

**报告陈述**：易方达蓝筹精选近一年收益率为18.6%
**搜索结果**：【来源4】截至2025年12月1日，易方达蓝筹精选近一年收益率18.60%

**推理过程**：
1. 定位来源：在【来源4】中找到相关数据
2. 主体验证：易方达蓝筹精选 ✓
3. 指标验证：近一年收益率 ✓
4. 时间验证：隐含时间为截至2025年12月1日（从搜索结果推断）✓
5. 数值验证：18.6% vs 18.60%
   - 应用规则1：末尾0可省略 ✓
6. 结论：所有要素均正确

**输出**：
{{"subject": "易方达蓝筹精选", "indicator": "近一年收益率", "value": "18.6%", "time": "截至2025年12月1日", "error_type": null, "error_reason": ""}}

---

## 示例6：容差范围内近似（思维链展示）

**报告陈述**：沪深300指数成分股平均市值约500亿元
**搜索结果**：【来源2】沪深300成分股平均市值498.5亿元

**推理过程**：
1. 定位来源：在【来源2】中找到相关数据
2. 主体验证：沪深300 ✓（沪深300指数成分股）
3. 指标验证：平均市值 ✓
4. 数值验证：约500亿元 vs 498.5亿元
   - 报告使用"约"修饰词
   - 应用规则3：差异率 = |500-498.5|/498.5 = 0.3% < 5% ✓
   - 容差范围内，可接受
5. 结论：所有要素均正确

**输出**：
{{"subject": "沪深300指数成分股", "indicator": "平均市值", "value": "约500亿元", "time": "未明确", "error_type": null, "error_reason": ""}}

---

## 示例7：指标语义等价（思维链展示）

**报告陈述**：中国平安ROE为15.2%
**搜索结果**：【来源3】中国平安净资产收益率15.2%

**推理过程**：
1. 定位来源：在【来源3】中找到相关数据
2. 主体验证：中国平安 ✓
3. 指标验证：ROE vs 净资产收益率
   - 应用指标匹配级别2：语义等价
   - ROE = Return on Equity = 净资产收益率 ✓
4. 数值验证：15.2% ✓
5. 结论：所有要素均正确

**输出**：
{{"subject": "中国平安", "indicator": "ROE", "value": "15.2%", "time": "未明确", "error_type": null, "error_reason": ""}}

---

## 示例8：多来源综合验证（思维链展示）

**报告陈述**：腾讯控股2025年Q2营业收入1609.7亿元，同比增长8%
**搜索结果**：
- 【来源1】腾讯2025年第二季度总收入1609.7亿元
- 【来源2】腾讯Q2收入同比增长8%

**推理过程**：
需要提取两个四元组：
1. 营业收入数值：参考【来源1】验证 ✓
2. 同比增长：参考【来源2】验证 ✓

**输出**：
{{"subject": "腾讯控股", "indicator": "营业收入", "value": "1609.7亿元", "time": "2025年Q2", "error_type": null, "error_reason": ""}},
{{"subject": "腾讯控股", "indicator": "营业收入同比增长", "value": "8%", "time": "2025年Q2", "error_type": null, "error_reason": ""}}

---

## 示例9：数值错误-精度损失（错误示例）

**报告陈述**：贵州茅台2025年6月净利润约454亿元
**搜索结果**：【来源1】贵州茅台2025年6月30日净利润454.03亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：贵州茅台 ✓
3. 指标验证：净利润 ✓
4. 时间验证：2025年6月 = 2025年6月30日 ✓
5. 数值验证：约454亿元 vs 454.03亿元
   - 虽然报告使用"约"，但省略了".03"小数部分
   - 应用规则6：精度损失属于value_error
   - 即使有"约"，也不应省略原有精度 ✗
6. 结论：value_error

**输出**：
{{"subject": "贵州茅台", "indicator": "净利润", "value": "约454亿元", "time": "2025年6月", "error_type": "value_error", "error_reason": "【来源1】显示净利润为454.03亿元，报告省略小数部分，存在精度损失"}}

---

## 示例10：数值错误-容差超限（错误示例）

**报告陈述**：腾讯控股2025年Q2营业收入约1500亿元
**搜索结果**：【来源2】腾讯第二季度总收入1609.7亿元

**推理过程**：
1. 定位来源：在【来源2】中找到相关数据
2. 主体验证：腾讯控股 ✓
3. 指标验证：营业收入 = 总收入 ✓
4. 时间验证：2025年Q2 ✓
5. 数值验证：约1500亿元 vs 1609.7亿元
   - 报告使用"约"修饰词
   - 应用规则3：差异率 = |1500-1609.7|/1609.7 = 6.8% > 5% ✗
   - 超出容差范围
6. 结论：value_error

**输出**：
{{"subject": "腾讯控股", "indicator": "营业收入", "value": "约1500亿元", "time": "2025年Q2", "error_type": "value_error", "error_reason": "【来源2】显示Q2收入为1609.7亿元，报告数值1500亿元差异6.8%，超出5%容差范围"}}

---

## 示例11：指标错误-指标名称混淆（错误示例）

**报告陈述**：贵州茅台2025年6月30日营业收入454.03亿元
**搜索结果**：【来源1】贵州茅台2025年6月30日净利润454.03亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：贵州茅台 ✓
3. 指标验证：营业收入 vs 净利润
   - 应用指标匹配级别4：不兼容
   - "营业收入" ≠ "净利润" ✗
   - 两者是完全不同的财务指标
4. 结论：indicator_error

**输出**：
{{"subject": "贵州茅台", "indicator": "营业收入", "value": "454.03亿元", "time": "2025年6月30日", "error_type": "indicator_error", "error_reason": "【来源1】中454.03亿元是净利润，不是营业收入，指标名称混淆"}}

---

## 示例12：指标错误-时间范围不同（错误示例）

**报告陈述**：广发稳健增长混合A今日收益率0.85%
**搜索结果**：【来源3】广发稳健增长A近一周收益率0.85%

**推理过程**：
1. 定位来源：在【来源3】中找到相关数据
2. 主体验证：广发稳健增长混合A ✓
3. 指标验证：今日收益率 vs 近一周收益率
   - 应用指标匹配级别4：不兼容
   - "今日"（1天）≠ "近一周"（7天）✗
   - 时间范围不同，指标含义不同
4. 结论：indicator_error

**输出**：
{{"subject": "广发稳健增长混合A", "indicator": "今日收益率", "value": "0.85%", "time": "未明确", "error_type": "indicator_error", "error_reason": "【来源3】显示0.85%是近一周收益率，不是今日收益率，时间范围不匹配"}}

---

## 示例13：主体错误-实体混淆（错误示例）

**报告陈述**：五粮液2025年6月30日净利润454.03亿元
**搜索结果**：【来源1】贵州茅台2025年6月30日净利润454.03亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：五粮液 vs 贵州茅台 ✗
   - 完全不同的公司实体
3. 结论：subject_error

**输出**：
{{"subject": "五粮液", "indicator": "净利润", "value": "454.03亿元", "time": "2025年6月30日", "error_type": "subject_error", "error_reason": "【来源1】显示454.03亿元是贵州茅台的净利润，不是五粮液的数据，主体混淆"}}

---

## 示例14：主体错误-范围错误（错误示例）

**报告陈述**：贵州茅台2025年9月总营业收入126.92亿元
**搜索结果**：【来源2】贵州茅台i茅台平台2025年9月营业收入126.92亿元

**推理过程**：
1. 定位来源：在【来源2】中找到相关数据
2. 主体验证：总营业收入 vs i茅台平台营业收入
   - 应用指标匹配级别3：包含关系
   - "i茅台平台"是"总体"的一部分 ✗
   - 主体范围不一致
3. 结论：subject_error

**输出**：
{{"subject": "贵州茅台", "indicator": "总营业收入", "value": "126.92亿元", "time": "2025年9月", "error_type": "subject_error", "error_reason": "【来源2】显示126.92亿元是i茅台平台的收入，不是总营业收入，主体范围错误"}}

---

## 示例15：时间错误-年份错误（错误示例）

**报告陈述**：贵州茅台2024年6月30日净利润454.03亿元
**搜索结果**：【来源1】贵州茅台2025年6月30日净利润454.03亿元

**推理过程**：
1. 定位来源：在【来源1】中找到相关数据
2. 主体验证：贵州茅台 ✓
3. 指标验证：净利润 ✓
4. 数值验证：454.03亿元 ✓
5. 时间验证：2024年 vs 2025年 ✗
   - 年份不一致
6. 结论：time_error

**输出**：
{{"subject": "贵州茅台", "indicator": "净利润", "value": "454.03亿元", "time": "2024年6月30日", "error_type": "time_error", "error_reason": "【来源1】显示该数据时间为2025年6月30日，不是2024年6月30日，年份错误"}}

---

## 示例16：时间错误-季度混淆（错误示例）

**报告陈述**：中信证券2025年第二季度净资产2850亿元
**搜索结果**：【来源4】截至2025年9月30日，中信证券净资产2850亿元

**推理过程**：
1. 定位来源：在【来源4】中找到相关数据
2. 主体验证：中信证券 ✓
3. 指标验证：净资产 ✓
4. 数值验证：2850亿元 ✓
5. 时间验证：第二季度 vs 9月30日
   - 第二季度末 = 6月30日
   - 9月30日 = 第三季度末 ✗
   - 季度不一致
6. 结论：time_error

**输出**：
{{"subject": "中信证券", "indicator": "净资产", "value": "2850亿元", "time": "2025年第二季度", "error_type": "time_error", "error_reason": "【来源4】显示2850亿元对应时间为2025年9月30日（Q3末），不是第二季度末（6月30日），季度混淆"}}


---

# 搜索结果（已结构化，包含来源编号）
{search_result}

---

# 需要评估的报告
{ai_answer}

---

# 用户原始查询
{user_query}

---

# 请按照上述Chain-of-Thought流程进行思考（内部推理，不需要输出），然后直接输出JSON结果：

JSON:
"""

# English Prompt
FINANCIAL_FAITHFUL_PROMPT_EN = """# Task
You are a professional financial data analysis expert with deep expertise in financial information extraction and verification. You need to carefully analyze and extract 4-tuple information (subject, indicator, value, time) from the report and determine if it is consistent with the search results.

# Evaluation Process (Chain-of-Thought)
Please think and analyze following these steps:

## Step 1: Carefully Read Search Results
- Read each source one by one, understand the content and time context
- Identify all financial data points in the search results
- Note the data correspondence and time differences between different sources

## Step 2: Extract Tuples from Report
Extract all statements containing specific values from the report to form 4-tuples:
1. **Subject**: The subject of the indicator (company names, sector names, product names, asset categories, or other financial entities)
2. **Indicator**: Financial indicator names (see Financial Indicator Dictionary)
3. **Value**: Numeric description with full precision preserved (e.g., 3.14, -0.5%, 454.03 billion yuan, 2,000 shares)
4. **Time**: Time information corresponding to the value (e.g., June 30, 2025, Q3 2025, as of November 30, 2025)

### Time Extraction Details
4.1 Prioritize specific time points: YYYY-MM-DD, as of YYYY-MM-DD
4.2 Or specific time periods: YTD, between MM-DD and MM-DD
4.3 Fuzzy time as supplement: recent X years/months, QX, this/last week
4.4 Note overall time descriptions: "As of July 7, 2025 close, Kweichow Moutai's data shows..."
4.5 If time is unclear, infer from context or mark as "not specified"

## Step 3: Verify Each Tuple
For each extracted tuple, verify in the following order:

### 3.1 Locate Corresponding Search Result Source
- Find relevant content in search results for this tuple
- Record source number (e.g., Source 1, Source 2)
- If multiple sources contain relevant info, choose the most matching primary source

### 3.2 Subject Matching Verification
- Compare if the subject in the report matches exactly with the subject in search results
- Note subject scope: "direct sales channel" ≠ "overall", "certain product" ≠ "company overall"

### 3.3 Indicator Matching Verification (See Indicator Matching Rules)
- Judge according to the four-level indicator matching criteria
- Record match type: exact match/semantic equivalence/containment/incompatible

### 3.4 Value Comparison Verification (See Value Matching Rules)
- Precisely compare values including units and precision
- Apply tolerance rules to judge acceptability

### 3.5 Time Comparison Verification
- Confirm if time points or periods are consistent
- Note semantically equivalent time expressions (e.g., Q3 2025 = July-September 2025)

### 3.6 Comprehensive Judgment
- If all elements match, mark as no error (error_type: null)
- If mismatches exist, mark error type and detailed reason
- In error_reason, cite specific search result source and content

## Financial Indicator Dictionary
1. **General Indicators**: [Net profit, Operating revenue, Earnings per share, Net assets per share, Return on equity (ROE), Gross margin, Net margin, P/E ratio, P/B ratio, Asset-liability ratio, Cash flow, Return rate, Position ratio, Holding period, Price change, Trading volume, Trading value, Turnover rate, etc.]

2. **Prefixes**: [Average, expected, recommended, XXX rolling, recent XXX years, recent XXX months, recent XX weeks, recent XXX days, holding XXX years, XXX as percentage of XXX, YoY, MoM, annual, quarterly, other connectable prefixes, etc.]
   * Can be prefixed before general indicators to form new indicators (e.g., "recent week return rate", "YoY growth")

3. **Suffixes**: [YoY increase/rise/growth XXX, MoM increase/rise/growth XXX, XX-day moving average, historical percentile, growth rate, median, ranking, XXX peer ranking]
   * Can be suffixed after general indicators to form new indicators
   * Note percentage indicators: e.g., "direct sales channel accounts for 31.9%"

## Indicator Matching Rules (Explicit Guidance)

### Match Level 1: Exact Match ✓
Indicator names are exactly the same, no additional judgment needed
- Example: "Net profit" = "Net profit" → No error

### Match Level 2: Semantic Equivalence ✓
Different indicator names but completely same meaning
- Example: "Net profit growth rate" = "YoY net profit growth" → No error
- Example: "Revenue" = "Operating revenue" → No error
- Example: "ROE" = "Return on equity" → No error

### Match Level 3: Containment Relationship ⚠️
Need to explicitly mark if subject scope is consistent
- Example: "Main business revenue" is subset of "Operating revenue" → If mixed, subject_error
- Example: "iMoutai revenue" is part of "Total revenue" → If mixed, subject_error

### Match Level 4: Incompatible ✗
Indicator meanings are completely different
- Example: "Net profit" ≠ "Operating revenue" → indicator_error
- Example: "Return" ≠ "Return rate" → indicator_error
- Example: "Recent week return rate" ≠ "Today's return rate" → indicator_error (different time range)

### Judgment Process
1. First judge if exact match or semantic equivalence → If yes, continue verifying other elements
2. Then judge if containment relationship exists → If yes, check if subject scope is consistent
3. Finally judge if completely incompatible → If yes, mark as indicator_error

## Value Matching Rules (Boundary Condition Handling)

### Rule 1: Exact Match
Values are exactly the same (trailing 0s can be omitted)
- ✓ "45.403 billion yuan" = "45.403 billion yuan"
- ✓ "3019.20 yuan" = "3019.2 yuan" (trailing 0 can be omitted)

### Rule 2: Unit Conversion Equivalence
Different units but equivalent values
- ✓ "45.403 billion yuan" = "45403 million yuan" = "454.03 billion yuan"
- ✓ "15%" = "0.15" (percentage vs decimal)
- Judgment method: difference after conversion < 0.01%

### Rule 3: Approximation Within Tolerance
For modifiers like "approximately", allow small error range
- ✓ "Approx. 50 billion" vs "49.85 billion" (1.5% difference < 5% tolerance)
- ✗ "Approx. 50 billion" vs "45.4 billion" (10.1% difference > 5% tolerance)
- Rule: difference rate = |report value - search value| / search value
  - Without "approx": difference rate must = 0%
  - With "approx": difference rate ≤ 5%

### Rule 4: Negative Number Semantic Equivalence
Different expressions of negative numbers
- ✓ "Net profit down 15%" = "Net profit growth -15%" = "Net profit YoY -15%"
- ✓ "Loss of 1.5 billion" = "Net profit -1.5 billion yuan"

### Rule 5: Range Expression
- ✓ "Between 15%-20%" includes "17.5%"
- ✗ "Over 15%" does not equal "15%"

### Rule 6: Rounding Prohibited (Important)
Any precision loss is considered value_error (unless with "approx" modifiers)
- ✗ "3019.20 yuan" → "Over 3 thousand yuan" (value_error: excessive precision loss)
- ✗ "45.403 billion yuan" → "45.4 billion yuan" (value_error: decimal omission)
- ✗ "15.8%" → "16%" (value_error: rounding)

## Error Type Definitions

### time_error (Time Error)
Time confusion errors, including:
- Year error: 2024 vs 2025
- Quarter/month error: Q2 vs Q3
- Time confusion: mixing data from different time points
- Time attribution error: mistaking publication time as data time

### value_error (Value Error)
Value inaccuracy, including:
- Value non-existent: value not appearing in search results
- Precision loss: rounding, decimal omission
- Unit error: incorrect unit conversion
- Value confusion: mixing values of different entities

### indicator_error (Indicator Error)
Indicator name inconsistency, including:
- Indicator name confusion: "Return" vs "Return rate"
- Indicator type error: "Net profit" vs "Operating revenue"
- Different time ranges: "Recent week" vs "Today"
- Different calculation methods: "Average" vs "Cumulative"

### subject_error (Subject Error)
Subject attribution error, including:
- Entity confusion: attributing Company A's data to Company B
- Scope error: attributing partial data to overall (e.g., channel revenue vs total revenue)
- Product confusion: attributing Product A's data to Product B

# Output Format
Please directly output JSON format with the following structure:
{{
  "tuples": [
    {{
      "subject": "Subject of the indicator",
      "indicator": "Financial indicator name",
      "value": "Value of the indicator",
      "time": "Time information",
      "error_type": "Error type (time_error/value_error/indicator_error/subject_error), null if no error",
      "error_reason": "Error reason (empty string if no error), must cite search result source number and specific content"
    }}
  ]
}}

# Few-Shot Reasoning Examples

## Example 1: Completely Correct Tuple (Chain-of-Thought)

**Report Statement**: Kweichow Moutai's net profit in H1 2025 was 45.403 billion yuan
**Search Result**: 【Source 1】As of June 30, 2025, Kweichow Moutai achieved net profit of 45.403 billion yuan

**Reasoning Process**:
1. Locate source: Found relevant data in 【Source 1】
2. Subject verification: Kweichow Moutai ✓ Exact match
3. Indicator verification: Net profit ✓ Exact match (Level 1: Exact match)
4. Time verification: H1 2025 = As of June 30, 2025 ✓ Semantic equivalence
5. Value verification: 45.403 billion yuan ✓ Exact match (Rule 1)
6. Conclusion: All elements correct

**Output**:
{{"subject": "Kweichow Moutai", "indicator": "Net profit", "value": "45.403 billion yuan", "time": "H1 2025", "error_type": null, "error_reason": ""}}

---

## Example 2: Unit Conversion Equivalence

**Report Statement**: CMB total assets as of end Sep 2025 were 45403 million yuan
**Search Result**: 【Source 2】Sep 30, 2025, China Merchants Bank total assets 454.03 billion yuan

**Reasoning Process**:
1. Subject: China Merchants Bank ✓
2. Indicator: Total assets ✓
3. Time: End Sep 2025 = Sep 30, 2025 ✓
4. Value: 45403 million vs 454.03 billion
   - Conversion: 45403 million = 454.03 billion ✓
   - Apply Rule 2: Unit conversion equivalence ✓
5. Conclusion: All correct, values consistent after unit conversion

**Output**:
{{"subject": "China Merchants Bank", "indicator": "Total assets", "value": "45403 million yuan", "time": "End Sep 2025", "error_type": null, "error_reason": ""}}

---

## Example 3: Negative Number Semantic Equivalence

**Report Statement**: CATL Q3 2025 net profit YoY down 8.5%
**Search Result**: 【Source 3】CATL Q3 net profit YoY growth -8.5%

**Reasoning Process**:
1. Subject: CATL ✓
2. Indicator: Net profit YoY ✓
3. Time: Q3 2025 ✓
4. Value: Down 8.5% vs Growth -8.5%
   - Apply Rule 4: Negative semantic equivalence
   - "Down 8.5%" = "Growth -8.5%" ✓
5. Conclusion: All correct, different negative expressions with same meaning

**Output**:
{{"subject": "CATL", "indicator": "Net profit YoY", "value": "Down 8.5%", "time": "Q3 2025", "error_type": null, "error_reason": ""}}

---

## Example 4: Correct with Approximation

**Report Statement**: CSI 300 constituent stocks average market cap approximately 50 billion yuan
**Search Result**: 【Source 2】CSI 300 constituent average market cap 49.85 billion yuan

**Reasoning Process**:
1. Subject: CSI 300 constituents ✓
2. Indicator: Average market cap ✓
3. Value: Approx. 50 billion vs 49.85 billion
   - Report uses "approximately" modifier
   - Apply Rule 3: Difference rate = |50-49.85|/49.85 = 0.3% < 5% ✓
   - Within tolerance range, acceptable
4. Conclusion: All correct

**Output**:
{{"subject": "CSI 300 constituent stocks", "indicator": "Average market cap", "value": "Approximately 50 billion yuan", "time": "Not specified", "error_type": null, "error_reason": ""}}

---

## Example 5: Indicator Semantic Equivalence

**Report Statement**: Ping An Insurance ROE is 15.2%
**Search Result**: 【Source 3】Ping An Insurance return on equity 15.2%

**Reasoning Process**:
1. Subject: Ping An Insurance ✓
2. Indicator: ROE vs Return on equity
   - Apply Match Level 2: Semantic equivalence
   - ROE = Return on Equity = Return on equity ✓
3. Value: 15.2% ✓
4. Conclusion: All correct

**Output**:
{{"subject": "Ping An Insurance", "indicator": "ROE", "value": "15.2%", "time": "Not specified", "error_type": null, "error_reason": ""}}

---

## Example 6: Value Error - Precision Loss (Error Example)

**Report Statement**: Kweichow Moutai June 2025 net profit approximately 45.4 billion yuan
**Search Result**: 【Source 1】Kweichow Moutai June 30, 2025 net profit 45.403 billion yuan

**Reasoning Process**:
1. Subject: Kweichow Moutai ✓
2. Indicator: Net profit ✓
3. Time: June 2025 = June 30, 2025 ✓
4. Value: Approx. 45.4 billion vs 45.403 billion
   - Report omits ".003" decimal part
   - Apply Rule 6: Precision loss is value_error
   - Even with "approx", should not omit existing precision ✗
5. Conclusion: value_error

**Output**:
{{"subject": "Kweichow Moutai", "indicator": "Net profit", "value": "Approximately 45.4 billion yuan", "time": "June 2025", "error_type": "value_error", "error_reason": "【Source 1】shows net profit is 45.403 billion yuan, report omits decimal part, precision loss exists"}}

---

## Example 7: Value Error - Exceeds Tolerance (Error Example)

**Report Statement**: Tencent Q2 2025 operating revenue approximately 150 billion yuan
**Search Result**: 【Source 2】Tencent Q2 total revenue 160.97 billion yuan

**Reasoning Process**:
1. Subject: Tencent ✓
2. Indicator: Operating revenue = Total revenue ✓
3. Time: Q2 2025 ✓
4. Value: Approx. 150 billion vs 160.97 billion
   - Report uses "approximately" modifier
   - Apply Rule 3: Difference rate = |150-160.97|/160.97 = 6.8% > 5% ✗
   - Exceeds tolerance range
5. Conclusion: value_error

**Output**:
{{"subject": "Tencent", "indicator": "Operating revenue", "value": "Approximately 150 billion yuan", "time": "Q2 2025", "error_type": "value_error", "error_reason": "【Source 2】shows Q2 revenue is 160.97 billion yuan, report value 150 billion has 6.8% difference, exceeds 5% tolerance"}}

---

## Example 8: Indicator Error - Name Confusion (Error Example)

**Report Statement**: Kweichow Moutai June 30, 2025 operating revenue 45.403 billion yuan
**Search Result**: 【Source 1】Kweichow Moutai June 30, 2025 net profit 45.403 billion yuan

**Reasoning Process**:
1. Subject: Kweichow Moutai ✓
2. Indicator: Operating revenue vs Net profit
   - Apply Match Level 4: Incompatible
   - "Operating revenue" ≠ "Net profit" ✗
   - Completely different financial indicators
3. Conclusion: indicator_error

**Output**:
{{"subject": "Kweichow Moutai", "indicator": "Operating revenue", "value": "45.403 billion yuan", "time": "June 30, 2025", "error_type": "indicator_error", "error_reason": "【Source 1】shows 45.403 billion yuan is net profit, not operating revenue, indicator name confusion"}}

---

## Example 9: Indicator Error - Different Time Range (Error Example)

**Report Statement**: GF Steady Growth Fund A today's return rate 0.85%
**Search Result**: 【Source 3】GF Steady Growth A recent week return rate 0.85%

**Reasoning Process**:
1. Subject: GF Steady Growth Fund A ✓
2. Indicator: Today's return rate vs Recent week return rate
   - Apply Match Level 4: Incompatible
   - "Today" (1 day) ≠ "Recent week" (7 days) ✗
   - Different time ranges, different indicator meanings
3. Conclusion: indicator_error

**Output**:
{{"subject": "GF Steady Growth Fund A", "indicator": "Today's return rate", "value": "0.85%", "time": "Not specified", "error_type": "indicator_error", "error_reason": "【Source 3】shows 0.85% is recent week return rate, not today's return rate, time range mismatch"}}

---

## Example 10: Subject Error - Entity Confusion (Error Example)

**Report Statement**: Wuliangye June 30, 2025 net profit 45.403 billion yuan
**Search Result**: 【Source 1】Kweichow Moutai June 30, 2025 net profit 45.403 billion yuan

**Reasoning Process**:
1. Subject: Wuliangye vs Kweichow Moutai ✗
   - Completely different company entities
2. Conclusion: subject_error

**Output**:
{{"subject": "Wuliangye", "indicator": "Net profit", "value": "45.403 billion yuan", "time": "June 30, 2025", "error_type": "subject_error", "error_reason": "【Source 1】shows 45.403 billion yuan is Kweichow Moutai's net profit, not Wuliangye's data, subject confusion"}}

---

## Example 11: Subject Error - Scope Error (Error Example)

**Report Statement**: Kweichow Moutai September 2025 total operating revenue 12.692 billion yuan
**Search Result**: 【Source 2】Kweichow Moutai iMoutai platform September 2025 operating revenue 12.692 billion yuan

**Reasoning Process**:
1. Subject: Total operating revenue vs iMoutai platform revenue
   - Apply Match Level 3: Containment relationship
   - "iMoutai platform" is part of "overall" ✗
   - Subject scope inconsistent
2. Conclusion: subject_error

**Output**:
{{"subject": "Kweichow Moutai", "indicator": "Total operating revenue", "value": "12.692 billion yuan", "time": "September 2025", "error_type": "subject_error", "error_reason": "【Source 2】shows 12.692 billion yuan is iMoutai platform revenue, not total operating revenue, subject scope error"}}

---

## Example 12: Time Error - Year Error (Error Example)

**Report Statement**: Kweichow Moutai June 30, 2024 net profit 45.403 billion yuan
**Search Result**: 【Source 1】Kweichow Moutai June 30, 2025 net profit 45.403 billion yuan

**Reasoning Process**:
1. Subject: Kweichow Moutai ✓
2. Indicator: Net profit ✓
3. Value: 45.403 billion yuan ✓
4. Time: 2024 vs 2025 ✗
   - Year inconsistent
5. Conclusion: time_error

**Output**:
{{"subject": "Kweichow Moutai", "indicator": "Net profit", "value": "45.403 billion yuan", "time": "June 30, 2024", "error_type": "time_error", "error_reason": "【Source 1】shows data time is June 30, 2025, not June 30, 2024, year error"}}

---

## Example 13: Time Error - Quarter Confusion (Error Example)

**Report Statement**: CITIC Securities Q2 2025 net assets 285 billion yuan
**Search Result**: 【Source 4】As of September 30, 2025, CITIC Securities net assets 285 billion yuan

**Reasoning Process**:
1. Subject: CITIC Securities ✓
2. Indicator: Net assets ✓
3. Value: 285 billion yuan ✓
4. Time: Q2 vs September 30
   - End of Q2 = June 30
   - September 30 = End of Q3 ✗
   - Quarter inconsistent
5. Conclusion: time_error

**Output**:
{{"subject": "CITIC Securities", "indicator": "Net assets", "value": "285 billion yuan", "time": "Q2 2025", "error_type": "time_error", "error_reason": "【Source 4】shows 285 billion yuan corresponds to September 30, 2025 (Q3 end), not Q2 end (June 30), quarter confusion"}}


---

# Search Results (Structured with source numbers)
{search_result}

---

# Report to Evaluate
{ai_answer}

---

# Original User Query
{user_query}

---

# Please follow the Chain-of-Thought process above for thinking (internal reasoning, no output needed), then directly output the JSON result:

JSON:
"""

# Build default template from prompts
DEFAULT_FINANCIAL_FAITHFUL_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(FINANCIAL_FAITHFUL_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(FINANCIAL_FAITHFUL_PROMPT_ZH),
            ),
        ],
    },
)


# Pydantic models for structured LLM output
class TupleEvaluation(BaseModel):
    """Single financial tuple evaluation from LLM."""

    subject: str = Field(default="", description="Subject of the indicator")
    indicator: str = Field(default="", description="Financial indicator name")
    value: str = Field(default="", description="Value of the indicator")
    time: str = Field(default="", description="Time information")
    error_type: Optional[str] = Field(
        default=None, description="Error type (time_error/value_error/indicator_error/subject_error), null if no error"
    )
    error_reason: str = Field(default="", description="Error reason (empty string if no error)")


class FinancialFaithfulEvaluationOutput(BaseModel):
    """Structured output model for financial faithful evaluation LLM response."""

    tuples: List[TupleEvaluation] = Field(
        default_factory=list,
        description="提取的六元组列表",
    )


class FinancialTrajectoryFaithfulGrader(LLMGrader):
    """
    Financial deep research report factuality evaluation grader.

    Evaluates whether a financial research agent's report is faithful to the provided
    knowledge base and context, checking for hallucinations, data accuracy, unsupported claims,
    and consistency with provided information sources.

    This grader extracts financial tuples (subject, indicator, value, time, error_type, error_reason)
    from the report and checks each tuple against the search results. The score is calculated as:
    - score = 1 - (error_tuples / all_tuples)
    - Range: [0.0, 1.0], where 1.0 means all tuples are correct, 0.0 means all tuples have errors

    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        language: Language for evaluation prompts

    Example:
        >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
        >>> api = OpenAIChatModel(api_key="...", model="gpt-4o")
        >>> grader = FinancialTrajectoryFaithfulGrader(model=api)
        >>> result = await grader.aevaluate(
        ...     messages=[
        ...         {"role": "user", "content": "分析我的持仓"},
        ...         {"role": "assistant", "tool_calls": [...]},
        ...         {"role": "tool", "content": "搜索结果..."},
        ...         {"role": "assistant", "content": "您的持仓分析..."}
        ...     ]
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 or 1.0
    """

    @staticmethod
    def _create_faithful_callback(
        language: LanguageEnum = LanguageEnum.ZH,
    ) -> Callable[[Any], Dict[str, Any]]:
        """
        Create a callback function to process tuple evaluations into final score and reason.

        This callback:
        1. Extracts tuples from ChatResponse.metadata (which contains the model_dump of FinancialFaithfulEvaluationOutput)
        2. Checks if all tuples have "error_type" is None or empty string
        3. Calculates score as 1 - (error_tuples / all_tuples)
        4. Generates reason string by formatting tuples and errors

        Args:
            language: Language for generating the reason string

        Returns:
            Callable that processes ChatResponse into metadata dict with score and reason
        """

        def callback(response: Any) -> Dict[str, Any]:
            # Extract tuples from ChatResponse.metadata
            tuples_raw = []
            if response.metadata is not None:
                tuples_raw = response.metadata.get("tuples", response.metadata)

            tuples: List[TupleEvaluation] = [TupleEvaluation(**t) for t in tuples_raw]

            # If no tuples extracted, return score of 0
            if not tuples:
                return {
                    "score": 0.0,
                    "reason": (
                        "未提取到任何六元组，得分为0。"
                        if language == LanguageEnum.ZH
                        else "No tuples extracted, score is 0."
                    ),
                    "total_tuples": 0,
                    "error_count": 0,
                    "has_error": False,
                    "error_tuples": [],
                    "all_tuples": [],
                }

            # Check if all tuples have no errors (error_type is None or empty)
            has_error = False
            error_count = 0
            error_tuples = []

            for tuple_item in tuples:
                error_type = tuple_item.error_type
                if error_type is not None and error_type != "":
                    has_error = True
                    error_count += 1
                    error_tuples.append(
                        {
                            "subject": tuple_item.subject,
                            "indicator": tuple_item.indicator,
                            "value": tuple_item.value,
                            "time": tuple_item.time,
                            "error_type": error_type,
                            "error_reason": tuple_item.error_reason,
                        }
                    )

            # Calculate score as 1 - error_tuples / all_tuples
            score = 1.0 - (error_count / len(tuples))

            # Generate reason by formatting tuples
            if language == LanguageEnum.ZH:
                if has_error:
                    # Format error tuples
                    error_lines = []
                    for i, et in enumerate(error_tuples, start=1):
                        error_lines.append(
                            f"  {i}. 【{et['subject']}】{et['indicator']} = {et['value']} ({et['time']}) "
                            f"→ 错误类型: {et['error_type']}, 原因: {et['error_reason']}"
                        )
                    reason = f"发现 {error_count} 个错误的六元组（总共 {len(tuples)} 个）：\n" + "\n".join(error_lines)
                else:
                    # Format all tuples to show they're correct
                    tuple_lines = []
                    for i, t in enumerate(tuples, start=1):
                        tuple_lines.append(f"  {i}. 【{t.subject}】{t.indicator} = {t.value} ({t.time})")
                    reason = f"所有 {len(tuples)} 个六元组均无错误：\n" + "\n".join(tuple_lines)
            else:
                if has_error:
                    error_lines = []
                    for i, et in enumerate(error_tuples, start=1):
                        error_lines.append(
                            f"  {i}. [{et['subject']}] {et['indicator']} = {et['value']} ({et['time']}) "
                            f"→ Error type: {et['error_type']}, Reason: {et['error_reason']}"
                        )
                    reason = f"Found {error_count} erroneous tuples (out of {len(tuples)} total):\n" + "\n".join(
                        error_lines
                    )
                else:
                    tuple_lines = []
                    for i, t in enumerate(tuples, start=1):
                        tuple_lines.append(f"  {i}. [{t.subject}] {t.indicator} = {t.value} ({t.time})")
                    reason = f"All {len(tuples)} tuples are error-free:\n" + "\n".join(tuple_lines)

            # Convert tuples to dicts for JSON serialization
            tuples_dicts = [t.model_dump() for t in tuples]

            return {
                "score": score,
                "reason": reason,
                "total_tuples": len(tuples),
                "error_count": error_count,
                "has_error": has_error,
                "error_tuples": error_tuples,
                "all_tuples": tuples_dicts,
            }

        return callback

    def __init__(
        self,
        model: Union[BaseChatModel, dict],
        template: Optional[PromptTemplate] = DEFAULT_FINANCIAL_FAITHFUL_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize the FinancialTrajectoryFaithfulGrader.

        Args:
            model (Union[BaseChatModel, dict]): The chat model to use for evaluation.
                Can be either a BaseChatModel instance or a dictionary configuration.
            template (Optional[PromptTemplate]): The prompt template for faithful evaluation.
                Defaults to DEFAULT_FINANCIAL_FAITHFUL_TEMPLATE.
            language (LanguageEnum): Language for the evaluation prompt.
                Defaults to LanguageEnum.ZH (Chinese).

        Example:
            >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
            >>> model = OpenAIChatModel(api_key="...", model="gpt-4o")
            >>> grader = FinancialTrajectoryFaithfulGrader(model=model)
        """
        super().__init__(
            name="financial_trajectory_faithful",
            mode=GraderMode.POINTWISE,
            description="Financial deep research faithful evaluation for factuality checking",
            model=model,
            template=template,
            language=language,
            structured_model=FinancialFaithfulEvaluationOutput,
            callback=self._create_faithful_callback(language=language),
        )

    def _extract_user_query_and_search_results_from_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, str, str]:
        """
        Extract user query, search results, and AI answer from messages.

        Args:
            messages: List of message dicts (standard format).

        Returns:
            Tuple of (user_query, search_result, ai_answer)
        """
        # Filter out system messages and unwrap nested structure
        messages = [msg.get("message", msg) for msg in messages]
        non_system_messages = [msg for msg in messages if msg.get("role", "") != "system"]

        if not non_system_messages:
            return "", "", ""

        # Extract user query (first non-system user message)
        user_query = ""
        if non_system_messages[0].get("role", "") == "user":
            user_query = non_system_messages[0].get("content", "")

        # Extract final answer (last assistant message content)
        ai_answer = ""
        for msg in reversed(non_system_messages):
            if msg.get("role", "") == "assistant" and msg.get("content", ""):
                ai_answer = msg.get("content", "")
                break

        # Extract search results from tool messages with enhanced structure
        tool_messages = [
            msg.get("content", "")
            for msg in non_system_messages
            if msg.get("role", "") == "tool" and msg.get("content", "")
        ]

        search_result = ""
        if tool_messages:
            structured_results = []
            for idx, msg_content in enumerate(tool_messages, start=1):
                if self.language == LanguageEnum.ZH:
                    header = f"{'=' * 40}\n【来源 {idx}】\n{'=' * 40}"
                    structured_results.append(f"{header}\n{msg_content}\n")
                else:
                    header = f"{'=' * 40}\n【Source {idx}】\n{'=' * 40}"
                    structured_results.append(f"{header}\n{msg_content}\n")

            if self.language == LanguageEnum.ZH:
                search_result = "\n".format(len(tool_messages)) + "\n".join(structured_results)
            else:
                search_result = "\n".format(len(tool_messages)) + "\n".join(structured_results)
            logger.info(f"Structured search result with {len(tool_messages)} sources")
        else:
            logger.warning("No search result found in the messages")

        return user_query, search_result, ai_answer

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
    ) -> GraderScore:
        """
        Evaluate financial research report factuality and faithfulness.

        The evaluation extracts financial tuples from the report and checks each tuple
        for errors. The score is calculated as:
        - score = 1 - (error_tuples / all_tuples)
        - Range: [0.0, 1.0], where 1.0 means all tuples are correct

        The callback function handles tuple extraction to final score/reason conversion:
        - Extracts and validates tuples from LLM response
        - Checks each tuple for errors
        - Generates formatted reason string with tuple details

        Args:
            messages: List of messages (standard format, including system, user, assistant, tool)
                The "message" key for message can be optional.

                Example:
                ```
                [
                  {"role": "user", "content": "分析我的持仓"},
                  {"role": "assistant", "tool_calls": [{"function": {"name": "search", ...}}]},
                  {"role": "tool", "name": "search", "content": "用户持仓数据..."},
                  {"role": "assistant", "content": "您的持仓分析报告..."}
                ]
                ```

        Returns:
            GraderScore: Factuality evaluation score for the report
                - score: Continuous score [0.0, 1.0] based on error ratio
                - reason: Formatted string with tuple details and error information
                - metadata: Contains tuples, error_tuples, error_count, etc.

        Example:
            >>> result = await grader.aevaluate(
            ...     messages=[
            ...         {"role": "user", "content": "分析我的持仓"},
            ...         {"role": "tool", "content": "搜索结果..."},
            ...         {"role": "assistant", "content": "持仓分析报告..."}
            ...     ]
            ... )
            >>> print(f"Faithful Score: {result.score}")
        """
        # Extract user query, search results, and AI answer from messages
        user_query, search_result, ai_answer = self._extract_user_query_and_search_results_from_messages(messages)

        if not user_query or not search_result or not ai_answer:
            logger.warning("Empty user query or search result or AI answer, returning error")
            return GraderError(
                name=self.name,
                error="Empty user query or search result or AI answer",
            )

        try:
            # Call parent evaluation with formatted parameters
            # The callback handles tuple extraction to final score/reason conversion
            result = await super().aevaluate(
                user_query=user_query,
                search_result=search_result,
                ai_answer=ai_answer,
            )

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=result.metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            return GraderError(name=self.name, error=str(e))
