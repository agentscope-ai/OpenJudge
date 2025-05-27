# evaluation task 拆分

prompt: 用于组装prompt并调用llm获取模型评估
parser: 用于解析任务，包括llm评估解析和数据处理解析
scorer: 根据llm评估解析进行打分

llm chain: prompt -> (parser) -> scorer
rule chain: scorer


# 评估真实性
data -> [parser(抽取answer的声明), parser(抽取事实的声明)] -> prompt(根据声明进行评估) -> parser(评估结果解析) -> scorer(打分)


# 文本总结评估

data -> parser(抽取title) -> prompt(总结title是否包含'intro', 'body', and 'conclusion')  -> condition_score
{
    "no": 0,
    "yes": prompt(title顺序检验) -> condition_score{
        "Two are out of order": 0,
        "All out of order": 0,
        "Yes": prompt(判断总结好坏) -> scorer
    }
}

# 蚂小财
data -> [
    intent_score(...),
    ...
    card_score(parse(检验是否有卡片) -> condition_score{
        "无": 0,
        "有": parse(format) -> condition_score{
            "错误": 0,
            "正确": 1,
        }
    })
]