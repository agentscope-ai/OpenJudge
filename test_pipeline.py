from src.data.data_schema import ContentDict, ContextDict, DataSample
from src.model.base import LLMllm
from src.rm.base import Var, VarType
from gallery.rm.code.rm import DataRM, LLMRM
from src.rm.schema import BestOfN, ViolatedPrinciples
from src.rm.template import EvaluationTemplate

llm = LLMllm(model="qwen-max")


data_parser = DataRM(
    name="data"
)

fact_parser = LLMRM(
    llm=llm,
    name="pointwise",
    desc="你是严谨性评估专家，请评估理财师回答的严谨性，确保理财师回答没有出现明显错误。",
    output=ViolatedPrinciples,
    template=EvaluationTemplate,
    principles="""1. 回答必须完全基于现有可用参考材料，所有观点和数据必须来源于参考资料，禁止使用参考材料未提供的来自经验知识的观点、数据或者结论。
2. 在使用数据相关内容时，避免无谓计算推理，除非用户明确要求，同时数据单位、周期必须与参考资料保持一致，禁止任何形式的单位和周期转换。
3. 在使用带时间维度的相关内容时，应严格区分并准确表述数据的日期或统计周期，避免混用不同日期或统计周期的数据，同时明确用户所处的当前时间，确保对“当前”“昨天”等时间指代词使用的准确性。
4. 应确保金融实体理解准确，禁止使用错误的因果关系推出错误的结论，避免对不同实体的各类数据和观点进行混淆。
    - 不同实体间的观点和数据不能互相使用。
    - 在进行推论时确保论据和论点描述的实体一致，例如收益率好不能推出回撤风险好。
    - 板块A包含基金B和结论C不能推理得到基金B包含结论C。
    - 时间B的实体A包含结论D不能得到时间C的实体A包含结论D。
""",
    input=[
        Var(name="actual_output", path="reward_contexts.data.actual_output"),
        Var(name="context", path="reward_contexts.data.context", vtype=VarType.INPUT),
    ]
)


list_parser = LLMRM(
    llm=llm,
    name="list_parser",
    desc="您是一位资深的理财内容评估专家，负责根据评分细则对多个回答进行评分和比较。",
    output=BestOfN,
    template=EvaluationTemplate,
    principles="""评估回答的表达清晰度和易懂程度。优质回答应以通俗易懂的语言解释专业概念，结构清晰，便于用户理解。""",
    input=[
        Var(name="actual_output", path="reward_contexts.data.actual_output"),
        Var(name="context", path="reward_contexts.data.context", vtype=VarType.INPUT),
    ]
)


sample = DataSample(
    input=[
        ContentDict(
            role="user", content="今天是几号"
        )
    ],
    outputs=[
        ContentDict(
            role="assistant", content="今天是周四"
        ),
        ContentDict(
            role="assistant", content="今天是周二"
        ),
        ContentDict(
            role="assistant", content="今天是2025年5月20号，周二"
        ),
    ],
    contexts=[
        ContextDict(
            context_type="str", context="当前日期：2025年5月20号，周二"
        )
    ]
)


data_parser.run_pointwise(sample)
fact_parser.run_pointwise(sample)
list_parser.run_listwise(sample)

print(sample)