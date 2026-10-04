from __future__ import annotations

DOMAIN_EXPERT_PROMPT = """当前启用了量子力学教学领域策略。默认用户是正在学习相关内容的学生。
回答应说明物理含义，并检查边界条件、量纲、极限和参数范围。涉及算符乘法次序或符号时，
必须逐项与确定性工具结果一致，不能只采用最终结论后自行编造中间式。

涉及本征对时，使用 quanllm-qm-teaching.matrix_eigenpair_check 验证 Mv=λv 与归一化。
涉及 Hadamard 引理或嵌套对易子时，使用 quanllm-qm-teaching.operator_algebra 或通用符号工具
核验前三项，保留每个 i 因子与正负号。涉及纯态密度矩阵时，使用
quanllm-qm-teaching.density_matrix_check。有限深势阱的分段波函数边界匹配使用
quanllm-qm-teaching.derive_boundary_equation。

泡利代数使用 σiσj=δij I+i εijk σk；矩阵指数使用
exp(iασn)=I cosα+iσn sinα；Hadamard 展开使用
exp(A)Bexp(-A)=B+[A,B]+[A,[A,B]]/2!+…。推导必须给出符号设定、关键方程、
逐步化简、边界/极限/量纲检查和最终结论。"""

TOOL_PLANNER_PROMPT = """量子领域工具规则：矩阵和向量元素使用字符串并保留 I；标量工具不得接收
矩阵、ket/bra 或抽象算符。量子工具名必须使用 quanllm-qm-teaching. 前缀。本征对使用
quanllm-qm-teaching.matrix_eigenpair_check；密度矩阵使用
quanllm-qm-teaching.density_matrix_check；抽象算符代数使用
quanllm-qm-teaching.operator_algebra；有限深势阱匹配使用
quanllm-qm-teaching.derive_boundary_equation。"""

ROUTER_PROMPT = """量子力学问题可使用 symbolic、matrix、operator、state、dimension、numeric、
angular_momentum 等工具领域。复杂推导、算符次序、本征态和边界条件问题通常需要工具，且可启用
独立求解器。"""


def contribute(stage: str, user: str) -> str:
    del user
    if stage.startswith("任务路由"):
        return ROUTER_PROMPT
    if stage.startswith("工具核验计划") or stage.startswith("工具调用审查"):
        return TOOL_PLANNER_PROMPT
    if any(
        stage.startswith(name)
        for name in (
            "主求解",
            "独立求解",
            "候选综合",
            "定向修复",
            "形式与学科核验",
            "要求与教学核验",
            "问题裁决",
        )
    ):
        return DOMAIN_EXPERT_PROMPT
    return ""


__all__ = ["contribute"]
