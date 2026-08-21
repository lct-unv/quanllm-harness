# QuanLLM Harness

[![CI](https://github.com/lct-unv/quanllm-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/lct-unv/quanllm-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/quanllm-harness.svg?v=0.1.0)](https://pypi.org/project/quanllm-harness/)
[![PyPI](https://img.shields.io/pypi/v/quanllm-harness.svg?v=0.1.0)](https://pypi.org/project/quanllm-harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`quanllm-harness` 是 `QuanLLM-v2.0-qm` 的独立可靠性工程。它把模型调用、固定职责 Agent、量子工具、逐断言核验、定向修复、运行记录与一个轻量 CLI 从原 QuanLLM CLI 中分离出来。原 CLI 不会被本工程导入或修改。

```text
                     quanllm-harness
                            ↑
             ┌──────────────┼──────────────┐
             │              │              │
       QuanLLM CLI        Web UI        REST API
```

三个入口位于 `src/quanllm_harness/interfaces/`，共享同一个 `HarnessService`，不复制任何审核或工具逻辑。服务端每个请求使用独立 Provider、工具注册表、事件流和证据空间。

## 设计原则

- 采用受控 Agent DAG，不使用自由讨论式 Agent 群。
- 推理请求开启 thinking、禁用 JSON mode；结构化请求关闭 thinking、启用 JSON mode。
- 简单题走单求解器，复杂题使用两个上下文隔离的独立求解器。
- 每次工具调用执行前都经过一次语义忠实性审查；工具运行成功不自动等于断言已获支持。
- 工具证据优先于模型意见；修订稿必须重新提取断言并重新核验。
- 只有模型产生的问题清零才标记为 `verified`；基础设施或协议失败时仍尽量交付，但标记为 `degraded_delivery`。
- 用户文本始终作为待处理数据传递给内部 Agent，不能覆盖内部协议。

## 运行要求

- Python 3.10 或更高版本。
- QuanLLM 网关固定为 `http://47.97.46.74:3000/v1`，用户无需配置地址。
- API Key 只从工程根目录的 `APIKEY` 文件读取，不写入源码或运行记录。

## 安装

普通用户直接安装完整版：

```bash
python -m pip install quanllm-harness
```

该命令同时安装 CLI、Web UI、REST API 与全部量子后端，不提供按后端选择的安装方式。由于 pycommute 包含 C++ 扩展，若当前平台没有可用的预编译包，安装时需要支持 C++17 的编译器。

安装完成后，在打算运行 Harness 的目录中新建 `APIKEY` 文件：

```bash
nano APIKEY
```

在 `APIKEY` 中只填写 Key 本身，不加引号、变量名或 `export`，保存后运行：

```bash
quanllm-harness '推导一维有限深势阱偶宇称束缚态方程'
```

从源码参与开发时才需要：

```bash
cd /path/to/quanllm-harness
python -m pip install -e '.[dev]'
```

也可从标准输入读取多行问题：

```bash
quanllm-harness <<'EOF'
请推导四次微扰下谐振子基态的一阶能量修正，检查量纲并解释物理含义。
EOF
```

`--json` 输出完整机器可读结果；`--run-dir DIR` 原子化保存运行记录；`--capabilities` 不联网列出工具；`--graph` 输出默认执行图；`--total-timeout` 设置整次运行的协作式墙钟预算。

会话 CLI：

```bash
quanllm-harness --interactive
```

支持 `:again` 重跑上一问题、`:paste` 输入多行内容、`:quit` 退出。只有等待用户输入时显示“你：”，Token 统计始终位于一次回答末尾。并行 Solver 的原始思维链按 Agent 分别归组，不会按流式片段交错拼接。CLI 的每个运行过程节点显示从本次请求开始累计的 `XX小时XX分钟XX秒`。

## Web UI 与 REST API

```bash
# 对非本机访问建议设置：export QUANLLM_SERVER_TOKEN='随机长令牌'
quanllm-server --host 127.0.0.1
```

默认端口为 `3921`，打开 `http://127.0.0.1:3921/` 使用 Web UI；OpenAPI 文档位于 `/docs`。需要更换端口时仍可显式传入 `--port` 或设置 `QUANLLM_PORT`。主要接口：

- `GET /healthz`：服务与网关配置状态。
- `GET /api/v1/capabilities`：确定性工具能力。
- `GET /api/v1/graph`：默认执行图。
- `POST /api/v1/answers`：一次性 JSON 回答。
- `POST /api/v1/answers/stream`：POST 请求上的 SSE 事件、原始思维链与终稿。

若设置了 `QUANLLM_SERVER_TOKEN`，两个回答接口要求 `Authorization: Bearer <token>`；健康、能力、执行图和本地 Web 静态资源保持可读。API 不接收或回传网关 API Key。Web UI 的服务器令牌只保存在当前输入框内，不写入浏览器存储。Web UI 按 Agent 建立独立思维链面板，同时保留各自的完整原始输出。执行过程框采用固定高度并在内部纵向滚动；计时器持续刷新。SSE 的每个事件和终止消息都包含数值型 `elapsed_seconds` 与 `XX小时XX分钟XX秒` 格式的 `elapsed`。

## 快速使用

```python
from quanllm_harness import HarnessSettings, OpenAIQuanLLMProvider, QuanLLMHarness

settings = HarnessSettings.from_api_key_file()
provider = OpenAIQuanLLMProvider(settings)
harness = QuanLLMHarness(provider=provider, settings=settings)

result = harness.answer("推导一维有限深势阱偶宇称束缚态方程")
print(result.status.value)
print(result.answer)
```

可通过 `create_harness(settings, event_sink=...)` 订阅类型化事件，查看各 Agent 的流式推理、工具证据、修复过程和用量。`HarnessResult.to_dict()` 可直接序列化；运行记录会移除密钥。

调用方可以在阶段边界取消长任务：

```python
from quanllm_harness import CancellationToken, create_harness

token = CancellationToken()
harness = create_harness(settings)
# 可从另一个线程调用 token.cancel()
result = harness.answer("问题", cancellation=token)
```

模型请求本身另有独立的读取和流式总时限，因此取消不依赖无限等待的网络请求结束。

## 受控 Agent 流程

深题和高风险题默认让两个隔离 Solver 并行作答，再由综合 Agent 生成候选终稿。Solver 与逐断言工具计划产生的每个调用都会先由一次性审查 Agent 核对工具领域和参数忠实性，再进入确定性执行。此后系统执行形式/学科审核、要求/教学审核、逐问题独立裁决，以及有界定向修复和全量复核。

审核 Agent 只提出问题，不拥有“放行权”。Python 编排器检查 JSON Schema、引文可定位性、证据 ID、问题来源、重复问题和收敛状态。所有问题即使被两个审核器同时报告，也必须经过恰好一次独立裁决；裁决协议失败只产生警告，不能触发答案重写。只有协议完整且模型产生的问题归零才返回 `verified`。

四种状态含义：

- `verified`：完整核验通过。
- `verified_with_input_ambiguity`：只剩用户原始输入自身的歧义或损坏。
- `degraded_delivery`：仍交付答案，但协议、基础设施或收敛不满足“已核验”标准。
- `failed_without_answer`：求解阶段未能产生可交付答案。

## 默认量子后端

- SymPy：通用符号数学、矩阵、角动量系数，以及默认可用的结构化算符代数。
- pycommute：玻色、费米和自旋算符代数。
- QuTiP：有限维量子态、密度矩阵和算符数值核验。
- OpenFermion：费米/玻色产生湮灭算符的正规序、共轭、对易与反对易运算。

普通安装会同时安装 Web/REST 服务及上述全部量子后端，不提供按后端选择的安装选项。`operator_algebra` 可精确核验一维正则位置—动量对、角动量、单模玻色和单模费米算符的对易子、反对易子、乘积与厄米共轭。它使用结构化算符树而非自然语言正则或可交换标量解析；内置工具遇到不同代数族或多模关系时会明确拒绝，并由专用后端处理，不会以有限维矩阵代替无限维证明。

Harness 还提供基于 mpmath 的高精度数值积分、局部求根与截断收敛检查。数值结果会携带精度或局限说明，不能冒充解析证明。

若能力状态显示某个默认后端缺失，说明安装环境不完整，应重新安装发行包。SymPy 标量工具会在执行前拒绝 ket/bra 和抽象产生湮灭算符，避免把非交换量误当普通变量得到“看似成功”的错误证据。

第三方发行包可以通过 `quanllm_harness.tools` Python entry-point 返回一个 `Tool` 或 `Tool` 序列。插件在当前 Python 进程中执行，只应安装经过审查的可信插件。

## 工程结构

```text
src/quanllm_harness/
├── interfaces/      # CLI、Web UI、REST API 与隔离服务
├── config/          # 模型请求 profile 与不可变运行设置
├── contracts/       # 请求、断言、证据、事件和结果契约
├── providers/       # Provider 抽象及 QuanLLM-v2 实现
├── agents/          # 路由、双 Solver、综合、审核和修复职责
├── protocols/       # JSON、断言提取和判卷协议
├── orchestration/   # 执行图、取消/预算、收敛与主编排器
├── tools/           # 注册表及 SymPy/数值/量子后端
├── verification/    # 结构、数学、语义与聚合核验
├── events/          # 线程安全事件流
└── public_api.py
```

测试按 `tests/unit`、`tests/integration`、`tests/regression` 和 `tests/fixtures` 分层。接口集成测试覆盖 Web 静态资源、健康检查、Bearer Token、同步回答和 SSE 事件终稿。旧版扁平模块仍保留轻量兼容导入，但新代码应使用上述包路径。

## 开发验收

```bash
python -m pip install -e '.[dev]'
ruff format --check .
ruff check .
mypy
pytest --cov=quanllm_harness
python -m build
python -m twine check dist/*
```

测试覆盖 Provider 模式互斥、并行 Solver、结构化协议重试、执行图与收敛、工具调用前审查、缓存证据去重、复数矩阵比较、量纲与边界条件、Fock 算符矩、数值后端、默认量子后端、插件发现、运行记录、协议回归 fixtures 和离线 CLI。流式接入见 [examples/streaming_answer.py](examples/streaming_answer.py)，架构细节见 [ARCHITECTURE.md](ARCHITECTURE.md)，发版步骤见 [RELEASING.md](RELEASING.md)。

## 共同贡献者

- [fanfan32123](https://github.com/fanfan32123)
- [Hxttt1](https://github.com/Hxttt1)

## 许可证

本工程使用 [MIT License](LICENSE)。
