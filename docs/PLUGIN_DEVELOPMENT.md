# QuanLLM Harness 插件开发

插件 API 直接由 `quanllm-harness` 主包提供，不存在、也不应发布独立的插件 SDK。
插件项目将 `quanllm-harness` 声明为运行依赖，并从 `quanllm_harness.plugins` 导入稳定契约。

## 最小插件

```python
from quanllm_harness.plugins import PluginManifest
from quanllm_harness.tools import Tool


class ExamplePlugin:
    manifest = PluginManifest(
        name="example_plugin",
        version="1.0.0",
        requires_harness=">=0.1.2,<0.2",
        permissions=("tools.register",),
    )

    def setup(self, context):
        context.tools.register(
            Tool(
                name="example_plugin.double",
                description="Double one integer",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                handler=lambda args: {"value": args["value"] * 2},
            )
        )


plugin = ExamplePlugin()
```

发行包的 `pyproject.toml` 声明：

```toml
[project.entry-points."quanllm_harness.plugins"]
example_plugin = "example_plugin:plugin"
```

entry-point 名必须与 `manifest.name` 相同。Tool 名必须以
`<plugin_name>.` 开头，Verifier、Provider 和 Service 的本地名会自动加此前缀。

## Manifest

- `api_version`: 插件契约版本，当前为 `1`。
- `requires_harness`: PEP 440 版本范围。
- `permissions`: 最小权限声明。
- `dependencies`: 其他插件 ID；决定启动和关闭顺序，也控制跨插件 Service 访问。
- `execution_mode`: `in_process` 或 `subprocess`。

当前权限为 `tools.register`、`verifiers.register`、`events.subscribe`、
`services.provide`、`services.consume`、`providers.register` 和 `secrets.api_key`。
Provider 必须同时声明后两项；高权限不在默认主机允许列表中。

## 扩展点

- `context.tools.register(Tool(...))`: 确定性工具，自动写入证据来源。
- `context.verifiers.register(name, callback)`: 返回 `PluginVerificationResult`。只能添加
  `Issue`/警告/摘要，不能授予 `verified`。
- `context.providers.register(name, factory)`: 返回 `QuanLLMProvider`，并通过
  `QUANLLM_PLUGIN_PROVIDER=<plugin>.<provider>` 选择。
- `context.events.subscribe(callback)`: 订阅 `HarnessEvent`。
- `context.services.provide/get`: 插件间注入服务。

`setup(context)` 可返回一个无参数清理函数。也可实现 `request_started(question)`、
`request_finished(result, error)` 和 `shutdown()`。主机会隔离 Event/请求钩子/关闭异常并写入诊断。

## 子进程 Tool 协议

`execution_mode="subprocess"` 插件不执行 `setup`，而是提供 `subprocess_tools` 序列，
其元素为 `SubprocessToolSpec`。命令直接启动，不经 shell。标准输入是 UTF-8 JSON：

```json
{"protocol": 1, "plugin": "example", "tool": "example.double", "arguments": {"value": 3}}
```

成功响应为 `{"protocol":1,"ok":true,"result":...}`，失败响应为
`{"protocol":1,"ok":false,"error":"..."}`。完整示例见 `examples/plugin_example.py` 和
`tests/fixtures/plugin_subprocess.py`。

## 安装后管理

```bash
quanllm-harness plugins list
quanllm-harness plugins enable example_plugin
quanllm-harness plugins inspect example_plugin
quanllm-harness plugins doctor
quanllm-harness plugins validate manifest.json
quanllm-harness plugins disable example_plugin
```

默认配置位于 `~/.config/quanllm-harness/plugins.json`，可用 `QUANLLM_PLUGIN_CONFIG`
改写。`QUANLLM_PLUGINS_ENABLED` 和 `QUANLLM_PLUGINS_DISABLED` 接受逗号分隔 ID。
REST 端的 `GET /api/v1/plugins` 返回同一份 doctor 状态。

旧 `quanllm_harness.tools` entry-point 仍可用，但必须以 `legacy:<entry-name>` 显式启用，
新开发不应继续使用它。
