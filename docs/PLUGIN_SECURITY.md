# 插件安全模型

## 信任边界

已启用的 `in_process` 插件是受信 Python 代码：它与 Harness 同权限，能够读写进程可访问的
文件、环境变量和网络。Manifest 权限是主机 API 的授权门，不是恶意 Python 代码的
OS 级沙箱。因此只应启用已审查的进程内插件。

`subprocess` 模式提供更窄的协议边界：无 shell、收窄环境变量、调用超时、请求/输出
字节上限。它仍然可以继承当前用户的文件系统和网络能力。对不可信代码，请另外使用
非特权账户、容器、seccomp/AppArmor/SELinux 或平台沙箱，并显式关闭不需要的网络与目录。

## 默认防护

- 默认 `allow_unlisted=false`；未启用入口只列出元数据，不进行 Python 导入。
- entry-point 名与 manifest ID 必须一致，所有扩展名受命名空间约束。
- API 版本、Harness 版本范围、权限和依赖图在 `setup` 前验证。
- 可开启 `require_trusted_digest`，在导入前对已安装发行内容计算 SHA-256。
- Provider 需要 `providers.register` 和 `secrets.api_key` 双重权限，且默认不授权。
- Verifier 没有“验证通过”接口；最终状态只由核心引擎决定。
- Tool Evidence 携带插件名、版本、发行摘要和执行模式，便于审计。
- Event、Verifier、请求钩子与关闭回调异常不会让核心服务崩溃。

## 建议上线流程

1. 锁定插件发行版本，审查源码和依赖。
2. 安装后运行 `plugins list`，获取发行摘要；将其写入 `trusted_digests`。
3. 将 `allowed_permissions` 限制到必需集合，并保持 `allow_unlisted=false`。
4. 启用插件后运行 `plugins doctor`，再运行隔离环境的功能测试。
5. 发布或重装后重新校验摘要；摘要变化应视为需要重审的供应链事件。

## 配置示例

```json
{
  "enabled": ["example_plugin"],
  "disabled": [],
  "allow_unlisted": false,
  "allowed_permissions": ["tools.register", "events.subscribe"],
  "trusted_digests": {"example_plugin": "<64-character-sha256>"},
  "require_trusted_digest": true,
  "call_timeout_seconds": 10,
  "max_output_bytes": 262144,
  "config": {"example_plugin": {"mode": "strict"}}
}
```
