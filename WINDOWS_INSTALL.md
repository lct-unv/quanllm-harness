# Windows installation / Windows 安装

[中文](#中文) | [English](#english)

## 中文

### 为什么需要专用安装器

QuanLLM Harness 默认包含 pycommute 量子算符后端。pycommute 1.0.0 在 PyPI 上目前只有源码包，且其上游源码有三处影响 Windows/MSVC 构建的问题：MSVC 需要显式更新 `__cplusplus`，一个比较器模板对 MSVC STL 不兼容，两个扩展模块使用了非 Windows 的路径式名称。

`install.ps1` 保留官方 pycommute 发行内容，只在经过 SHA-256 校验的临时副本中应用最小兼容补丁。它不会修改 Python 安装目录中的源码，也不会永久修改 `CL` 环境变量。

### 前置要求

- 64 位 Windows 10 或 Windows 11。
- Python 3.10 或更高版本，可通过 `py` 启动。
- Visual Studio Build Tools，并安装“使用 C++ 的桌面开发”工作负载。
- 能够通过当前 pip 索引下载 PyPI 包。

不应卸载 MSVC 或用 MinGW/GCC 替换官方 Windows CPython 的构建工具链。

### 安装

在仓库根目录打开 PowerShell，然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

默认安装 QuanLLM Harness 0.1.2。也可显式指定版本和 Python 启动器：

```powershell
.\install.ps1 -HarnessVersion 0.1.2 -PythonLauncher py
```

安装器将依次：

1. 使用 `pip download` 下载 pycommute 1.0.0 源码包；
2. 校验发布文件的 SHA-256；
3. 在独立临时目录修补 MSVC C++17 检测、比较器与扩展模块名称；
4. 构建并安装修补后的 pycommute；
5. 安装指定版本的 QuanLLM Harness；
6. 验证包版本与费米反对易关系。

成功后临时目录会被清理。失败时目录会保留，终端会显示其位置，便于收集完整构建日志。

### 常见问题

- `Microsoft Visual C++ 14.0 or greater is required`：安装或修复 Visual Studio Build Tools 的“使用 C++ 的桌面开发”工作负载。
- 下载失败：检查 `py -m pip config list` 显示的索引和代理设置；不要通过关闭 TLS 验证处理证书问题。
- 源码哈希不匹配：停止安装，不要运行未经验证的源码；确认 pip 索引没有替换或重新打包上游文件。
- 构建失败：保留安装器给出的临时目录，并提交从首个 `error` 开始的完整日志、Windows 版本、Python 版本和 MSVC 版本。

## English

### Why a dedicated installer is required

QuanLLM Harness includes the pycommute quantum-operator backend by default. PyPI currently provides
pycommute 1.0.0 only as a source archive, and its upstream source has three Windows/MSVC build
problems: MSVC needs an explicit updated `__cplusplus` value, one comparator template is
incompatible with the MSVC standard library, and two extension modules use non-Windows path-style
names.

`install.ps1` preserves the official pycommute release content and applies minimal compatibility
patches only to a temporary copy after verifying its SHA-256 digest. It does not modify source files
inside the Python installation and does not permanently change the `CL` environment variable.

### Prerequisites

- 64-bit Windows 10 or Windows 11.
- Python 3.10 or newer, available through the `py` launcher.
- Visual Studio Build Tools with the **Desktop development with C++** workload.
- Access to PyPI packages through the active pip index.

Do not uninstall MSVC or replace the official Windows CPython toolchain with MinGW/GCC.

### Installation

Open PowerShell in the repository root and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The default Harness version is 0.1.2. The version and Python launcher may also be explicit:

```powershell
.\install.ps1 -HarnessVersion 0.1.2 -PythonLauncher py
```

The installer performs these steps:

1. downloads the pycommute 1.0.0 source archive with `pip download`;
2. verifies the published SHA-256 digest;
3. patches MSVC C++17 detection, the comparator, and extension names in an isolated temporary tree;
4. builds and installs the patched pycommute package;
5. installs the selected QuanLLM Harness version; and
6. verifies package versions and the canonical fermionic anticommutator.

The temporary directory is removed after success. On failure it is preserved and printed to the
terminal so the complete build log can be collected.

### Troubleshooting

- `Microsoft Visual C++ 14.0 or greater is required`: install or repair the **Desktop development
  with C++** workload in Visual Studio Build Tools.
- Download failure: inspect the index and proxy settings reported by `py -m pip config list`. Do not
  work around certificate problems by disabling TLS verification.
- Source digest mismatch: stop the installation and do not run unverified source. Check whether the
  configured pip index has replaced or repackaged the upstream file.
- Build failure: keep the temporary directory reported by the installer and provide the complete log
  from the first `error`, together with the Windows, Python, and MSVC versions.
