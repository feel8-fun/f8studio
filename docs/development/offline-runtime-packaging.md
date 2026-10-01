# 基础运行时离线发行调研

调研日期：2026-09-30。状态：离线基础 ZIP 已接入发行构建与验证脚本；Windows 实际运行仍需 CI 确认。当前采用脚本首次本地解包，不是 Inno Setup 安装器。

## 结论

推荐使用固定版本的 `pixi-pack` / `pixi-unpack`，将发行清单的 `studio-runtime` 环境连同 Python、第三方依赖和本地非 editable wheels 一起发行。在安装阶段完成一次本地解包，以后直接调用包内 Python。用户无需安装 Pixi、Python 或联网求解依赖，也无需使用 PyInstaller 冻结应用。

Windows 首选普通安装程序：安装程序携带基础运行时包、`pixi-unpack.exe`、Web/原生服务资源和脚本启动器，将基础环境解包到最终安装目录。可以采用 Inno Setup 等安装器，但本轮尚未验证安装器集成。用户支付的成本变为下载一次安装包和一次本地解压，不再在打开 Studio 时等待下载环境。

## 已完成的原型验证

工具：QuantCo `pixi-pack` / `pixi-unpack` **v0.7.11**，下载官方预编译二进制。使用当前源码重新生成的非 editable 发行 wheels、v7 发行锁文件和 `studio-runtime` 环境，没有打包开发 `.pixi`，也没有启用忽略 source distribution 的选项。

| 验证 | 结果 |
| --- | --- |
| Linux 命名平台 `linux-glibc228` 打包 | 成功，24 个 Conda 包、107 个 PyPI 包，400.81 MiB |
| Windows `win-64` 跨平台打包 | 成功，19 个 Conda 包、109 个 PyPI 包，305.60 MiB |
| Linux 无网络解包到带空格的新目录 | 成功，实测 4.41 秒 |
| 无网络、清空继承环境且 PATH 无 Pixi，以自带 Python `-I` 启动检查 | 成功 |
| 包内项目模块路径、内嵌 Web、`/api/health`、页面根路径 | 全部通过 |
| Linux 展开环境磁盘占用 | 约 1.4 GiB（初步包样本） |

离线验证使用 `unshare -Urn` 禁用网络。测试未依赖仓库源码导入或宿主 Python site-packages。上述体积仅为 Python 基础环境包，不包含整个发行目录中的原生服务、模型、解包工具或安装器；时间只代表本机 Linux，不代表 Windows 杀毒扫描、磁盘速度或最终首次启动时间。Windows 尚未实际解包运行；GPU 推理、完整图执行和所有外部设备也不属于本次原型验证范围。

复现命令（工具需预先下载，发行清单目录需先由 `build_runtime_manifest` 生成）：

```sh
pixi-pack -e studio-runtime -p win-64 -o base-win64.tar /path/to/release/pixi.toml
pixi-pack -e studio-runtime -p linux-glibc228 -o base-linux.tar /path/to/release/pixi.toml
pixi-unpack base-linux.tar -o '/path/to/final installation'
```

`pixi-pack` 从锁文件收集包，支持本地 wheel 路径；不支持直接打包 PyPI source distribution。正式流程应在遇到 sdist 时失败并先构建对应 wheel，不能使用 `--ignore-pypi-non-wheel` 隐藏缺失依赖。

## 发行结构与启动方式

基础包包含 Studio Web/server、普通 Python 服务、原生服务及其运行库。CUDA/cuDNN/ONNX GPU 和 MediaPipe 保持可选组件，不纳入基础启动必须完成的安装。未安装组件在 UI 中明确显示未安装，不能在启动时自动下载。

安装后保留一个基础 Python 环境，以及现有 `config`、Web（位于 server wheel 内）和 `runtime/bundles`。启动器调用环境中的 Python，例如 Windows 的 `env/python.exe -m f8studio_server`，并执行解包工具生成的激活逻辑以提供 DLL 搜索路径。正式参数应沿用现有 `studio_launch` 任务的行为，并在 Windows 实测确认。

服务入口必须同步调整：普通 Python 服务从 `pixi run -e ... TASK` 改为安装包自带 Python 加显式模块名；原生服务继续使用已部署的可执行文件与 DLL。只修改 Studio 启动器而不改服务入口，会在运行图时再次触发 Pixi 安装。

安装器应在最终目录解包，成功后才写就绪标记/快捷方式；升级使用版本化目录，完成验证后再切换入口。保留用户配置和模型数据。基础环境不假定可以随意搬动：移动安装位置时重新离线解包或重装，不复用包含旧前缀的环境。便携模式如要支持移动，需额外设计目录变化检测与本地重建。

## 与其他方式的比较

| 方式 | 适配判断 |
| --- | --- |
| `pixi-pack` + 一次离线解包 | 首选；沿用现有锁文件与 wheels，支持 Windows/Linux，可跨平台生成包 |
| `conda-pack` | 可备选；依赖已安装环境，前缀修复后不能任意再次迁移，现有 wheel+锁流程适配更弱 |
| 嵌入式 Python + wheel 安装目录 | Windows 可研究，但需自管 Python 搜索路径、Conda 来源依赖及原生 DLL，维护成本更高 |
| PyInstaller/Nuitka | 本任务不需要；会重新引入编译/收集依赖与动态导入维护成本 |
| 直接复制开发 `.pixi` | 不采用；包含 editable 源码路径和环境前缀，无法保证迁移后运行 |

## 接入正式流程前的必要验证

1. 固定打包工具版本与下载校验，构建 `studio-runtime` 离线包，将正式 CI 的打包缓存与用户运行时分开。
2. 重写基础服务入口并实现直接启动，保留参数转发、日志和异常反馈，移除基础启动路径中的 Pixi bootstrap/install。
3. Windows 干净 runner 解包到非构建目录（包括空格及非 ASCII 路径），禁止外部网络、清理 Pixi/Conda 的 PATH 后验证 Studio、Web、媒体服务和原生 DLL。
4. 验证连续两次启动、无 GPU 时基础功能、可选组件未安装状态、升级与用户数据保留。第二次启动不得解包或求解环境。
5. Windows 验证通过后接入安装器；Linux 单独验证所支持的 glibc 基线。不要将 Linux 原型通过等同于 Windows 完成。

## 资料

- [pixi-pack v0.7.11 文档](https://github.com/Quantco/pixi-pack/blob/v0.7.11/README.md)
- [pixi-pack v0.7.11 官方发布](https://github.com/Quantco/pixi-pack/releases/tag/v0.7.11)
- [conda-pack 及迁移限制](https://conda.github.io/conda-pack/)
