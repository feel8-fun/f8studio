# 扩展源码仓库与 superbuild

当前仓库是开发工作区：launcher 源码放在 `platform/`，应用和服务源码放在 `extensions/`；`external/` 留给第三方构建依赖。`sdk/` 是 [feel8-fun/f8sdk](https://github.com/feel8-fun/f8sdk) 的 submodule，拥有 Python/C++ SDK、共享协议、生成工具和跨语言测试。`cloud/` 是 [feel8-fun/f8assetcloud](https://github.com/feel8-fun/f8assetcloud) 的 submodule，独立拥有云端资产 API、认证、D1 迁移和管理前端。服务扩展的源码边界由 `config/extension-workspace.toml` 声明。WebStudio 和媒体网关是独立应用扩展；WebStudio 前后端一起发布。媒体协议属于 SDK。launcher 不导入应用实现，应用通过 SDK 和进程协议连接；`extensions_check` 检查这条边界。

| 源码目录 / 独立仓库 | 拥有的扩展 |
| --- | --- |
| [`extensions/f8screencap`](https://github.com/feel8-fun/f8screencap) | Screen Capture |
| [`extensions/f8audiocap`](https://github.com/feel8-fun/f8audiocap) | Audio Capture |
| [`extensions/f8cvkit`](https://github.com/feel8-fun/f8cvkit) | Tracking、Template Match、Video Stabilization、Dense Optical Flow、Flow Metric |
| [`extensions/f8implayer`](https://github.com/feel8-fun/f8implayer) | Image / Video Player |
| [`extensions/f8cppengine`](https://github.com/feel8-fun/f8cppengine) | C++ Engine |
| [`extensions/f8pyengine`](https://github.com/feel8-fun/f8pyengine) | Python Engine、Python Script、Python Expression |
| [`extensions/f8pymppose`](https://github.com/feel8-fun/f8pymppose) | MediaPipe Pose |
| [`extensions/f8pydl`](https://github.com/feel8-fun/f8pydl) | ONNX / DL 六个服务 |
| [`extensions/f8pyaudiofeat`](https://github.com/feel8-fun/f8pyaudiofeat) | Audio Features、Rhythm |
| [`extensions/f8proclauncher`](https://github.com/feel8-fun/f8proclauncher) | Process Launcher |

一个扩展包对应一个环境，可以包含多个服务和工具；不为 cvkit 的每个入口创建一套相同依赖环境。各目录拥有 `extension.json`、服务索引、服务启动声明、源码、专属测试、模型元数据和依赖声明。描述 JSON 从构建后的真实入口生成，不在源码仓库复制描述缓存或模型权重。

独立服务仓库发布到 `https://github.com/feel8-fun/<目录名>`，各目录通过 HTTPS submodule 固定到已发布的提交。CI 使用各仓库工作流中固定的已发布 SDK 完整提交 SHA，独立维护 Pixi 锁、Conan 锁和 Windows/Linux 构建。主仓库提交记录集成使用的 gitlink，更新扩展时需先推送扩展提交，再提交新的 gitlink。

首次 clone 和旧工作区更新：

```bash
git clone --branch dev --recurse-submodules https://github.com/feel8-fun/f8studio.git
# 已有工作区在 pull 主仓库之后：
git submodule sync --recursive
git submodule update --init --recursive
```

开发某个扩展时，在其目录创建工作分支，提交并推送到对应独立仓库；随后在主仓库运行 `extensions_check`、提交并推送 gitlink。不要直接在 submodule 的 detached HEAD 上遗留未发布的提交。主仓库 CI 已使用 recursive checkout。

`extensions/f8unitymods` 已是独立仓库的 submodule，提供游戏检测、Unity 插件安装和导出器，属于游戏集成扩展，而非 F8 服务进程。Studio 已移除对 `f8unitymods-setup` 的直接依赖和专用检测、安装入口；游戏工具需要由扩展声明 tools、skills 和 resources 后，通过通用扩展管理和执行接口提供。具体 Unity 工具迁移尚待扩展仓库实现。其上游仓库、固定提交和 submodule 身份保持不变。

Cloud 的版本、开发和集成流程见 [Cloud 独立仓库](cloud-repository.md)。

## 独立构建和制品

Python 包通过 `pyproject.toml` 声明 SDK 和自身依赖，不再从测试代码猜测旁边的 SDK 源码路径。安装 SDK 后可在独立仓库运行本包测试。构建 wheel 后使用 SDK 内的通用工具生成扩展 ZIP：

```bash
pixi run python -m pip wheel --no-deps --no-build-isolation -w build/wheels .
pixi run python -m f8pysdk.extension_packaging --wheel-dir build/wheels --output build/extension.zip
```

工具把本包 wheel 的模块及 `.dist-info` 放入 `python/`，将源码中的 `workspace` 声明转换成 `shared`，复用所声明的官方环境名。它不把 SDK 或依赖重复打入 wheel payload，也不替用户安装不兼容依赖。最终能否安装仍由 Studio 的环境依赖检查决定。官方环境须具备扩展要求的依赖；当前旧的完整预装环境仍含部分服务包，不能把这种导出等同于完成了最小基础发行包的裁剪。

C++ 包有自己的 CMake 入口和 Conan 配方/锁。它们通过 `find_package(f8cppsdk 0.1 CONFIG REQUIRED)` 使用安装后的 SDK，不引用 Studio 源码树的部署函数。先安装 SDK，再构建具体包：

```bash
git clone https://github.com/feel8-fun/f8sdk.git .sdk
git -C .sdk checkout <published-sdk-commit-sha>
# 使用 SDK CI 中的锁定 Conan 安装、CMake 构建和安装命令。
# 在扩展环境中构建 SDK 时，额外传入 -DF8SDK_BUILD_TESTS=OFF，
# 并将 F8_PIXI_CPP_ENV_DIR 指向扩展自己的 Pixi 依赖前缀。
# 在扩展仓库中，使用自身 Conan toolchain 和 Pixi 依赖前缀：
pixi run cmake -S . -B build/native \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/deps/conan_toolchain.cmake" \
  -DCMAKE_PREFIX_PATH="/path/to/sdk-install;$PWD/.pixi/envs/default" \
  -DCMAKE_BUILD_TYPE=Release
pixi run cmake --build build/native --parallel 2
pixi run python -m f8pysdk.extension_packaging \
  --runtime-root build/native/runtime/bundles --output build/extension.zip
```

SDK 与扩展必须使用同一平台、工具链和 Linux sysroot。不能把主机新 GCC/新 glibc 编译的静态 SDK 库交给旧部署基线的工具链链接。SDK 的依赖前缀可通过 `F8_PIXI_CPP_ENV_DIR` 显式指定；部署输出默认进入扩展的构建目录。开发 workspace 把集成输出部署到 `build/workspace/runtime/bundles`。

两类 ZIP 都包含通用扩展目录和索引，执行真实 `--describe`、服务类及 monitor 契约验证后再落盘，并附带 `.zip.sha256`。C++ 制品携带部署后的运行库。GUI 截图、摄像头、模型推理等硬件测试仍由各仓库按平台补充，描述验证不替代功能测试。

## 新增和维护扩展仓库

现有扩展直接在独立仓库和 submodule 中维护。新增扩展时，在独立仓库维护源码、元数据、SDK 固定提交、依赖锁和 CI；发布提交后，将仓库作为 submodule 添加到 `extensions/<package>`，并在 `config/extension-workspace.toml` 声明归属。

```bash
pixi run -e build-check python scripts/extension_workspace.py sync
pixi run -e build-check python scripts/extension_workspace.py check
```

`extension_workspace.py` 负责生成开发目录、启动适配、模型元数据副本和归属检查。

`extension.json` 和包内 `config/services` 是各包声明的唯一来源；根目录 `config/extension-workspace.toml` 维护源码归属和开发预装选择。修改声明后运行 `extension_workspace.py sync`，在 `build/workspace/config` 生成汇总清单和开发启动适配，检查会拒绝漏配或重复归属。扩展的直接模块声明用于独立制品，开发适配从它生成 Pixi 任务入口。

模型描述和默认资源随对应扩展维护；下载权重及用户模型 YAML 存放在 `${F8_MODEL_ROOT}`，默认使用平台用户数据目录。workspace 清理和扩展卸载保留这些用户数据。

各独立仓库的 CI checkout `feel8-fun/f8sdk` 的固定提交到 `.sdk`，不再下载 Studio 仓库。Python 依赖路径是 `.sdk/python`。C++ SDK 使用自己的精简 Conan 配方和锁，扩展使用本包的 Conan 锁；依赖准备成功后立即保存 Conan 缓存，避免后续编译失败时丢失缓存。每个仓库维护自己的 Pixi/Conan 锁，升级 SDK 时同时更新固定提交和锁。生成锁时的 SDK 源码须与固定提交一致。

生成 Pixi lock 时，`.sdk` 必须使用真实目录 checkout 或源码副本。不要用指向开发 workspace 其他仓库的 symlink：Pixi 会把链接目标写入 lock，导致独立 checkout 和 CI 找不到依赖。WebStudio 通过 SDK 客户端连接 Launcher，无需 `.platform` 源码依赖。

SDK 自己的 CI 负责 Python/C++ 单元测试、协议生成校验、跨语言通信及安装后 CMake 包验证，上传 wheel 和 CMake SDK 制品。主仓库的默认 quality 检查只负责 Studio 核心及集成测试；需要单独运行 SDK 测试时使用 `pixi run -e build-check pytest_sdk`。共享协议的唯一来源是 `sdk/schemas/`，Studio 私有 HTTP/document 合同归 WebStudio 子仓库所有。原 `packages/f8sdk_demo` 已精简为 SDK 内的 `cpp/examples/minimal_service`，默认不构建、不进入发行包。

更新 submodule 前先发布独立仓库提交，再提交主仓库 gitlink；不要提交指向仅存在于本机的提交或 `file://` 仓库。

Pixi 的 editable 开发路径及 CMake superbuild 使用 `extensions/`，转换 submodule 后路径保持一致。主仓库 pytest 默认只运行核心与集成测试；各扩展的单元测试在本包运行。Windows/Linux 的独立 CI 随仓库 push 自动触发；本地 Linux 验证不能代替 Windows 构建。

`extensions/f8diagnostics` 是 `feel8-fun/f8diagnostics` 的 Git submodule，作为通用诊断工具集独立版本管理，当前为 0.2.0。它没有 node/service，工具声明来自自己的 `extension.json`；Studio 通过通用工具任务接口执行，发送和验证可并行，持续发送由 Stop 或 Studio 关闭结束。
