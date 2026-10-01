# 扩展源码仓库与 superbuild

核心源码放在 `packages/`，可选功能源码放在 `extensions/`；`external/` 留给第三方构建依赖。服务扩展的源码边界由 `config/extension-workspace.toml` 声明。核心保留 Web Studio 前端、服务端、图领域、媒体网关、媒体协议和 Python/C++ SDK。业务服务通过 SDK 和进程协议连接核心，核心实现不能直接导入扩展实现；`extensions_check` 检查这条边界。

| 源码仓库 / 原路径 | 拥有的扩展 |
| --- | --- |
| `extensions/f8screencap` | Screen Capture |
| `extensions/f8audiocap` | Audio Capture |
| `extensions/f8cvkit` | Tracking、Template Match、Video Stabilization、Dense Optical Flow、Flow Metric |
| `extensions/f8implayer` | Image / Video Player |
| `extensions/f8cppengine` | C++ Engine |
| `extensions/f8pyengine` | Python Engine |
| `extensions/f8pyscript` | Python Script、Python Expression |
| `extensions/f8pymppose` | MediaPipe Pose |
| `extensions/f8pydl` | ONNX / DL 六个服务 |
| `extensions/f8pyaudiofeat` | Audio Features、Rhythm |
| `extensions/f8proclauncher` | Process Launcher |

一个仓库可以包含多个服务；不为 cvkit 的每个入口创建一套相同依赖环境。各目录拥有 `extension.json`、服务索引、服务启动声明、源码、专属测试、模型元数据和依赖声明。描述 JSON 从构建后的真实入口生成，不在源码仓库复制描述缓存或模型权重。

当前这 11 个服务目录仍由主仓库跟踪，尚未写入新的 gitlink。独立 Git 仓库的本地导出位于 `build/extension-repositories`，用于发布前审核；远程仓库和可拉取的提交准备好后，才将原目录替换为对应 submodule。

`extensions/f8unitymods` 已是独立仓库的 submodule，提供游戏检测、Unity 插件安装和导出器，属于游戏集成扩展，而非 F8 服务进程。它目前仍通过 `f8unitymods-setup` 被 Studio 直接依赖和预装；仅移动源码目录并不意味着服务扩展管理器已支持安装、禁用或卸载它。其上游仓库、固定提交和 submodule 身份保持不变。

## 独立构建和制品

Python 包通过 `pyproject.toml` 声明 SDK 和自身依赖，不再从测试代码猜测旁边的 SDK 源码路径。安装 SDK 后可在独立仓库运行本包测试。构建 wheel 后使用 SDK 内的通用工具生成扩展 ZIP：

```bash
pixi run python -m pip wheel --no-deps --no-build-isolation -w build/wheels .
pixi run python -m f8pysdk.extension_packaging --wheel-dir build/wheels --output build/extension.zip
```

工具把本包 wheel 的模块及 `.dist-info` 放入 `python/`，将源码中的 `workspace` 声明转换成 `shared`，复用所声明的官方环境名。它不把 SDK 或依赖重复打入 wheel payload，也不替用户安装不兼容依赖。最终能否安装仍由 Studio 的环境依赖检查决定。官方环境须具备扩展要求的依赖；当前旧的完整预装环境仍含部分服务包，不能把这种导出等同于完成了最小基础发行包的裁剪。

C++ 包有自己的 CMake 入口和 Conan 配方/锁。它们通过 `find_package(f8cppsdk 0.1 CONFIG REQUIRED)` 使用安装后的 SDK，不引用 Studio 源码树的部署函数。先安装 SDK，再构建具体包：

```bash
pixi run -e cpp cpp_bootstrap
pixi run -e cpp cmake -S . -B build/sdk-only \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/Release/generators/conan_toolchain.cmake" \
  -DF8_EXTENSION_PACKAGES= -DF8_BUILD_SDK_DEMO=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/build/sdk-install"
pixi run -e cpp cmake --build build/sdk-only --parallel 2
pixi run -e cpp cmake --install build/sdk-only
# 在扩展仓库中，使用自身 Conan toolchain 和 Pixi 依赖前缀：
pixi run cmake -S . -B build/native \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/deps/conan_toolchain.cmake" \
  -DCMAKE_PREFIX_PATH="/path/to/sdk-install;$PWD/.pixi/envs/default" \
  -DCMAKE_BUILD_TYPE=Release
pixi run cmake --build build/native --parallel 2
pixi run python -m f8pysdk.extension_packaging \
  --runtime-root build/native/runtime/bundles --output build/extension.zip
```

SDK 与扩展必须使用同一平台、工具链和 Linux sysroot。不能把主机新 GCC/新 glibc 编译的静态 SDK 库交给旧部署基线的工具链链接。SDK 的依赖前缀可通过 `F8_PIXI_CPP_ENV_DIR` 显式指定；部署输出默认进入扩展的构建目录。superbuild 仍把集成输出部署到自己的 `runtime/bundles`。

两类 ZIP 都包含通用扩展目录和索引，执行真实 `--describe`、服务类及 monitor 契约验证后再落盘，并附带 `.zip.sha256`。C++ 制品携带部署后的运行库。GUI 截图、摄像头、模型推理等硬件测试仍由各仓库按平台补充，描述验证不替代功能测试。

## 导出和迁移为 submodule

先提交并发布本轮核心 SDK 改动，然后用可拉取的完整提交 SHA 导出各仓库。导出会生成各包自己的 Pixi 锁、Windows/Linux CI，以及初始本地 Git 提交；不会向 GitHub 发布。新输出目录必须不存在，避免覆盖已开始开发的独立仓库。

```bash
pixi run -e build-check python scripts/extension_workspace.py check
pixi run -e build-check python scripts/extension_workspace.py export \
  --sdk-ref <published-core-commit-sha> \
  --output-dir build/extension-repositories-v2 --git
```

`extension.json` 是各包的扩展元数据；`config/extensions.json` 是 superbuild 的合并目录和预装选择。修改包元数据后运行 `extension_workspace.py sync`，检查会拒绝漏配或重复归属。源码包的直接模块声明用于独立制品，主仓库的服务启动声明目前仍是集成环境中的 Pixi 任务入口。

各导出仓库的 CI checkout 固定 SDK 提交到 `.sdk`，只安装自身 Python / C++ 依赖，测试本包并上传扩展 ZIP。C++ SDK 使用 `with_extensions=False` 准备通信层依赖，扩展使用自己的 Conan 锁；依赖准备成功后立即保存 Conan 缓存，避免后续编译失败时丢失缓存。每个仓库维护自己的 Pixi/Conan 锁，升级 SDK 时同时更新固定提交和锁。首次导出是在当前 SDK 源码上求解的，正式发布前需用已发布 SDK 提交重新导出并核对。

初次本地导出是快照；完整历史仍在主仓库。需要保留单包历史时，先提交源目录修改，再执行 `git subtree split --prefix=extensions/<package>`，将导出的 CI/元数据提交叠加到分支上。目录迁移前的历史在旧的 `packages/<package>` 路径下，完整历史迁移需要同时处理旧路径。发布成功后，在 `extensions/<package>` 添加 submodule，并把已发布提交作为 gitlink；不要提交指向仅存在于本机的提交或 `file://` 仓库。

Pixi 的 editable 开发路径及 CMake superbuild 已使用 `extensions/`；迁移后的第一次运行需让 Pixi 更新 editable 安装。以后转换 submodule 时保留这个路径。clone/CI 使用 `git submodule update --init --recursive`。主仓库 pytest 默认只运行核心与集成测试；各扩展的单元测试显式在本包运行。独立仓库 CI 需要在正式远程仓库上验证 Windows 构建；本地 Linux 验证不能代替该步骤。
