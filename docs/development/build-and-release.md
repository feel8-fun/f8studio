# 开发、构建与发行

开发环境保留 Python editable 安装；发行环境安装固定的 wheel。修改 Python 实现后无需重新安装，重启对应服务即可生效；只有修改依赖或包元数据时才需要重新同步环境。不要用发行环境调试源码，也不要把 `.pixi` 打包给用户。

功能服务包的独立构建、扩展 ZIP、独立 CI 与 submodule 迁移流程见[扩展源码仓库与 superbuild](extension-repositories.md)。主仓库的默认 pytest 范围为核心与集成测试；各功能包的单元测试由其独立仓库运行。

## 目录职责

| 目录 | 内容 |
| --- | --- |
| `packages/` | 核心源码、SDK、包定义和受版本控制的资源 |
| `extensions/` | 可选服务与游戏集成扩展源码；独立仓库可通过 submodule 接入 |
| `external/` | 第三方构建依赖 |
| `.pixi/` | 本机开发环境，Python 包 editable 指向源码 |
| `build/Release/` | Conan/CMake 配置、对象文件、原生库和可执行文件 |
| `build/web-studio/` | Vite 生产静态资源 |
| `build/wheel-staging/` | 可删除的 Python 包副本；仅在此嵌入 Web 资源 |
| `runtime/bundles/` | 本机已部署的版本化服务运行产物，不是编译工作目录 |
| `config/` | 服务索引和启动声明 |
| `build/dist/` | 完整发行目录、启动脚本和压缩包 |
| `build/release-smoke/` | 使用 `--keep` 时保留的 wheel 安装验证环境 |

`packages/f8studio_server/f8studio_server/web_dist` 不再接收构建输出。开发服务器默认读取 `build/web-studio`；安装后的服务器读取 wheel 内嵌的 `web_dist`。旧源码目录中残留的静态文件不会覆盖开发构建。

## 开发与调试

首次检出包含子模块的仓库后：

```sh
git submodule update --init --recursive
pixi install --locked -e default -e web-studio-test -e ci
pixi run -e web-studio npm --prefix packages/f8studio_web ci
pixi run install_services
```

Python 调试使用 `.pixi/envs/default` 中的解释器，以仓库根目录为工作目录，启动模块 `f8studio_server` 或相应服务模块。纯 Python 修改只需重启进程。服务描述发生变化时执行 `pixi run install_services --refresh`。

前后端分别启动：

```sh
pixi run studio_server
# 或使用 pixi run studio_launch 在服务器就绪后自动打开浏览器
# 另一个终端：Vite 热更新并代理 API 到本机 8210 端口
pixi run -e web-studio studio_web_dev
```

预览生产静态资源：

```sh
pixi run -e web-studio studio_web_build
pixi run studio_server
```

C++ 使用已有的增量构建和部署流程：

```sh
pixi run -e cpp cpp_bootstrap
pixi run -e cpp cpp_configure_release
pixi run -e cpp cpp_build_release
pixi run -e cpp cpp_test_release
```

bootstrap 主要在首次构建或 Conan 依赖变化后执行。日常修改执行 build 即可；CMake 的聚合部署目标把产物复制到 `runtime/bundles`。原生调试器对应 `build/Release` 的构建产物；当前公共 preset 是 Release，本次没有增加未经验证的跨平台 Debug 工具链。

## 发行流程

CI 的 `setup-pixi` 固定使用 **v0.81.0**，与当前 v7 锁文件及本地版本一致。升级 Pixi 时需一并迁移锁文件并验证各运行平台，不能让 CI 默默追随 latest；新版 Pixi 的 `lock --check` 可能因格式升级而失败，即使依赖安装成功。Linux 使用命名平台 `linux-glibc228` 显式保留 glibc 2.28 基线，替代已弃用的 `[system-requirements]` 配置。

手动运行 `dist-windows` 时，默认构建 GitHub 页面所选的分支或 tag；`git_ref` 留空即可，只有要覆盖检出目标时才填写。工作流会打印实际检出的 commit，非 tag 产物版本使用该 commit 的短 SHA。

Windows 缓存预热只在每周一 UTC 07:00 的定时任务（默认分支）或手动选择 `job_mode=warm-caches` 时运行；普通 `build` 中显示 skipped 是预期行为。预热与普通构建使用相同的依赖准备流程：

1. `setup-pixi` 只安装固定版本的 CLI，禁用它在 job 收尾阶段保存的隐式缓存。
2. 显式恢复并安装三个 CI 环境：`build-check`（Python/Web 构建、测试及普通/ONNX 描述）、`cpp`（原生工具链）、`mediapipe`（独立运行依赖）。安装成功后立即保存这三个环境目录；使用 `v4-merged` 缓存键，不恢复旧环境集合。开发及发行环境仍保留，但 CI 不再逐一安装。
3. 实际刷新全部 Python 服务描述；此时不允许隐式安装环境，绑定或导入错误会在原生编译之前暴露。
4. 恢复 Conan 缓存，执行 `cpp_bootstrap` 下载/编译第三方依赖，成功后立即保存 `.conan2`。不缓存项目 C++ 编译产物或发行包。

分支和 tag 构建都保存未命中的缓存，后续打包失败不影响已完成的保存步骤。GitHub 缓存按 workflow 运行的 ref 隔离，检出 `git_ref` 不会改变缓存作用域：分支可读取自身和默认分支的缓存，tag 缓存可供同 tag 重跑复用，但其他 tag 无法读取它。需要跨分支/tag 复用时，应在默认分支运行 `warm-caches`。Pixi 缓存键包含格式版本、OS、CLI 版本、锁文件哈希和工作区绝对路径；新增环境会通过锁文件哈希自动产生新键；安装策略或缓存布局变化时需更新键中的版本。Conan 按配方和锁文件哈希匹配，并允许回退到旧依赖缓存。

`install_services` 会在执行任何服务之前验证所有选中 Pixi 服务的显式环境及任务绑定，再集中执行 `pixi install --locked`。CI 使用 `--build-check` 将除 MediaPipe 外的 Python 服务描述统一映射到 `build-check` 并校验任务存在；普通开发调用将 ONNX 描述映射到 `onnx-describe`，服务启动声明仍指向 `onnx`。描述子进程使用 `--frozen --no-install`，依赖下载不再计入描述超时。CI 提供 `--no-install` 复用已准备环境，`--python-only` 在原生编译前检查 Python 服务；完整发行仍刷新并验证全部服务。失败信息包含服务类名、命令、工作目录及子进程 stdout/stderr，并保留异常链。描述全部验证通过后才写入文件。

当前发行不要求用户克隆仓库。开发者或 CI 执行：

```sh
pixi run --locked -e build-check dist_ci --archive
```

流程依次为：

1. 构建 C++ 并部署运行库；构建 Web；刷新服务声明。
2. 复制服务运行产物、配置、模型资源和 Windows Unity 资产到发行目录。
3. 从源码的临时副本构建非 editable wheels，把 Web 静态资源嵌入 server wheel。源码目录不接收构建产物。
4. 从开发清单提取带 `launcher-runtime` 标记的环境，将本地 editable 依赖改写成 `wheels/*.whl`。保留所属 feature，清除依赖源码脚本的开发任务。
5. 以根锁文件为种子生成发行锁文件，同时锁定第三方依赖和本地 wheels。
6. 复制轻量启动脚本、生成安装脚本、输出 zip（Windows）或 tar.gz（Linux）。无需 Nuitka/PyInstaller 编译，不再打包第二套 Python/Tk。

用户解压后，双击 `f8studio.cmd`（Windows）或执行 `./f8studio`（Linux）。基础运行时、Python、所有基础第三方包和本地 wheels 已包含在 `offline/base-runtime.tar` 中，官方 `pixi-unpack` 工具也随包提供。第一次启动只做本地解包并写入 `.runtime-location`，不下载 Pixi、不联网安装依赖；后续启动直接复用 `env`。也可提前运行 `install_env.bat` / `install_env.sh` 完成这一步。移动整个发行目录后会使用本地包重新准备环境，修复绝对前缀。

启动器激活包内运行时并直接执行 `python -I -m f8studio_server --open-browser`；基础 Python 服务的启动声明同样直接指向包内解释器。`config/service-index.json` 保留全部服务元数据，`config/extensions.json` 声明服务归属、运行环境和预装清单。默认 `standard` 预装普通 Python、音频和 C++ 扩展，DL 和 MediaPipe 按需安装；`pixi run -e ci dist_ci --preset core` 不预装服务扩展。这个 preset 控制初始安装状态，当前仍携带基础环境及可重装服务的文件，尚未把物理包体裁剪成最小 Web Studio。

Web Studio 的 Services 页面统一管理全部服务扩展，支持安装、取消、启停、卸载和从发布者的 HTTPS ZIP 链接与 SHA-256 导入新包。第三方 Python 扩展以 `shared` 引用官方 `pixi.toml` 的环境名，如 `studio-runtime`、`onnx`、`mediapipe`。开发时复用对应 workspace 环境；发行时基础环境使用包内 `env`，可选环境复用官方扩展已准备的托管目录。安装要求从扩展 wheel 的元数据读取，无需重复写依赖清单；共享模式只检查要求并保存自身代码，不下载环境或修改官方包。依赖不满足时发布者可以提供独立 Pixi 环境。独立模式发现 Pixi 或用官方脚本安装固定的 0.81.0；锁定环境及 wheels 存在用户目录 `runtimes/<environment>-<digest>`，相同配置复用同一环境。安装及服务描述检查通过后才激活；失败保留缓存和已准备的环境，便于重试。共享扩展与官方扩展按同一环境计数，卸载最后一个使用者后才回收托管环境；源码环境和包内基础环境保留。模型元数据与按需权重位于用户目录 `models`，不随服务卸载删除。运行中或正在启动的服务会阻止停用/卸载。完整格式见 [extensions.md](extensions.md)。

Windows CI 产物为一个离线 ZIP，不再同时上传展开目录；ZIP 上传不再次压缩，内部已压缩的运行时包/wheels 也不重复压缩。构建前已检查的 Python 描述通过 `--reuse-python-describes` 复用，原生描述仍在编译后刷新。验证直接解压最终 ZIP，在独立目录内运行两次启动器（第二次必须不重复解包），然后用包内 Python 检查服务描述、Web/health、包安装位置和编辑器工具；不再逐环境联网安装依赖。保留原生契约测试作为语义验证。

离线打包工具固定为 `pixi-pack` / `pixi-unpack` 0.7.11，构建下载时校验官方发布 SHA-256。工具和包下载缓存位于 `build/offline-cache`。开发依赖与 GPU 推理验证不等于基础包离线验证。

## 验证与 CI

推送 Web 改动前运行 `pixi run --locked -e web-studio-test studio_web_ci`。该命令与 GitHub 的 Web job 共用，依次执行 `npm ci`、TypeScript 检查和全部 Vitest 测试；任一步失败都会停止。

```sh
pixi run pytest tests/test_dist_ci.py tests/test_launcher_scripts.py tests/test_release_wheels.py -q
pixi run --locked -e web-studio-test studio_release_smoke --verify-dist-lock --keep
pixi run --locked -e ci python scripts/verify_dist.py build/dist/f8studio-windows-x86_64.zip
```

最后一条应在与发行包对应的平台执行，也支持 Linux tar.gz。

- Python 单元测试不依赖本机的 `runtime/bundles` 或原生编译产物；需要引擎目录时，从真实 PyEngine 注册代码生成临时描述与启动声明。验证 CI 时应使用未安装服务的干净检出目录，避免本机缓存掩盖缺失依赖。
- quality CI：Python、Web、协议契约检查；额外构建非 editable wheels，验证内嵌页面、HTTP health/root 和全部运行环境的发行锁文件。
- wheel smoke 的临时 venv 复用测试环境的第三方依赖，但断言项目模块来自已安装 wheel。它不是完全隔离的依赖安装测试。
- Windows dist CI：构建和原生契约检查后，将离线压缩包解压到仓库外，验证自带运行时、基础服务及连续两次启动，然后才允许上传。
- tag `v*` 或手动工作流触发 Windows 发行；发布开关和 release tag 仍由工作流控制。Linux 有打包脚本支持，但没有同等的自动发布工作流。

### Quality 的延迟触发

使用 GitHub 原生 Environment 等待规则和 workflow concurrency，无需额外 Action 或定时扫描脚本：

1. 在仓库 **Settings → Environments** 中创建 `quality-debounce`。
2. 将 **Wait timer** 设置为 **720 分钟**并保存。不要添加 required reviewers；允许需要检查的分支使用此环境。
3. push 触发的 `debounce` job 先等待该规则放行，再运行 Python、Web 和 release-wheels 检查。

同一分支的新 push 通过 `cancel-in-progress: true` 取消旧运行，新运行重新等待 12 小时。等待发生在 runner 分配之前，不消耗计费运行时间；等待结束后的实际启动仍受 GitHub 排队影响。PR 和手动运行跳过等待，不会被 push 的并发组取消。环境会产生 GitHub deployment 记录，但这个 job 不部署应用，只作为检查前的等待入口。

**720 分钟是仓库 Environment 设置，不能仅靠 YAML 设置。必须先配置上述环境，否则自动创建的同名环境没有等待规则，检查将立即运行。** 原生方案无需等待工作流合入默认分支才能启动计时。公开仓库可使用 wait timer；私有仓库须确认 GitHub 套餐是否支持。

参考：[原生 concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)、[Environment wait timer](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments#wait-timer)。

## 桌面托盘与退出

离线启动器默认使用 `--tray`。托盘菜单提供 Open Studio、Open console / logs、Exit；关闭浏览器不会停止服务器。日志追加到用户数据目录的 `studio-console.log`，重复启动不会覆盖之前的退出诊断。Windows 日志窗口使用 PowerShell，Linux 使用系统终端实时追踪日志，没有终端时尝试默认文件查看器。`./f8studio --no-tray` 保留前台运行方式。
托盘使用 `assets/icon.png`，打包时纳入 f8studio-server wheel，因此离线安装后也能显示相同图标。

托盘通过关闭父进程管道请求 Studio 退出，Studio 关闭 Media Gateway 的父管道后等待其正常结束；超时才强制终止并记录日志。Media Gateway 使用原始文件描述符读取管道，避免 Python 退出时 BufferedReader 锁导致 fatal error。受管理的 Gateway 不直接接收终端 Ctrl+C，避免重复中断。

事件和实时数据 WebSocket 会监听客户端断开并取消发送任务，避免空闲网页连接阻塞退出。HTTP 服务的连接清理最多等待 5 秒。Linux 托盘在 GTK 初始化后恢复 Ctrl+C 处理，退出菜单和终端中断均由同一个清理路径等待服务器结束。`server.lock` 使用操作系统锁；文件留在磁盘上是正常的，不应通过删除文件解除运行中的实例锁。

Linux 托盘使用 GTK StatusIcon，需要桌面提供托盘支持（已在 Cinnamon 实测）；GNOME/Wayland 等没有传统托盘区域的桌面可能需要托盘扩展，不能保证显示。可用 `--no-tray` 运行；后端初始化失败会警告并回退终端模式。Windows 原生托盘仍需 Windows 实机验证。没有使用 PyInstaller 或 Qt。
