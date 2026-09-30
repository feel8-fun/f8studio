# 开发、构建与发行

开发环境保留 Python editable 安装；发行环境安装固定的 wheel。修改 Python 实现后无需重新安装，重启对应服务即可生效；只有修改依赖或包元数据时才需要重新同步环境。不要用发行环境调试源码，也不要把 `.pixi` 打包给用户。

## 目录职责

| 目录 | 内容 |
| --- | --- |
| `packages/`、`external/` | 源码、包定义、受版本控制的资源 |
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
2. 显式恢复 `.pixi`，使用 `pixi install --locked --all` 安装当前平台的全部环境，成功后立即保存 Pixi 缓存。新增环境只需更新 `pixi.toml` 和锁文件，无需维护 CI 环境列表；文档和测试环境也会安装，因此首次安装与缓存体积会相应增加。
3. 实际刷新全部 Python 服务描述；此时不允许隐式安装环境，绑定或导入错误会在原生编译之前暴露。
4. 恢复 Conan 缓存，执行 `cpp_bootstrap` 下载/编译第三方依赖，成功后立即保存 `.conan2`。不缓存项目 C++ 编译产物或发行包。

分支和 tag 构建都保存未命中的缓存，后续打包失败不影响已完成的保存步骤。GitHub 缓存按 workflow 运行的 ref 隔离，检出 `git_ref` 不会改变缓存作用域：分支可读取自身和默认分支的缓存，tag 缓存可供同 tag 重跑复用，但其他 tag 无法读取它。需要跨分支/tag 复用时，应在默认分支运行 `warm-caches`。Pixi 缓存键包含格式版本、OS、CLI 版本、锁文件哈希和工作区绝对路径；新增环境会通过锁文件哈希自动产生新键；安装策略或缓存布局变化时需更新键中的版本。Conan 按配方和锁文件哈希匹配，并允许回退到旧依赖缓存。

`install_services` 会在执行任何服务之前验证所有选中 Pixi 服务的显式环境及任务绑定，再集中执行 `pixi install --locked`。描述子进程使用 `--frozen --no-install`，依赖下载不再计入描述超时。CI 提供 `--no-install` 复用已准备环境，`--python-only` 在原生编译前检查 Python 服务；完整发行仍刷新并验证全部服务。失败信息包含服务类名、命令、工作目录及子进程 stdout/stderr，并保留异常链。描述全部验证通过后才写入文件。

当前发行不要求用户克隆仓库。开发者或 CI 执行：

```sh
pixi run --locked -e ci dist_ci --archive
```

流程依次为：

1. 构建 C++ 并部署运行库；构建 Web；刷新服务声明。
2. 复制服务运行产物、配置、模型资源和 Windows Unity 资产到发行目录。
3. 从源码的临时副本构建非 editable wheels，把 Web 静态资源嵌入 server wheel。源码目录不接收构建产物。
4. 从开发清单提取带 `launcher-runtime` 标记的环境，将本地 editable 依赖改写成 `wheels/*.whl`。保留所属 feature，清除依赖源码脚本的开发任务。
5. 以根锁文件为种子生成发行锁文件，同时锁定第三方依赖和本地 wheels。
6. 复制轻量启动脚本、生成安装脚本、输出 zip（Windows）或 tar.gz（Linux）。无需 Nuitka/PyInstaller 编译，不再打包第二套 Python/Tk。

用户解压后，双击 `f8studio.cmd`（Windows）或执行 `./f8studio`（Linux）。启动脚本首先检测 Pixi；缺失时使用官方安装脚本（Linux：`https://pixi.sh/install.sh`，Windows：`https://pixi.sh/install.ps1`），安装后立即继续，无需重开终端。Linux 需要 curl 或 wget，Windows 使用 PowerShell。启动脚本随后调用 `install_env.bat` / `./install_env.sh` 执行 `pixi install --locked`，再通过 `pixi run --locked -e studio-runtime studio_launch` 启动服务器。服务器就绪后打开浏览器。保留终端以查看日志，Ctrl+C 停止；`--no-browser` 禁用自动打开浏览器。不再额外运行 pip 安装本地包，因此首次由启动器创建环境时也能安装全部应用代码。

发行包仍要求 Pixi 和首次安装时的依赖下载，不是完全离线包。每个版本应解压到独立目录；不要覆盖一个仍在运行或已有旧 `.pixi` 环境的版本目录。构建与上传是不同步骤，本地打包不会自动发布。

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
- Windows dist CI：构建和原生契约检查后，把压缩包解压到仓库外的临时目录；使用自己的锁文件安装各运行环境，验证本地包安装位置、内嵌页面及实际启动脚本，然后才允许上传。
- tag `v*` 或手动工作流触发 Windows 发行；发布开关和 release tag 仍由工作流控制。Linux 有打包脚本支持，但没有同等的自动发布工作流。

### Quality 的延迟触发

使用 GitHub 原生 Environment 等待规则和 workflow concurrency，无需额外 Action 或定时扫描脚本：

1. 在仓库 **Settings → Environments** 中创建 `quality-debounce`。
2. 将 **Wait timer** 设置为 **720 分钟**并保存。不要添加 required reviewers；允许需要检查的分支使用此环境。
3. push 触发的 `debounce` job 先等待该规则放行，再运行 Python、Web 和 release-wheels 检查。

同一分支的新 push 通过 `cancel-in-progress: true` 取消旧运行，新运行重新等待 12 小时。等待发生在 runner 分配之前，不消耗计费运行时间；等待结束后的实际启动仍受 GitHub 排队影响。PR 和手动运行跳过等待，不会被 push 的并发组取消。环境会产生 GitHub deployment 记录，但这个 job 不部署应用，只作为检查前的等待入口。

**720 分钟是仓库 Environment 设置，不能仅靠 YAML 设置。必须先配置上述环境，否则自动创建的同名环境没有等待规则，检查将立即运行。** 原生方案无需等待工作流合入默认分支才能启动计时。公开仓库可使用 wait timer；私有仓库须确认 GitHub 套餐是否支持。

参考：[原生 concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)、[Environment wait timer](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments#wait-timer)。
