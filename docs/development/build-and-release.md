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
