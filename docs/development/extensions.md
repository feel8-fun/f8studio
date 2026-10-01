# 服务扩展与运行环境

Web Studio 管理服务包的安装状态；一个包可以提供多个服务。核心 Studio 服务始终由服务端提供，其余服务归属于显式扩展。停用移出服务目录，卸载删除安装登记，用户项目与模型数据保持原样。发行版通过 `config/extensions.json` 的 `preinstalled` 选择初始扩展集合。

## 清单与制品

`config/service-index.json` 保存服务的启动声明和预先构建的描述，`config/extensions.json` 保存服务归属、扩展版本、环境要求和模型元数据目录。每个服务只能属于一个扩展，索引中的服务必须全部有归属。清单类型定义在 `f8studio_server/extension_models.py`；独立包可使用生成的 `schemas/extensions.gen.json` 验证格式。

一个可导入的 ZIP 在根目录包含下列内容，不额外包一层目录：

```text
config/extensions.json
config/service-index.json
config/services/<service>/service.<platform>.yml
runtime/bundles/<package>/<version>/...
pixi.toml                 # 仅 Pixi 扩展需要
pixi.lock                 # 已求解、不可在客户端更新
wheels/*.whl              # 本地 Python 服务的非 editable wheels
python/<package>/...      # 共享官方运行时的扩展代码，不包含基础依赖
resources/models/<name>/*.yaml
```

发布者提供 HTTPS 地址和 ZIP 的 SHA-256。导入校验下载哈希、归档路径与服务归属，并保存发布源；不会执行服务，也不会采纳外部包的预装标记。用户点击安装后才准备环境并执行各服务的 `--describe`，校验协议、监控契约与服务类身份。当前导入来源由用户指定并信任，哈希用于完整性校验；没有公共市场、发布者签名或代码沙箱。版本冲突会明确拒绝，不覆盖现有扩展。

原生扩展声明 `runtime.kind = "native"`，随包携带目标平台可执行文件及动态库。Python 扩展声明 `runtime.kind = "pixi"` 和明确的 `environment`，提供自己的发布 manifest、lock 和 wheels。服务启动声明必须明确使用该环境：

```yaml
launch:
  command: pixi
  args: [run, -e, inference, detector]
  workdir: ../../..
```

安装器将其绑定到受管理 workspace，启动时使用 `--frozen --no-install --manifest-path ...`，避免继承 Studio 的活动环境或启动时临时安装。安装前检查环境已经声明，安装后的描述检查将 stderr 日志与 stdout JSON 分开。服务描述不需要下载模型权重。

本仓库开发清单使用 `workspace` 复用根目录下的开发环境，保留 editable 调试方式；发行包生成时把基础 Python 服务改为 `bundled`，把可选服务改为锁定的 `pixi`。外部制品接受 `native`、`shared` 或 `pixi`；不会接受任意 `workspace` 路径或改动官方环境。

## 第三方共享官方运行时

Python 扩展使用两种发布方式：已经满足依赖的扩展引用官方运行时；需要其他版本或新增依赖的扩展使用独立、锁定的 Pixi 环境。安装器不求解不同扩展之间的兼容性，不合并或拆分环境，也不向官方环境执行 pip 安装。轻量 venv 暂不支持。

共享扩展直接引用官方 `pixi.toml` 中的环境名，例如 `studio-runtime`、`onnx`、`mediapipe`，无需单独的运行时编号或清单。开发时复用对应的 `.pixi/envs/<name>`；发行包中 `studio-runtime` 使用包内 `env`，可选环境复用官方扩展已准备好的托管目录。`GET /api/environments/presets` 返回当前发行版提供的环境名及就绪状态。尚未准备的环境会明确提示先安装对应官方扩展或准备开发环境，不会借用第三方包自己的同名 workspace，也不会现场求解或修改官方依赖。环境名称表示预设用途；兼容性以实际 Python、依赖版本及服务描述校验为准，发布者仍需在目标平台测试原生库和 GPU 要求。

扩展的 `runtime` 可以声明：

```json
{
  "kind": "shared",
  "environment": "onnx"
}
```

ZIP 内提供独立的 `python/` 目录和服务清单，无需提供 `pixi.toml`、`pixi.lock` 或完整环境。推荐在第三方仓库的构建环境中将自己的 wheel 用 `pixi run python -m pip install --no-deps --target <staging>/python <wheel>` 准备为已安装布局，保留 `.dist-info` 的依赖元数据。服务入口声明为模块：

```yaml
launch:
  command: python
  args: [-m, my_tracker.main]
  workdir: ../../..
```

扩展自身的 Python 和依赖要求写在 `pyproject.toml`，构建后从 `.dist-info/METADATA` 的 `Requires-Python` 和 `Requires-Dist` 自动读取，扩展 JSON 无需重复。没有包元数据的脚本扩展可以选填 `requiresPython` 和 `dependencies` 作为检查约束，它们不触发安装。安装前通过官方解释器的独立进程读取实际版本和环境标记，验证这些要求；支持 PEP 508 环境标记、extras 和传递依赖。URL 依赖、缺失或不兼容的包会明确拒绝，并提示发布者使用独立 Pixi 环境。扩展不能携带替换官方包的同名发行物或遮蔽官方/标准库的顶层模块。

验证成功后只将扩展代码复制到用户目录 `extensions/registrations/<id>/python`。启动使用官方 Python 的隔离模式，在子进程内加入该扩展目录并运行声明的模块，不向 Studio 服务端导入扩展代码，也不继承外部 `PYTHONPATH`。服务仍需通过 `--describe` 的协议契约校验才会激活。共享扩展和官方扩展使用相同的环境登记和引用计数；先卸载官方扩展也不会删除仍被第三方使用的托管环境，最后一个使用者卸载后才回收。源码环境与包内基础环境始终保留。项目、模型及下载缓存保留；官方依赖锁发生变化时，需要重新安装验证扩展。

## 环境生命周期

环境与扩展是多对一关系。受管理环境的身份由平台、架构、环境名称、完整 manifest、完整 lock 和 wheel 内容生成。完全相同的配置共享运行环境；配置或锁改变会得到新的目录。安装不升级现有环境的依赖，运行时也不求解依赖。

多个扩展可以声明同一发布环境。启用和停用不改变环境引用；只有卸载最后一个使用者才回收托管环境。安装失败或取消时保留 Pixi 下载缓存、部分安装目录和已准备的环境，重试可复用。模型、用户项目、发布包缓存与只读基础环境独立保留。

独立 Pixi 环境只做精确配置复用；共享官方运行时只做现有依赖检查。依赖不满足时明确失败，由发布者提供独立环境制品，不由 Studio 猜测或升级依赖。不同 manifest 的兼容性求解、自动合并/拆分和全局最省空间分组不在当前计划中。

## API 与持久化

| API | 行为 |
| --- | --- |
| `GET /api/extensions` | 可用扩展及安装状态 |
| `POST /api/extensions/import` | `{ "url": "https://...", "sha256": "..." }` 导入 ZIP |
| `GET /api/extensions/{id}/plan` | 查看复用、创建、基础或开发环境计划 |
| `POST /api/extensions/{id}/install` | 启动后台安装与契约检查 |
| `POST /api/extensions/{id}/cancel` | 终止安装进程树并等待清理 |
| `PUT /api/extensions/{id}/enabled` | `{ "enabled": false }` 停用或重新启用 |
| `DELETE /api/extensions/{id}` | 卸载并回收不再使用的托管环境 |
| `GET /api/environments` | 环境就绪状态和扩展使用者 |
| `GET /api/environments/presets` | 官方预设环境名和就绪状态 |

安装登记写入用户目录 `extensions/state.json`，发布源写入 `extensions/sources.json`；使用临时文件替换持久化状态。扩展配置和描述检查成功后才更新服务目录。失败有日志与 traceback，单个扩展的损坏元数据在恢复时被停用，避免阻断 Studio 启动。完整安装输出位于 `extensions/logs/<id>.log`。

## 独立仓库与 superbuild

各服务仓库负责单元测试、协议描述、平台二进制/wheel、自己的依赖锁和扩展 ZIP。Python 服务通过进程与协议协作，不导入 Studio 服务端内部模块。发布依赖必须使用非 editable wheel，依赖锁应在各仓库 CI 中检查，发布前执行安装和真实入口描述验证。

各功能包的源码边界、独立 CMake/SDK 接口、制品工具和本地仓库导出见[扩展源码仓库与 superbuild](extension-repositories.md)。主仓库 pytest 默认运行核心及集成测试，各包自己的测试和 Windows/Linux CI 随独立仓库导出。源码尚未替换成新的远程 submodule；远程仓库需要先有可拉取的固定提交。

主仓库后续应锁定各扩展的版本、制品 SHA-256、平台和发行预装清单，再做安装、协议、进程启停与关键跨扩展数据流测试。远程 superbuild 制品锁、签名目录、扩展升级回滚、可选扩展的预打包完整环境和最小物理发行包仍是后续工作。
