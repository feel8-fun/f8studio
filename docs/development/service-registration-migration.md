# 显式服务注册与安装布局

仓库根目录的 `services/` 已移除。服务注册、安装产物、共享模型和用户配置分别管理。

## 当前布局

| 位置 | 内容与生命周期 |
| --- | --- |
| `config/service-index.json` | 显式注册索引：平台 manifest、describe 路径和模型根目录 |
| `config/services/<serviceClass>/service*.yml` | 纳入版本控制的启动声明 |
| `runtime/bundles/<bundle>/0.0.1/` | 构建/安装生成的描述、平台可执行文件、动态库和字体；不纳入版本控制 |
| `resources/models/onnx` | 纳入版本控制的模型 YAML，以及本地安装的 ONNX 权重 |
| `resources/models/mediapipe`、`tracking` | MediaPipe 与 tracking 模型 |
| 平台用户配置目录 `f8studio/services/` | 用户配置；可通过绝对路径 `F8_CONFIG_ROOT` 覆盖 |

CVKit 多个服务共用 `f8.cvkit` 可执行包；每个服务有自己的描述。版本升级需同步索引、manifest 和构建输出路径。

## 加载与安装

Studio 默认读取索引，不递归扫描服务目录、不计算源码指纹、不运行 `--describe`。相对索引路径以索引文件所在目录为基准；manifest 的工作目录以 manifest 目录为基准。Python 服务显式使用安装根目录，C++ 服务使用对应版本的平台运行目录，因此安装目录可整体移动。

Catalog 保存已解析的启动配置副本；ProcessManager 不在启动进程时重新读取 YAML。修改声明后刷新 Catalog。普通刷新只重读登记；明确请求的开发动态刷新仅针对索引中的服务。

```sh
# 先通过现有 C++ 构建流程构建并部署运行产物，再准备静态描述。
pixi run install_services
# 修改服务定义后显式刷新所有描述或一个服务。
pixi run update_describes
pixi run update_describes --service-class f8.pyengine
```

`F8_SERVICE_INDEX` 可指定其他安装的索引。重复服务类、服务类不匹配、非法或缺失描述会报错，不自动回退到动态发现；不支持的平台跳过，不尝试其他平台启动文件。
需要临时禁用服务时，设置 `F8_DISABLED_SERVICE_CLASSES`（多个服务类用逗号分隔）；旧的 `config/service_discovery_policy.yml` 已移除。

CMake 已部署到 `runtime/bundles/`。发行打包复制声明、运行包及共享资源，不携带迁移备份或历史用户配置。文档与节点图鉴也默认读取索引。旧 `scripts/update_static_describes.py` 已删除。

SDK 仍保留显式 `roots` 目录工具用于外部迁移和测试；它不是默认启动路径。旧 `F8_SERVICE_DISCOVERY_DIRS` 不控制正式加载，请使用 `F8_SERVICE_INDEX`。

## 迁移其他旧工作区

```sh
pixi run install_services --migrate-layout /path/to/old/services --migrate-resources /path/to/old/services
pixi run update_describes
```

工具校验 SHA256 并复制已知运行包、模型及配置，保留源文件和可执行权限；目标内容不同则报冲突，相同内容允许重复运行。也可重新构建 C++ 产物，无需复用旧二进制。

| 旧资源目录 | 新共享目录 |
| --- | --- |
| `services/f8/dl/weights` | `resources/models/onnx` |
| `services/f8/mp/pose/models` | `resources/models/mediapipe` |
| `services/f8/cvkit/tracking/models` | `resources/models/tracking` |

索引为服务注入绝对 `F8_MODEL_ROOT`。空 `weightsDir` / tracking `modelDir` 使用安装默认目录；用户输入的其他相对路径仍相对于服务工作目录，不再回退搜索源码仓库。独立运行服务时，未设置环境变量会使用平台用户数据目录下的 `f8studio/models`。模型不是可随意删除的临时缓存。

旧导出工程可转换已知模型目录默认值：

```sh
pixi run python scripts/migrate_service_paths.py old.f8studio.json migrated.f8studio.json
```

工具只接受 `f8studio-document/2`，写入新文件并保留输入；不修改任意用户状态、自定义绝对路径、SQLite 数据库或 f8graph 交换文件。复杂自定义路径需通过 Inspector 显式更新。

## 本机迁移与保留内容

旧目录完整保存在 `runtime/migration-backup/legacy-service-tree/`，不参与发现、运行或发行打包。未注册的历史 offline player 也仅保存在迁移备份中。

旧 `implayer/imgui.ini` 已复制至用户配置目录下的 `f8.implayer/imgui.ini`。当前仓库没有消费该配置的 ImGui 代码；保留此文件用于用户恢复，不宣称当前服务会读取它。

模型定义 YAML 已迁移到新目录；31 个资源文件与 163 个运行产物/配置文件已进行复制校验。原文件保存在备份中，未自动清除用户数据。

## 验证与边界

- 旧根目录不存在时，22 个服务描述重新生成成功；C++ 全部运行产物构建、部署成功。
- 真实 Studio → PyEngine 启动与部署通过；C++ 视频服务通过 1920×1080 WebRTC 连续变化帧解码验证。
- Python lint、SDK/Studio 类型检查和 CTest 通过；全量 Python 回归 **1066 passed、7 skipped、1 warning**。
- 节点图鉴通过索引生成成功。完整服务文档生成仍受原有缺失手册 `docs/modules/manual/operators/f8-cppengine/f8-data-mux.md` 阻断。
- 本次未构建完整跨平台发行包；发行目录复制与环境改写有定向测试。
- 稳定模型 ID/内容摘要绑定、下载版本锁定和数据库工程批量迁移仍是后续工作，不影响根目录 `services/` 的移除。
