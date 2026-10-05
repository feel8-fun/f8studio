# 显式服务注册与安装布局

仓库根目录的 `services/` 已移除。服务注册、安装产物、共享模型和用户配置分别管理。

## 当前布局

| 位置 | 内容与生命周期 |
| --- | --- |
| `config/service-index.json` | 显式注册索引：平台 manifest、describe 路径和模型根目录 |
| `config/services/<serviceClass>/service*.yml` | 纳入版本控制的启动声明 |
| `runtime/bundles/<bundle>/0.0.1/` | 构建/安装生成的描述、平台可执行文件、动态库和字体；不纳入版本控制 |
| 包内 `resources/models/onnx` | 只读模型 YAML 元数据；安装时复制到可写模型目录 |
| `${F8_MODEL_ROOT}/onnx`、`mediapipe`、`tracking` | 用户模型元数据和下载权重，与安装目录分离 |
| 平台用户配置目录 `f8studio/services/` | 用户配置；可通过绝对路径 `F8_CONFIG_ROOT` 覆盖 |

CVKit 多个服务共用 `f8.cvkit` 可执行包；每个服务有自己的描述。版本升级需同步索引、manifest 和构建输出路径。

## 路径根标识

| 标识 | 解析来源 |
| --- | --- |
| `${F8_PACKAGE_ROOT}` | 索引所在包的安装根目录；标准索引位于 `<package>/config/service-index.json` |
| `${F8_BUNDLE_ROOT}` | 索引中当前服务、平台的 `bundleRoots` 声明；不按服务类推断 |
| `${F8_CONFIG_ROOT}` | 用户配置目录，使用绝对环境变量 `F8_CONFIG_ROOT` 或平台默认值 |
| `${F8_RESOURCE_ROOT}` | 用户资源目录，使用绝对环境变量 `F8_RESOURCE_ROOT` 或平台默认值 |
| `${F8_MODEL_ROOT}` | 使用绝对环境变量 `F8_MODEL_ROOT`；否则为用户资源目录下的 `models/` |

`F8_PACKAGE_ROOT` 和 `F8_BUNDLE_ROOT` 由加载器提供，不从任意环境变量取值。用户可写目录不能通过设置这两个名称来重定向。Studio 扩展管理器在未指定资源/模型环境变量时，继续将需安装的模型元数据放在 Studio 数据目录的 `models/`；显式覆盖会同时用于元数据复制和服务启动。

索引示例：

```json
{
  "schemaVersion": "f8serviceIndex/1",
  "modelRoot": "${F8_MODEL_ROOT}",
  "services": [{
    "serviceClass": "f8.audiocap",
    "manifests": {"win32": "${F8_PACKAGE_ROOT}/config/services/f8.audiocap/service.win.yml"},
    "describe": "${F8_PACKAGE_ROOT}/config/describes/f8.audiocap.json",
    "bundleRoots": {"win32": "${F8_PACKAGE_ROOT}/runtime/bundles/f8.audiocap/0.0.1/win"}
  }]
}
```

对应启动配置：

```yaml
launch:
  command: ${F8_BUNDLE_ROOT}/f8audiocap_service.exe
  args: []
  env: {}
  workdir: ${F8_BUNDLE_ROOT}
```

引用必须是预定义根标识加 `/` 分隔的后缀。未知名称、格式错误、缺少当前平台 bundle 根、路径或符号链接越界均报错。不会进行 shell 展开、任意环境变量插值或递归查找。启动命令、工作目录、参数和环境字段中的引用在加载时解析为绝对路径。

包内 manifest、describe、可执行产物与模型元数据属于包；用户配置和下载权重属于可写目录。源文件里的路径引用保留可搬迁性；安装器生成的本地注册文件可以保存绝对路径和已解析根环境。

## 加载与安装

Studio 默认读取索引，不递归扫描服务目录、不计算源码指纹、不运行 `--describe`。当前配置使用明确的路径根标识，由 SDK 的 `ServicePaths` 统一解析；不依赖 manifest 的嵌套层数。旧相对路径仍可读取，供既有安装及迁移使用。Python 服务显式使用安装根目录，C++ 服务使用对应版本的平台运行目录，因此安装目录可整体移动。

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

开发 workspace 的 CMake 产物部署到 `build/workspace/runtime/bundles/`。各扩展维护自己的 `config/services`、`config/service-index.json` 和资源声明；`pixi run workspace_catalog` 只在 `build/workspace` 生成汇总索引、启动适配和模型元数据副本。下载权重位于 `${F8_MODEL_ROOT}`，不参与 workspace 清理。正式发行由独立发行仓库组装已发布扩展产物。文档与节点图鉴默认读取生成索引。

SDK 仍保留显式 `roots` 目录工具用于外部迁移和测试；它不是默认启动路径。旧 `F8_SERVICE_DISCOVERY_DIRS` 不控制正式加载，请使用 `F8_SERVICE_INDEX`。

## 迁移其他旧工作区

```sh
pixi run install_services --migrate-layout /path/to/old/services --migrate-resources /path/to/old/services
pixi run update_describes
```

工具校验 SHA256 并复制已知运行包、模型及配置，保留源文件和可执行权限；目标内容不同则报冲突，相同内容允许重复运行。也可重新构建 C++ 产物，无需复用旧二进制。

| 旧资源目录 | 新共享目录 |
| --- | --- |
| `services/f8/dl/weights` | `${F8_MODEL_ROOT}/onnx` |
| `services/f8/mp/pose/models` | `${F8_MODEL_ROOT}/mediapipe` |
| `services/f8/cvkit/tracking/models` | `${F8_MODEL_ROOT}/tracking` |

索引为服务注入绝对 `F8_MODEL_ROOT`。空 `weightsDir` / tracking `modelDir` 使用安装默认目录；用户输入的其他相对路径仍相对于服务工作目录，不再回退搜索源码仓库。独立运行服务时，未设置环境变量会使用平台用户数据目录下的 `f8studio/models`。模型不是可随意删除的临时缓存。

## 开发 workspace 与扩展的边界

根目录只保留 `config/extension-workspace.toml`，声明参与集成的源码包、默认启用策略和 bootstrap runtime。扩展的服务定义、模型描述和默认资源由各自仓库维护。

```sh
pixi run --locked -e build-check workspace_prepare
pixi run --locked -e build-check workspace_catalog
```

生成的 `build/workspace/config/service-index.json` 用 `packageRoot` 声明源码根的绝对路径，使深层生成目录仍能解析扩展配置和构建产物。这个字段用于本地登记；普通扩展索引省略它，继续相对于安装包解析。导入扩展时仍检查所有元数据路径是否位于包内。

`extension-sources.json` 按 extension ID 关联源码 workspace。各扩展的 Pixi 环境名称和数量独立，相同名称不会合并解释器或造成冲突。开发集成需要的 Pixi 启动适配由生成器从扩展的 module 声明和任务生成。

模型 YAML 的原始定义保留在扩展资源目录。安装时向 `${F8_MODEL_ROOT}` 补充缺失定义，保留用户已有配置。模型下载地址、后端和下载行为由对应扩展管理；平台负责通用安装、路径解析与生命周期。

`workspace_clean` 可删除整个 `build/workspace`。后续 workspace 任务重新生成目录；模型和用户配置仍留在独立可写目录。
