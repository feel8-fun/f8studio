# 开发工作区与发行仓库

当前 `f8studio` 仓库是开发工作区：固定 SDK、launcher 和 extension 的源码版本，
提供联合调试、协议生成及跨仓库集成检查。各个独立仓库负责自己的构建、测试和发布。
工作区的本地 snapshot 用于开发验证，不作为正式发行制品。

## 职责划分

| 仓库 | 职责 | 发布内容 |
| --- | --- | --- |
| f8studio | 开发工作区与集成检查 | 源码版本组合与开发工具 |
| f8sdk | 公共协议、调用约定、SDK | Python/C++ SDK |
| f8platform（launcher/） | 安装、运行环境、进程生命周期、版本选择与回滚 | bootstrap runtime |
| f8webstudio | Web UI、后端、私有图文档/API | 前后端一起发布的 extension |
| f8mediagateway | 媒体传输应用 | extension 与私有锁定环境 |
| f8distribution | 官方发行组合、校验与离线打包 | 已发布产物的组合 |

应用使用普通 extension 元数据中的 `application` 能力。
service、tool、skill、resource 和 application 共用 extension 发布格式。
launcher 是启动这些扩展的基础设施，不作为 extension 安装。

## 开发验证

```bash
pixi run -e build-check python scripts/workspace_inputs.py prepare
pixi run -e build-check pytest -q
pixi run -e build-check studio_web_ci
pixi run -e build-check typecheck
pixi run -e build-check workspace_snapshot --output build/workspace-snapshot
```

SDK 是公共协议的唯一来源；应用自己的私有接口在应用仓库维护。
WebStudio 前端版本、后端版本和 extension 版本必须一致，不能分别升级。

## 官方发行

在独立 `f8distribution` 仓库维护 `f8platformRelease/1` release lock，
声明 platform、bootstrap、startup，以及每个制品的身份、版本、位置和 SHA-256。

```bash
pixi run --locked assemble releases/<release>.json
pixi run --locked verify build/dist/f8-linux-x86_64.tar.gz
```

组装器验证制品身份、平台、路径安全、哈希、应用协议依赖和启动集合，
仅搬运锁定输入，并打包 launcher 的离线解释器。
不编译扩展、不构建前端、不重新求解应用依赖。
应用运行环境在安装时按照各自锁文件准备，Pixi cache 在平台数据目录复用。
离线 bootstrap 不表示所有应用依赖都已离线打包。

当前独立仓库的工作区 checkout 和发行验证可以本地完成；正式远程发布需要
对应仓库及其 publisher 上传真实制品，随后把发布地址和哈希写入 release lock。
