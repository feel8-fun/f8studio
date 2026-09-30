# Web Studio 后续工作

2026-09-30 核对：Qt → Web 的核心迁移和 P7 后的图编辑精修已经落地，旧 Qt Studio、临时 Media Lab 和旧扩展加载器已移除。核心包含图编辑/部署、独立媒体网关、正式 Outputs、本地资产、Monaco/LSP、Agent、CLI/MCP、Web 启动器与本地前端 bundle。原实施前方案和完成流水账已删除；本文只保留未完成项、验收条件和仍有用的基准。

现行代码中的扩展仍为显式本地注册，媒体 peer 池仍按页面复用。以下性能数据和 Windows 结果来自先前保存的验证记录，本次文档整理没有重新执行长时基准或 Windows 实机验收。

## 原迁移的剩余验收与性能工作

| 工作 | 当前边界 | 完成条件 |
| --- | --- | --- |
| 组合媒体性能 | 每个 WebRTC sender 仍独立编码；历史 main 为 26.86–27.52 FPS，低于初始 28 FPS 预算，瓶颈已定位 | 分段测量解码/转换/编码/传输/显示，评估共享或硬件编码；重跑 1080p main + 4 thumbnails + 3D 的 5 分钟组合与 30 分钟稳定性；主预览 ≥28 FPS、控制响应 p95 ≤200 ms、队列有界、资源清理回归基线。调整预算须明确记录原因 |
| Windows 干净构建与安装 | 已有构建树上的 zip、启动器 dry-run、非 editable wheel smoke 和 Chrome E2E 已有记录 | 空构建树 bootstrap/build，在全新目录解压并运行 `install_env.bat`，验证启动器、内嵌静态资源、真实服务与关闭回收 |
| 平台与硬件验证 | Win32 热键注册/注销已验证；Windows 焦点外触发、真实串口和游戏目标缺少完整验收记录 | 在目标机验证实际热键触发、设备枚举和目标安装/数据流；仅测试已声明支持的能力。Unreal/VaM 没有仓库自有 installer，不能标记通用安装完成 |
| 真实模型调用 | 确定性 Agent 和 mock provider 已覆盖业务闭环；历史在线 smoke 因无凭据跳过 | 在已配置 provider 的环境显式运行 `studio_agent_model_smoke`，记录真实工具调用/错误路径，不能把 skip 当作通过 |
| 浏览器交互缺口 | 空白区拖放回滚在旧验收中没有稳定的浏览器手势用例 | 为该路径补充可重复的浏览器回归，确认权威图和 UI 一致 |

首发支持范围仍为 Windows/Linux Chromium。其他浏览器、远端跨主机视频延迟与时钟同步、gapless 音频及音视频同步没有验收承诺；如扩大支持范围，应单独定义测试。性能与帧计数走 monitor/data，不进入 service stateFields。

## 后续扩展提案

这些是 P7 之后的能力扩展，不是未完成的 Qt 功能搬迁。

- **第三方扩展包**：当前 registry 只装配受信任的本地源码。后续需定义版本化 manifest、显式目录、文件 hash、能力声明和缺失扩展诊断；再拆分 TCode / Template 适配包，验证未安装、安装、卸载和原图恢复。开发者模板与稳定 JS API 应以这组契约为基础。不可信代码需要独立 iframe/Worker 边界。
- **通用 Zenoh 数据桥接**：尚无任意流的浏览器订阅接口。设计时需明确项目/service/key/schema、权限与大小、latest/有界队列、epoch/sequence/timestamp 和重连；高频流用二进制通道与 Worker，音视频继续经媒体网关。先完成骨骼流端到端示例。
- **跨标签媒体与共享看板**：当前 source/quality peer 池只在单页面内共享，固定输出排序只保存在 localStorage。独立弹出页需设计 owner 关闭后的接管；任意标签并发可评估网关共享编码，每个标签仍需自己的 peer。项目级共享看板布局另需持久化契约。
- **加载体积**：Monaco 和 3D 扩展已按需加载；进一步压缩包体需用生产构建与实际加载时间评估。

## 可复用的历史基准

基线机器为 Linux x86_64、Intel Core i7-8700（12 logical CPUs）、GTX 1080 / UHD 630，使用软件 WebRTC 编码。数值只代表该机器和当时版本，不外推到其他平台。

- [5 分钟组合](evidence/p3-combined-5m.json)：main 26.86 FPS、thumbnail 7.69–7.73 FPS，含音频，控制 p95 47.38 ms，错误 0。
- [30 分钟稳定性](evidence/p3-stability-30m.json)：main 27.52 FPS、thumbnail 7.66–7.69 FPS，控制 p95 43.18 ms，最后 5 分钟 RSS 净增 19.9 MiB，session/source 清理归零。
- [复杂图基准](evidence/p4-graph-performance.json)：300 节点/600 边首次可交互 1.680 s、拖拽约 60 FPS、交互 p95 26.2 ms；1000/2000 压力场景首次可交互 4.589 s。300/600 的预算为拖拽 ≥50 FPS、交互 p95 ≤100 ms；压力场景记录曲线，不代替常规预算。

可复验入口：

```sh
pixi run -e web-studio-test studio_graph_bench
pixi run -e web-studio-test studio_p3_combined_bench
pixi run -e web-studio-test python scripts/web_studio/benchmark_p3_combined.py --duration 1800 --output docs/development/evidence/p3-stability-30m.json
pixi run -e web-studio-test studio_release_smoke
pixi run -e web-studio-test studio_no_qt_check
pixi run -e web-studio studio_agent_model_smoke
```

架构重构的其他提案见 [后续架构工作](remaining-architecture-work.md)；现行图格式见 [图 schema](graph-schema.md)，远程连接见 [Web Studio 远程访问](web-studio-remote-access.md)。
