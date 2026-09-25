# Web Studio 图编辑与扩展精修

更新：2026-09-23。本文记录 P7 之后的图编辑工作和仍需实现的扩展边界。

## 已落地

- Node Library 提供按 Service 或 Category 分组的折叠列表；搜索同时匹配算子、服务、分类和标签，搜索结果自动展开。分组只改变浏览方式，不改变 catalog 身份。
- Inspector 暴露节点声明的 commands；`showOnNode` 命令也显示节点内按钮。参数表单按照声明的类型、默认值、枚举和必填项输入。Service 命令调用运行时 command endpoint 并显示返回结果；Operator 命令写入其声明的 command input 隐藏状态，界面只声明已提交，不把异步执行当作已完成。
- Video Viz 继续使用节点内 WebRTC 预览。3D Viz 收到有效骨架场景并进入可见区域后才创建节点内 Three.js 画布，按约 15 FPS 更新；全屏 3D 视图保持交互帧率。节点上的查看按钮在同一浏览器标签中打开对应输出，URL 使用 `?view=outputs&node=...`，浏览器前进/后退可回到图。
- Live Outputs 聚合 text、wave、track、TCode、video 和 3D 输出，提供固定输出及排序，顺序存在本机浏览器的 localStorage。该看板当前只管理在线输出，项目级共享布局尚未实现。
- 前端 TCode renderer 与 Template 工具已移入 `src/extensions/`，通过显式、类型化的本地注册表接入。注册时拒绝重复 extension、renderer、tool ID 和重叠命令前缀；扩展可从 `src/extensions/sdk.ts` 使用 presentation hooks、视频组件、command 和 state API。模块按需加载。

## 扩展下一步

当前注册表是**受信任的本地源码扩展边界**，不是可安装的第三方包系统。后端 `f8.viz.tcode` 仍由 Studio runtime 静态注册，Template Match 的处理仍属于原服务。要完成真正的第三方扩展，按以下顺序实施：

1. 定义 `f8studio-extension/1` manifest：唯一 ID、版本、Web bundle、backend package、所需能力和 renderer/tool/operator 声明。Studio 从明确配置的本地目录加载；锁定 manifest 版本和文件 hash，不从 CDN 执行代码。卸载扩展时 catalog 中的节点显示“扩展缺失”，原项目文档不删除。
2. 把 TCode runtime/operator spec 和 Template Match UI/服务适配分别迁入独立包。核心 Studio 仅装配显式启用的包；在未安装这些包的干净环境验证服务启动、catalog 缺席和项目错误提示。再验证安装后原图文档可运行。
3. 为开发者提供模板和稳定的 JS API：输出 renderer 接收有界 presentation payload；工具组件可调用声明的 command 和 state；视频复用 `PresentationVideo`；图节点内控件使用固定尺寸，并由 host 控制生命周期。扩展错误在 UI 中显示并记录 stack，不得吞掉异常。
4. 自定义 Zenoh 流经后端受控桥接进入浏览器。订阅请求应指定项目、service、精确 key、payload schema 与 `latest`/有界 queue 策略；服务端验证权限和大小，提供 epoch、sequence、timestamp 与断线重连。高频数据走二进制 WebSocket/WebRTC data channel，在 Worker 中解码并按动画帧绘制；不能放进 service stateFields 或 graph 事件。视频/音频继续走媒体网关。先用骨骼流做端到端扩展示例，再开放通用桥接。
5. 受信任扩展可在主页面加载本地 bundle。需要运行不可信代码时，另用隔离 iframe/Worker 和限定的消息桥；不能把主页面的 API、令牌或任意 Zenoh key 直接交给不可信脚本。

## 多标签与媒体

当前 `VideoSessionPool` 在**单个页面内**按 source/quality 复用一个 peer，节点预览与同页 Outputs 视图不重复协商。2026-09-23 的 Chrome 探针确认：Studio 主页面打开的同源弹出页可以直接使用主页面的 `MediaStream`；主页面关闭后该流结束。`SharedWorker` 中 `RTCPeerConnection` 不可用，`MediaStream` 经 `BroadcastChannel` 发送会抛 `DataCloneError`。因此当前交互使用同标签页视图，不能宣称任意浏览器标签共享同一个 peer。若以后需要独立弹出页，必须设计主页面关闭后的 owner 接管；若需要任意标签页并发，网关侧共享编码可降低重复编码，但每个标签仍需自己的 WebRTC peer。

## 验收

- Node Library 分组和 command 参数/运行时路径有组件测试。
- 原图编辑和工作区桌面 E2E 通过；新增 3D 节点预览在桌面/移动端完成 canvas 像素、截图和同页跳转验证。
- Outputs 固定、排序、刷新恢复在桌面/移动端 E2E 通过。
- 后续扩展包与 Zenoh 桥接在完成独立测试前不标记为已实现。
