# P2 Cloud 与 Studio 接入

2026-10-09。实现已覆盖 Cloud 发布与 Library 闭环；线上部署和远程数据库迁移不由 Studio gitlink 更新触发。

## 数据与合同

Cloud 新入口为 `/v2/library`，支持 graph、component、variant。Component 和 Variant 都使用 `f8component/1`；完整图使用 `f8graph/4`。内容统一包在 `f8publication/1` 中。Studio 合同生成工具向独立 Cloud 仓库发布 schema、TypeScript 类型和历史/hash fixture，Cloud 运行时不加载 Python 或 Studio。

`0002_publication_library.sql` 新增独立资产、不可变版本、持久发布回执和社区关系表；不修改 `0001` 或任何旧资产 blob。旧 API/内容继续保留，旧格式转换属于 P4。

发布由一个 INSERT 和数据库触发器完成原子检查及写入。更新必须提供 expectedVersion；相同内容保留版本，重复请求返回原回执，同一请求 ID 配不同内容报冲突。元数据另走更新接口。规范化 hash 与 Python/Web 的 fixture 一致，定义 default/examples 不清理；实例值按 persistent/publishable、访问方式及上游绑定检查。

每个 publication 限 1 MiB，保证 D1 行大小；附件和资源托管留待后续。派生作品记录固定来源，校验访问权与该来源版本的许可；未知/不可复用许可不自动授予派生发布权。当前支持常见 permissive 许可和保留同一许可的指定 copyleft/ShareAlike 许可。

## Studio 行为

在全局 Settings → Cloud 配置 HTTPS origin、登录；本地开发允许 loopback HTTP。默认不开启 Cloud。也可通过 `F8STUDIO_CLOUD_URL` 配置。连接前检查 v2 capabilities，旧线上后端不能直接作为 v2 provider 使用。

2026-10-09 实测生产域名 `https://assetcloud.feel8.fun` 的 `/v1/auth/providers` 返回 200，但 `/v2/library/capabilities` 返回 404：线上仍缺少 Library v2。Studio 将这种连接失败明确标记为 `unsupported_cloud_api`，提示先迁移并部署 Cloud；失败保留已有连接和登录信息，同一 URL 的重新保存也会检查兼容性。部署后才可用该域名连接新版 Studio。

登录复用 PKCE loopback 流程。Studio Server 保存并刷新 token，原子写入权限为 0600 的独立文件；前端只获得账号状态和授权 URL。认证回调只允许完成已有、短期、一次性的 state，不开放其他 Studio API。远程托管 Studio 的登录流程不在此次范围。

Add from Library 共用 Local/Online 搜索；网络失败不阻塞本地。在线结果只下载元数据，预览或添加时才获取固定版本。下载校验 publication/hash/依赖，通过后复用本地模板插入事务。Cloud 来源、映射与图一起提交；失败不会改变项目。在线浏览/插入不增加 local_assets。

Assets 使用 My Local、My Cloud、Discover、Following 四个入口。My Local 在同一个列表管理本地图、Component 和 Variant；四个入口共用 All / Graphs / Components / Variants 筛选，搜索和结果统一在主左侧栏，右侧只展示所选内容的详情。图保留打开、发布、保存为 Component 和本地快照操作；草稿保留编辑、预览、插入和发布操作。同名作品按类型和独立身份区分，本地项目 ID 与草稿 ID 也不会互相覆盖。旧 Projects / History 链接进入 My Local 的 Graphs 筛选。未配置 Cloud 时仍显示入口及连接说明；本地草稿标记为仅保存在当前设备，并展示当前账号对应的云端发布基线或其他作者来源。删除本地草稿不会删除云端作品。

三个 Cloud 入口在主左侧栏提供 All / Graphs / Components / Variants 类型筛选，
与关键词、账号和关注范围一同传给 Cloud。筛选发生在数据库分页之前；cursor
绑定查询、入口和类型，切换类型从第一页重新搜索并清空详情，迟到响应不会覆盖
当前选择。未指定 `kind` 仍表示全部类型，旧版四项 cursor 仅用于未筛选的请求。
作品名不是唯一键，同名的 Graph / Component / Variant 使用各自的 `assetId`，
更新、删除、点赞及关注均独立；列表图标、类型文字和详情类型标记帮助区分。

类型筛选验收：59 项 Cloud 后端、27 项 Cloud 服务端集成、24 项相关 Web
测试及桌面/手机两条真实浏览器流程通过，覆盖同名作品、筛选后的分页、旧 cursor
兼容、迟到响应、关注与所有权范围，以及筛选按钮文字不溢出。严格 Python /
TypeScript 检查和 Web 构建通过。新增查询能力需要加载新版 Cloud 和 Studio
服务端；无需额外数据库迁移。

My Cloud 只查询当前账号的作品，作者直接管理云端名称、Markdown 介绍、标签和 public/private；这些修改不增加内容版本，也不修改本地草稿。内容更新从本地草稿保存后显式发布；已有关联草稿可直接打开，不重复克隆。他人的作品详情提供点赞/取消、关注、使用及独立草稿操作，不提供修改云端信息、删除或覆盖入口。Studio 和 Cloud 均校验作者身份；派生发布使用新资产身份并保留固定来源。作者可从 More publication actions 删除整份发布，确认后移除列表、所有版本、点赞和作品关注；作者关注、本地草稿及已插入项目保留。删除回包丢失后可重试；旧发布请求不能恢复已删除作品。保留最小重试回执，数据库迁移 `0003_publication_deletion.sql` 清除回执中的原始内容和介绍。删除本地发布关联后可重新发布为新作品，派生来源保留。

版本提示不自动下载、部署或替换现有节点。Graph 可预览和打开为独立项目；模板可显式创建独立本地草稿。

实测准备补齐：Cloud 详情统一选择固定版本，模板插入、独立草稿和完整图打开使用同一引用。历史版本返回其自身许可，避免用最新许可描述旧内容。完整图发布复用配置选择界面，`excludedStates` 与发布重试一同保存；空排除选择保留早期待发送请求的指纹。搜索、详情和模板下载提供显式重试；Cloud URL 编辑不再被迟到的初始状态覆盖，新增 Disconnect Cloud，切换账号或作品时不会应用过期响应。

`pixi run -e web-studio cloud_sandbox` 提供前台运行的独立本地 Cloud，使用生产路由、真实迁移和持久 SQLite 数据库，预置两个已验证的普通账号。重启保留作品、登录、社区关系及发布回执；不需要 Cloudflare 认证，不连接生产数据库。具体步骤见 [Cloud Library 本地实测](cloud-library-manual-test.md)。沙盒不替代 Worker/D1 线上验收。

保存草稿只写本地。Publish 提交已保存版本及显式许可；未保存的本地编辑提示先保存。待发送快照先落库，断网/重启后重试同一快照，成功后记录发布身份和基线。浏览器保存账号/registry/草稿对应的请求 ID 和作者选项；刷新页面或继续本地编辑后，Retry 仍确认原来的保存版本。结果不确定时锁定该次发布选项，明确拒绝后才允许修改重试。发布冲突保留草稿，由作者决定另建草稿或新作品；第一版不自动合并。Manage Cloud listing 转入云端管理入口，避免把本地保存和云端信息修改混在同一表单。

发布依赖包含 Web Studio 内置 runtime：`f8.pystudio` 和其已注册 operators
明确归属于 application extension `webstudio`。Platform 的 application 条目没有
`serviceClasses` 时，Studio 用实际 builtin registry 补齐；存在的版本与 capabilities
保留，外部未知节点仍必须声明扩展归属。发布准备校验失败返回带具体原因的 422，
不保存待发送快照，也不发送 Cloud 请求；界面可以结束该次待确认状态。

发布时将定义 JSON 中的整数浮点表示（例如 default/examples 内的 `0.0`）
统一成整数，并用现有定义哈希算法重建 service/operator/host binding 引用，
避免 JavaScript 返回 `0` 后无法校验。定义值的含义、本地项目及草稿内容保留；
类型声明为 float 的字段按 SDK 解码规则恢复。`f8graph/4`、`f8component/1` 和
定义哈希算法不变，历史定义仍按原引用严格校验，不通过跳过哈希来兼容。

## 检查与发布准备

2026-10-09 管理界面补齐：238 项 Web 完整回归、追加后最终 9 项管理界面/发布相关测试、18 项 Cloud 与生成合同服务端测试通过。Cloud/Component/Variant 的 3 条桌面浏览器流程通过；Cloud 流程覆盖真实两账号切换、作者云端介绍编辑、他人点赞/取消、独立草稿派生发布及来源身份。严格类型检查、生成合同一致性及 Web 构建通过。当前运行的 Studio 进程需要重启后加载新增 API。

2026-10-09 实测准备验收：248 项 Web 完整回归、27 项 Cloud/合同服务端测试、55 项 Cloud 后端及 21 项 console 回归通过；本地 D1 迁移检查、严格 Python/TypeScript 检查和构建通过。桌面 Component/Variant 与桌面、手机 Cloud 浏览器流程通过。新增覆盖旧版本草稿与对应许可、发布配置的断网恢复、旧请求指纹兼容、迟到响应隔离，以及本地沙盒重启后账号、作品和发布回执恢复。

```sh
pixi run -e web-studio npm --prefix cloud run check
pixi run -e web-studio-test python scripts/web_studio/generate_contracts.py --check
pixi run -e web-studio-test python -m pytest extensions/f8webstudio/f8studio_server/tests/test_cloud.py -q
pixi run -e web-studio-test studio_web_test
pixi run -e web-studio-test studio_web_build
pixi run -e web-studio-test npm --prefix extensions/f8webstudio/f8studio_web run test:e2e -- e2e/cloud.spec.ts --project=desktop
```

Cloud `check` 包括 TypeScript、真实迁移的临时本地 D1 校验、后端/旧 console 回归和 console 构建。Python 集成使用生产 Worker/auth 路由及所有迁移，通过隔离的真实 HTTP 请求验证；浏览器使用测试 Studio 端口及独立 Cloud 数据库。正式迁移仍使用 Cloud 仓库已有的明确远程迁移/部署命令，执行前应完成线上数据备份与测试环境验收。

全局 Settings 使用 AI / Cloud 左侧分类，Cloud URL、连接、登录和退出统一在 Cloud 分类管理。Assets 不再嵌入连接配置。My Local 的 Graphs 筛选用于完整图；Local snapshots 是保存在本机的命名检查点，可修改名称和备注、恢复和删除，与 Cloud 内容版本相互独立。草稿详情突出 Variant / Component 的名称、介绍和预览，发布选项在独立对话框中；JSON 编辑、导出和删除通过 More draft actions 进入。

2026-10-09 侧栏与发布删除验收：My Cloud / Discover / Following 的
搜索和作品列表统一到主侧栏，右侧为详情，模板介绍不重复显示。259 项 Web
测试、57 项 Cloud 测试、14 项 Cloud 集成测试和生成合同检查通过；桌面与手机
共 4 条浏览器流程覆盖所有入口、所有权、取消删除和删除回包丢失后重试。
本地 D1 的三份迁移通过真实 Wrangler 校验，严格类型检查和前端构建通过。
删除功能上线需更新 Studio/Cloud 服务端并应用 `0003_publication_deletion.sql`；
持久本地沙盒重启时自动应用该增量迁移。

2026-10-09 My Local / My Cloud 合并验收：35 项相关 Web 测试和桌面、手机
共 4 条浏览器流程通过，覆盖同名本地图及草稿、独立选择、关键词和类型筛选、
迟到响应隔离、旧 Projects / History 链接、本地图保存为 Component、草稿编辑、
快照管理与 Cloud 发布/使用。TypeScript 检查和 Web 构建通过，筛选按钮文字
不溢出。本次为前端导航与展示调整，刷新页面即可使用，无需新增服务端迁移。
