# P2 Cloud 与 Studio 接入

2026-10-09。实现已覆盖 Cloud 发布与 Library 闭环；线上部署和远程数据库迁移不由 Studio gitlink 更新触发。

## 数据与合同

Cloud 新入口为 `/v2/library`，支持 graph、component、variant。Component 和 Variant 都使用 `f8component/1`；完整图使用 `f8graph/4`。内容统一包在 `f8publication/1` 中。Studio 合同生成工具向独立 Cloud 仓库发布 schema、TypeScript 类型和历史/hash fixture，Cloud 运行时不加载 Python 或 Studio。

`0002_publication_library.sql` 新增独立资产、不可变版本、持久发布回执和社区关系表；不修改 `0001` 或任何旧资产 blob。旧 API/内容继续保留，旧格式转换属于 P4。

发布由一个 INSERT 和数据库触发器完成原子检查及写入。更新必须提供 expectedVersion；相同内容保留版本，重复请求返回原回执，同一请求 ID 配不同内容报冲突。元数据另走更新接口。规范化 hash 与 Python/Web 的 fixture 一致，定义 default/examples 不清理；实例值按 persistent/publishable、访问方式及上游绑定检查。

每个 publication 限 1 MiB，保证 D1 行大小；附件和资源托管留待后续。派生作品记录固定来源，校验访问权与该来源版本的许可；未知/不可复用许可不自动授予派生发布权。当前支持常见 permissive 许可和保留同一许可的指定 copyleft/ShareAlike 许可。

## Studio 行为

在 Assets 的 Feel8 Cloud 区域配置 HTTPS origin、登录；本地开发允许 loopback HTTP。默认不开启 Cloud。也可通过 `F8STUDIO_CLOUD_URL` 配置。连接前检查 v2 capabilities，旧线上后端不能直接作为 v2 provider 使用。

登录复用 PKCE loopback 流程。Studio Server 保存并刷新 token，原子写入权限为 0600 的独立文件；前端只获得账号状态和授权 URL。认证回调只允许完成已有、短期、一次性的 state，不开放其他 Studio API。远程托管 Studio 的登录流程不在此次范围。

Add from Library 共用 Local/Online 搜索；网络失败不阻塞本地。在线结果只下载元数据，预览或添加时才获取固定版本。下载校验 publication/hash/依赖，通过后复用本地模板插入事务。Cloud 来源、映射与图一起提交；失败不会改变项目。在线浏览/插入不增加 local_assets。

Assets 提供 Online Library、My publications、Following，以及点赞/取消、作者关注和资产更新关注。版本提示不自动下载、部署或替换现有节点。Graph 可预览和打开为独立项目；模板可显式创建独立本地草稿。

保存草稿只写本地。Publish 提交已保存版本及显式许可；待发送快照先落库，断网/重启后重试同一快照，成功后记录发布身份和基线。浏览器保存账号/registry/草稿对应的请求 ID 和作者选项；刷新页面或继续本地编辑后，Retry 仍确认原来的保存版本。结果不确定时锁定该次发布选项，明确拒绝后才允许修改重试。发布冲突保留草稿，由作者决定另建草稿或新作品；第一版不自动合并。Update listing 单独更新名称、介绍、标签和可见性，不增加内容版本。

## 检查与发布准备

```sh
pixi run -e web-studio npm --prefix cloud run check
pixi run -e web-studio-test python scripts/web_studio/generate_contracts.py --check
pixi run -e web-studio-test python -m pytest extensions/f8webstudio/f8studio_server/tests/test_cloud.py -q
pixi run -e web-studio-test studio_web_test
pixi run -e web-studio-test studio_web_build
pixi run -e web-studio-test npm --prefix extensions/f8webstudio/f8studio_web run test:e2e -- e2e/cloud.spec.ts --project=desktop
```

Cloud `check` 包括 TypeScript、真实迁移的临时本地 D1 校验、后端/旧 console 回归和 console 构建。Python 集成使用生产 Worker/auth 路由及所有迁移，通过隔离的真实 HTTP 请求验证；浏览器使用测试 Studio 端口及独立 Cloud 数据库。正式迁移仍使用 Cloud 仓库已有的明确远程迁移/部署命令，执行前应完成线上数据备份与测试环境验收。
