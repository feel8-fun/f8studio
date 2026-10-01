# Cloud 独立仓库

云端资产服务源码由 [feel8-fun/f8assetcloud](https://github.com/feel8-fun/f8assetcloud) 管理，Studio 通过根目录 `cloud/` submodule 固定已发布提交。原 `packages/f8assetcloud_worker/` 的相关 Git 历史已迁入独立仓库。

## 获取和开发

```bash
git submodule sync --recursive
git submodule update --init --recursive
cd cloud
git switch main
npm ci
npm --prefix console_web ci
npm run check
```

Studio 开发环境也可以从主仓库运行 `pixi run -e web-studio npm --prefix cloud run check`。cloud 独立仓库只要求 Node.js 24 和 npm，不依赖 Studio 的 Pixi 环境或 SDK 源码。

本地 `.dev.vars`、`.wrangler/` 和依赖缓存不纳入版本管理；变量模板和生产资源配置由 cloud 仓库维护。具体本地运行、数据库迁移和部署命令见 cloud 的 README。

## 版本和集成

cloud 的版本由自己的 `package.json` 和 `vX.Y.Z` Git 标签管理，管理前端使用同一发行版本。资产内容修订号、`/v1` API 版本与仓库发行版本分别维护。API 兼容性变化需要协调 Studio 消费者，数据库迁移随 cloud 发行。

先在 cloud 仓库提交、验证并推送，再在 Studio 提交新的 gitlink。Studio 只记录集成使用的 SHA，不随 Studio 发行自动部署 cloud。独立 CI 执行后端测试、前端测试和构建；生产部署、远程迁移继续通过 cloud 的明确部署命令执行。
