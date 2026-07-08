# 基本信息
- 姓名: 李思
- 邮箱: lisi@email.com
- 电话: 13800002222
- 学历: 浙江大学 计算机工程 本科 2016-2020

# 技能

Python, Django, FastAPI, Docker, MySQL, Redis, Linux

---

## 项目：订单系统微服务改造

- 时间：2023.01-2023.06
- 角色：核心开发
- 技术栈：Python, Django, FastAPI, Docker, Kubernetes, MySQL, Redis, RabbitMQ
- 详情：
  负责将公司核心订单系统从 Django 单体架构拆分为微服务。
  设计并实现了服务间通信方案，采用 RabbitMQ 异步消息 + HTTP 同步调用混合模式。
  数据库层面实施了分库分表策略，订单表按用户 ID 哈希拆分至 4 个分库。
  搭建了基于 Docker Compose 的本地开发环境，K8s 生产环境部署。
  改造后系统 QPS 从 2000 提升至 15000，P99 延迟从 800ms 降至 120ms。
  编写了完整的服务间接口文档和运维手册，指导 2 名初级工程师熟悉新架构。

## 项目：公司核心 API 重构

- 时间：2023.07-2024.03
- 角色：后端负责人
- 技术栈：FastAPI, Python, Redis, PostgreSQL, Docker, Prometheus, Grafana
- 详情：
  使用 FastAPI 重写公司核心 API 层，将原有的 Django REST 接口全部迁移。
  引入异步编程模型，接口响应时间平均降低 40%。
  集成 Redis 缓存层，热点数据命中率 95% 以上，数据库查询压力降低 60%。
  系统上线后日均处理 200 万订单，峰值 QPS 达到 5000，P99 延迟控制在 200ms 以内。

## 项目：CI/CD 流水线搭建

- 时间：2022.08-2022.10
- 角色：DevOps 负责人
- 技术栈：GitLab CI, Docker, Kubernetes, Helm, Prometheus, Grafana
- 详情：
  从零搭建公司 CI/CD 自动化流水线，覆盖 5 个核心微服务。
  实现代码提交 → 自动测试 → 构建 Docker 镜像 → 部署到 K8s 集群的全流程。
  集成 Prometheus + Grafana 监控面板，告警响应时间从 30 分钟缩短到 5 分钟。
  编写部署文档和回滚 SOP，3 个月内零故障部署。

## 项目：用户认证与权限管理系统

- 时间：2020.07-2021.06
- 角色：全栈开发
- 技术栈：Django, Vue.js, MySQL, Redis, Celery
- 详情：
  独自设计并开发了完整的用户认证与权限管理系统。
  后端使用 Django REST framework，实现 JWT 认证、RBAC 权限模型、操作审计日志。
  前端使用 Vue.js + Element UI，实现权限可视化管理界面。
  集成 Redis 做 Session 管理，Celery 处理异步任务（邮件通知、数据导出）。
  系统支撑起 500+ 内部用户的日常访问。

## 项目：支付系统对接

- 时间：2021.07-2021.12
- 角色：后端开发
- 技术栈：Django, Redis, MySQL, Docker
- 详情：
  负责对接支付宝、微信支付、银联三家支付渠道。
  设计统一的支付网关层，抽象各家 SDK 差异，业务方只需调用统一接口。
  实现幂等性保证，处理支付回调的重复通知、网络超时等边界情况。
  设计对账系统，自动比对平台订单与支付渠道账单，差异率低于 0.01%。
