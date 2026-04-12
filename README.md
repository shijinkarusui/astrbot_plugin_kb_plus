# astrbot-plugin-kb-plus

AstrBot 知识库增强检索插件。

## 功能

- `kb list [关键词]`
  - 列出 AstrBot 当前可用知识库
  - 列出每个知识库下的文件
  - 支持对库名、文件名模糊过滤
- `kb ask [库名,文件名...] [问题]`
  - 由主对话模型在 `kb` 命令上下文里主动调用 `astr_plus_kb_*` 工具
  - 优先按指定知识库或指定文件检索
- `kb free [问题]`
  - 由主对话模型自行决定先列出、匹配还是直接检索
  - 限制只使用 `astr_plus_kb_*` 工具链
- `kb topk`
  - 查看当前插件默认 `top_k` 与最大 `top_k` 配置

## 提供的函数工具

- `astr_plus_kb_list`
  - 列知识库和文件
- `astr_plus_kb_match`
  - 对库名和文件名做模糊匹配
- `astr_plus_kb_search`
  - 对指定知识库或指定文件执行检索
  - `top_k=0` 时使用插件配置中的默认值

## 检索策略

- 未指定文件时：
  - 复用 AstrBot 内置 `kb_manager.retrieve(...)`
  - 使用内置向量检索、稀疏检索、融合和重排能力
- 指定文件时：
  - 先精确定位目标文档
  - 再只在这些文档对应的 chunks 中进行检索与排序
  - 比单纯按结果 `doc_name` 过滤更严格

## 插件配置

本插件支持 WebUI 插件配置，配置文件为 `_conf_schema.json`。

可配置项：

- `default_top_k`
  - 默认返回结果数量
  - 当工具调用没有传 `top_k`，或者传 `0` 时生效
- `max_top_k`
  - 允许的最大返回结果数量
  - 实际请求值会被截断到这个上限
- `strict_doc_chunk_limit`
  - 严格文件检索时，每个文件最多读取多少个 chunk
- `enable_multi_round_hint`
  - 是否在 `kb ask` / `kb free` 中强化多轮工具调用提示

## 注意事项

- 本插件不会修改 AstrBot 原生知识库工具的全局行为。
- 仅在 `kb ask` / `kb free` 命令场景下，向主对话模型限制暴露 `astr_plus_kb_*` 工具。
- 如果你的主对话模型工具调用能力较弱，建议优先使用 `kb list` 和 `kb ask`。

## 开发参考

- [AstrBot](https://github.com/AstrBotDevs/AstrBot)
- [AstrBot Plugin Development Docs (Chinese)](https://docs.astrbot.app/dev/star/plugin-new.html)
- [AstrBot Plugin Development Docs (English)](https://docs.astrbot.app/en/dev/star/plugin-new.html)
