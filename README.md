# 两张昇腾卡部署 Qwen3.8-27B W8A8：从下载到图像、推理和工具调用

> 这篇是同一台 16 卡昇腾服务器上的第二段实战。DeepSeek 占用一组 8 卡后，我们没有让剩下的卡闲着，而是拿其中两张部署了一个 Qwen3.8-27B W8A8 服务。和前面的 DeepSeek 排障相比，这次顺利得多，但也遇到了几个很有代表性的小坑：ModelScope 显示下载完成却伴随失败警告、模型明明能看图但 Agent 说“不支持图片”，以及模型能生成工具调用却不代表工具已经被执行。

## 这次最后跑成了什么样

```text
客户端 / Agent
      │ OpenAI-compatible API
      ▼
Nginx / API Key 网关
      │ /v1/qwen38/chat/completions
      ▼
vLLM-Ascend :7001
      │
      ├── Qwen3.8-27B W8A8
      ├── TP=2 / DP=1
      ├── NPU 8,9（示例，可自行替换）
      ├── 128K context / 32K output cap
      ├── Qwen3.5 MTP，预测 3 token
      └── 文本、视觉理解、reasoning、tool_calls 均通过验证
```

| 项目 | 本次配置 | 说明 |
|---|---:|---|
| 模型 | `Eco-Tech/Qwen3.8-27B-w8a8` | 约 30GiB，10 个权重分片 |
| 架构 | `Qwen3_5ForConditionalGeneration` | `config.json` 内带视觉配置，是原生视觉语言模型 |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.23.0` | 本次实际验证版本 |
| NPU | 2 × Ascend 910B2C 64GB | 使用示例设备 8、9 |
| 并行 | `TP=2 / DP=1` | 与官方 A2/A3 示例一致 |
| 最大上下文 | 131072 | 输入与输出之和的上限 |
| 单轮最大输出 | 32768 | 服务端兜底，防止 Agent 放大预算 |
| Batch tokens | 16384 | 官方示例值，本环境实测可用 |
| 最大序列数 | 32 | 结合实际并发和 HBM 再调整 |
| Prefix Cache | 开启 | 重复前缀场景可节省 prefill |
| MTP | `qwen3_5_mtp × 3` | Qwen3.8 沿用这一 MTP 方法名 |
| 图模式 | `FULL_DECODE_ONLY` | 减少 decode 阶段的算子下发开销 |

这里有个挺让人安心的地方：我们跑通以后再回头看官方文档，关键参数基本对得上。vLLM-Ascend 的 [Qwen3.8-27B 官方教程](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-27B.html)说明该模型从 vLLM-Ascend 0.23.0 开始支持，并给 Atlas 800 A2/A3 提供了 `TP=2`、131072 上下文、16384 batch tokens、0.85 HBM 利用率、MTP 3-token 和 `FULL_DECODE_ONLY` 的启动示例。

## 第一步：下载模型，但别只相信“Snapshot ready”

先准备目录：

```bash
mkdir -p \
  /srv/models/Qwen3.8-27B-w8a8 \
  /srv/modelscope-cache \
  /srv/hf-cache \
  /srv/cache \
  /srv/tmp \
  /srv/download-logs
```

后台下载：

```bash
download_dir=/srv/models/Qwen3.8-27B-w8a8
log_file=/srv/download-logs/qwen38-27b-w8a8-download.log
pid_file=/srv/download-logs/qwen38-27b-w8a8-download.pid

nohup env \
  MODELSCOPE_CACHE=/srv/modelscope-cache \
  HF_HOME=/srv/hf-cache \
  XDG_CACHE_HOME=/srv/cache \
  TMPDIR=/srv/tmp \
  modelscope download \
  Eco-Tech/Qwen3.8-27B-w8a8 \
  --local-dir "$download_dir" \
  --max-workers 8 \
  > "$log_file" 2>&1 &

download_pid=$!
printf '%s\n' "$download_pid" > "$pid_file"
printf 'download_pid=%s\n' "$download_pid"
```

查看进度：

```bash
download_pid=$(sed -n '1p' /srv/download-logs/qwen38-27b-w8a8-download.pid)
ps -fp "$download_pid"
du -sh /srv/models/Qwen3.8-27B-w8a8
tail -n 30 /srv/download-logs/qwen38-27b-w8a8-download.log
```

我们当时遇到的日志是：进度显示 100%，随后又提示有 3 个文件下载失败，最后仍然打印 `Snapshot ready`。这类情况下不要凭最后一行下结论，至少做下面三项检查：

```bash
MODEL_DIR=/srv/models/Qwen3.8-27B-w8a8

echo '完整权重分片：'
find "$MODEL_DIR" -maxdepth 1 -type f \
  -name 'quant_model_weights-*.safetensors' | wc -l

echo '未完成文件：'
find "$MODEL_DIR" -type f -name '*.incomplete' | wc -l

echo '目录大小：'
du -sh "$MODEL_DIR"
```

本次正确结果是：10 个权重分片、0 个 `.incomplete`、目录约 30GiB。再检查索引实际引用的文件是否全部存在：

```bash
MODEL_DIR=/srv/models/Qwen3.8-27B-w8a8 python3 - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
index = model_dir / "quant_model_weights.safetensors.index.json"
data = json.loads(index.read_text(encoding="utf-8"))
shards = sorted(set(data.get("weight_map", {}).values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
size = sum((model_dir / name).stat().st_size for name in shards if (model_dir / name).is_file())

print("referenced_shards =", len(shards))
print("referenced_size_GiB =", round(size / 1024**3, 2))
print("missing_shards =", missing)
PY
```

预期为 10 个引用分片、约 29.92GiB、`missing_shards = []`。如果确实缺分片，直接用同一条 `modelscope download` 命令重跑即可，它会复用已经完成的文件。

## 第二步：用两张卡启动服务

完整脱敏脚本见 [`examples/docker-run-qwen38.sh`](examples/docker-run-qwen38.sh)。默认值如下：

```bash
export QWEN_MODEL_PATH=/srv/models/Qwen3.8-27B-w8a8
export QWEN_VISIBLE_DEVICES=8,9
export QWEN_PORT=7001
export HOST_NIC=bond0

bash examples/docker-run-qwen38.sh
```

脚本实际执行的核心启动参数是：

```bash
vllm serve /models/qwen38 \
  --host 0.0.0.0 \
  --port 7001 \
  --served-model-name qwen3.8 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --quantization ascend \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --generation-config auto \
  --override-generation-config '{"max_new_tokens":32768}'
```

几个容易误解的点：

- 模型叫 Qwen3.8，但投机解码方法仍是 `qwen3_5_mtp`。这是官方定义的实现名，不是写错版本；
- `--max-model-len` 是输入加输出总窗口，`max_new_tokens` 是服务端单轮输出上限；
- `--reasoning-parser qwen3` 负责把思考内容放到 `reasoning` 字段，不要让 `</think>` 混进普通正文；
- `--tool-call-parser qwen3_xml` 负责把模型文本解析成 OpenAI 兼容的 `tool_calls`；
- `--enable-auto-tool-choice` 只允许模型自主选择工具，不会真的执行工具；
- `--enable_cpu_binding` 在官方文档中属于明确的性能优化项，不只是“看起来更专业”的装饰参数。

## 第三步：确认容器和健康状态

```bash
docker inspect -f \
'Status={{.State.Status}} Restart={{.HostConfig.RestartPolicy.Name}} RestartCount={{.RestartCount}}' \
qwen38-vllm

curl --max-time 5 -sS -o /dev/null \
  -w 'health HTTP=%{http_code} time=%{time_total}s\n' \
  http://127.0.0.1:7001/health

curl --max-time 5 -sS http://127.0.0.1:7001/v1/models
```

健康检查 200 之后，再看日志里有没有真正完成初始化：

```bash
docker logs --tail 300 qwen38-vllm 2>&1 | \
grep -E 'Application startup complete|Available KV cache memory|KV cache size|Maximum concurrency|ERROR|Traceback'
```

## 第四步：不要只测“你好”，四种能力分别验证

### 1. 文本与 reasoning 分离

```bash
curl --max-time 120 -sS \
  http://127.0.0.1:7001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {"role": "user", "content": "你好，请用两句话介绍一下自己。"}
    ],
    "max_completion_tokens": 256,
    "temperature": 0,
    "stream": false
  }'
```

正常情况下，最终回答在 `choices[0].message.content`，思考内容在 `choices[0].message.reasoning`。如果正文里仍出现 `</think>`，优先检查 `--reasoning-parser qwen3` 是否真的出现在容器的实际启动命令中。

### 2. 工具调用结构

```bash
curl --max-time 120 -sS \
  http://127.0.0.1:7001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {"role": "user", "content": "请计算 17 乘以 23，必须调用 multiply 工具。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "multiply",
          "description": "计算两个数字的乘积",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {"type": "number"},
              "b": {"type": "number"}
            },
            "required": ["a", "b"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_completion_tokens": 256,
    "temperature": 0,
    "stream": false
  }'
```

预期响应包含 `tool_calls`，函数名为 `multiply`，并以 `finish_reason: tool_calls` 结束。到这里仅证明模型和 vLLM 能生成、解析工具调用。接下来仍需要 Agent 执行函数，把结果作为 `role: tool` 的消息追加回对话，再请求模型生成最终回答。vLLM 官方的 [Tool Calling 文档](https://docs.vllm.ai/en/latest/features/tool_calling/)也把这两层职责分开：服务端产生结构化调用，调用方负责执行和回传结果。

### 3. 图像理解

```bash
curl --max-time 300 -sS \
  http://127.0.0.1:7001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "<PUBLIC_IMAGE_URL>"}
          },
          {
            "type": "text",
            "text": "请描述这张图片中的主要内容。"
          }
        ]
      }
    ],
    "max_completion_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

本次直连测试已经能正确描述远程图片，说明模型、处理器和 vLLM 多模态入口都在工作。vLLM 的 [Multimodal Inputs 文档](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)给出了图片、视频等输入格式。

生产环境还要注意：服务端需要能够访问图片 URL。更稳妥的办法是使用受控对象存储地址或客户端支持的 base64/data URL；不要让模型服务随意访问内网地址，以免引入 SSRF 风险。

### 4. MTP 是否真正生效

```bash
curl -sS http://127.0.0.1:7001/metrics | \
grep -E '^vllm:.*(spec_decode_num_drafts_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total)'
```

只有 draft 和 accepted 计数在请求后持续增长，才能说明投机解码真正进入运行路径。是否加速还要结合接受率、单路输出速度和并发吞吐一起判断。

## 第五步：通过 Nginx 暴露独立路径

我们的目标是让外部使用：

```text
/v1/qwen38/chat/completions
```

而 vLLM 本身仍监听：

```text
http://127.0.0.1:7001/v1/chat/completions
```

Nginx 的最小路由示例：

```nginx
location /v1/qwen38/ {
    proxy_pass http://127.0.0.1:7001/v1/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_read_timeout 600s;
}
```

两个尾部斜杠都不能随手删。按上面的写法，`/v1/qwen38/chat/completions` 才会被重写成 `/v1/chat/completions`。如果返回 `{"detail":"Not Found"}`，通常不是模型没启动，而是路径没有按预期重写。

```bash
docker exec <nginx-container> nginx -t
docker exec <nginx-container> nginx -s reload
```

API Key 校验应由现有网关、Nginx 鉴权模块或专门的 API Gateway 处理。不要把真实 Key 写进仓库、脚本或截图。

## 第六步：Reasonix / Agent 为什么会说“不支持图片”

这是这次最容易让人误判的地方。直连 API 已经能识图，但 Reasonix 最初仍提示“当前服务商没有可用的图片理解模型”。根因不在 vLLM，而在客户端的模型能力元数据：客户端只知道这个名字对应一个文本模型，没有自动推断它具备视觉能力，于是在请求发出前就拦截了附件。

脱敏后的 provider 配置可以写成：

```toml
[[providers]]
name     = "qwen38"
kind     = "openai"
base_url = "http://<GATEWAY_HOST>:<PORT>/v1/qwen38"
models   = ["qwen3.8"]

model_overrides = { "qwen3.8" = { context_window = 131072, vision = true } }
```

保存后需要完整退出并重启 Reasonix，再刷新模型。模型能力模式保持“自动识别”即可。

这里要分清三层：

1. **模型能力**：Qwen3.8-27B 是原生视觉语言模型；
2. **服务能力**：直连 `/v1/chat/completions` 的图片请求成功，证明 vLLM 能接收视觉输入；
3. **客户端声明**：Reasonix 还需要 `vision = true` 才允许把图片发给模型。

少了第三层，前两层再正常，界面仍会显示“不支持图片”。这不是重复配置，而是 OpenAI-compatible `/v1/models` 返回的信息通常不足以表达完整能力元数据。

## 常见问题速查

| 现象 | 常见原因 | 先做什么 |
|---|---|---|
| 下载显示完成，又提示若干文件失败 | 非权重小文件失败，或并发下载局部失败 | 检查分片数、`.incomplete` 和索引引用 |
| `config.json` 顶层很多字段是 `None` | Qwen3.8 采用嵌套配置 | 看 `architectures`、`vision_config` 和子配置，不要只读顶层 |
| 正文出现思考过程或 `</think>` | reasoning parser 未启用或客户端不识别字段 | 检查 `--reasoning-parser qwen3` 和实际响应 JSON |
| 模型输出了工具 JSON，但工具没运行 | vLLM 只负责生成/解析调用 | 检查 Agent 是否执行工具并回传 `role: tool` |
| Agent 说不支持图片，直连图片测试正常 | 客户端能力元数据缺失 | 给该模型设置 `vision = true` 并重启客户端 |
| 网关路径返回 404 | `proxy_pass` 路径或尾部斜杠不对 | 直连 7001，再检查 Nginx 重写结果 |
| 第一次请求慢 | 图捕获、编译与模型预热 | 先预热，再记录正式性能数据 |
| 启动时显存不足 | 设备选择冲突、上下文/并发过大 | 查 `npu-smi`、容器可见设备和 KV Cache 日志 |

## 写在最后

Qwen 这次确实比 DeepSeek 好部署不少：模型只有约 30GiB，两张 64GB 昇腾卡就能按官方思路跑起 128K 服务，文本、图片、思考和工具调用也都能通过同一个 OpenAI-compatible 接口完成。

但“容易”不代表可以跳过验证。真正让服务可用的，不只是容器健康，而是下面这条完整链路都要通：

```text
权重完整
  → 版本和启动参数匹配
  → 文本请求成功
  → reasoning 字段正确
  → tool_calls 能被 Agent 执行
  → 图片能到达模型
  → Nginx 路径和鉴权正确
  → 客户端声明 vision 能力
```

如果你也准备在一台多卡服务器上同时跑多个模型，我会更推荐这种思路：先按通信拓扑划分设备组，再让每个服务只看见自己的卡。DeepSeek 使用完整的 8 卡组，Qwen 这种 27B W8A8 模型用两张卡独立运行，通常比把所有模型和所有卡揉进一个复杂并行方案更容易维护、观察和回滚。

最后再提醒一次：本文参数是本次环境的已验证基线，不是所有昇腾服务器的万能最优值。特别是设备型号、驱动/CANN、镜像版本、上下文长度和并发目标变化后，都应该重新做容量和性能验证。

