# nanoBrain API 配置

nanoBrain 的 AutoResearch 候选循环不绑定具体供应商。需要调用第三方
OpenAI-compatible 服务时，在项目根目录填写 `.env`：

```dotenv
NANOBRAIN_API_KEY=填入你的 key
NANOBRAIN_API_BASE_URL=https://你的服务/v1
NANOBRAIN_API_MODEL=你的模型名
```

`.env.example` 是同样的模板。`source env.sh` 后，项目会自动加载 `.env`；
已经在 shell 中导出的变量优先于文件内容。也可以直接检查配置：

```bash
cd /data/zim_rider/brain
source env.sh
python scripts/check_api_config.py
python scripts/check_api_config.py --require-configured
```

检查器只显示 URL、模型名和 `api_key_present`，不会显示 key。训练本身不需要
API；只有需要让外部模型参与候选提议时才实例化
`nanobrain.OpenAICompatibleClient`。客户端调用标准的
`/chat/completions`。`NANOBRAIN_API_MAX_RETRIES=-1` 会对网络错误、限流和
服务端临时错误持续重试；认证错误和请求格式错误仍会立即报告，避免错误配置
无限占用实验进程。默认校验证书；如果供应商的 URL 证书域名不匹配，可明确设置
`NANOBRAIN_API_TLS_VERIFY=0`，并在供应商修复证书后恢复为 `1`。不需要额外安装
厂商 SDK。

常见兼容变量也支持回退：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、
`OPENAI_MODEL`，以及 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、
`DASHSCOPE_MODEL`。nanoBrain 专属变量优先级最高。
