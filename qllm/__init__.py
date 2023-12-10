import os
if os.getenv("PROXY_PORT", None):
    proxy_port = os.getenv("PROXY_PORT", None)
    os.environ["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["HTTPS_PROXY"] = f"http://127.0.0.1:{proxy_port}"

# TODO To work around autogptq/vLLM
os.environ['compatible_with_autogptq'] = '1'
__version__ = '0.1.3'
