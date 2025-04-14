import json
import time
import psutil
import GPUtil
import logging
import requests
from colorlog import ColoredFormatter

# 自定义的WebhookHandler (用于更新到飞书上远程监控)
class WebhookHandler(logging.Handler):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url

    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'text': log_entry,
        }
        try:
            requests.post(self.webhook_url, json.dumps(payload), headers={'Content-Type': 'application/json'})
        except:
            pass
# 创建logger
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.INFO)  # 设置日志级别

# 创建Webhook处理器
# webhook_url = "https://www.feishu.cn/flow/api/trigger-webhook/2cb53580f503dd289012deea0a6abe5c"
webhook_url = "请输入线上机器人的地址"
webhook_handler = WebhookHandler(webhook_url)
webhook_handler.setLevel(logging.INFO)  # 设置Webhook处理器的日志级别

# 设置不带颜色的日志格式
webhook_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置带颜色的日志格式
stream_formatter = ColoredFormatter(
                        # "%(log_color)s%(asctime)s - %(name)s - %(levelname)-8s%(reset)s %(white)s%(message)s",
                        "%(blue)s%(asctime)s - %(name)s - %(reset)s%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
                        datefmt=None,
                        reset=True,
                        log_colors={
                            'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'purple',
                        },
                        secondary_log_colors={},
                        style='%'
                    )

# 为Webhook处理器设置格式
webhook_handler.setFormatter(webhook_formatter)

# 将Webhook处理器添加到logger
logger.addHandler(webhook_handler)

# 创建StreamHandler以在终端输出日志
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # 设置StreamHandler的日志级别
stream_handler.setFormatter(stream_formatter)  # 为StreamHandler设置相同的格式

# 将StreamHandler添加到logger
logger.addHandler(stream_handler)

logger = logger

# 资源监控用代码
def monitor_memory(interval=120):
    while True:
        print_text = ''
        # CPU 
        cpu_usage = psutil.cpu_percent(interval=1)
        print_text += "-" * 80 + '\n'
        print_text += f"CPU Usage: {cpu_usage}%" + '\n'
        print_text += "-" * 40 + '\n'

        # 内存使用情况
        memory = psutil.virtual_memory()
        print_text += f"Total Memory: {memory.total / (1024**3):.2f} GB | Available Memory: {memory.available / (1024**3):.2f} GB | Memory Usage: {memory.percent}%"+ '\n'
        print_text += "-" * 40 + '\n'

        # 获取所有可用的GPU
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print_text += f"GPU ID: {gpu.id}, Name: {gpu.name}" + '\n'
            print_text += f"Load: {gpu.load*100}%" + '\n'
            print_text += f"Free Memory: {gpu.memoryFree}MB" + '\n'
            print_text += f"Used Memory: {gpu.memoryUsed}MB" + '\n'
            print_text += f"Total Memory: {gpu.memoryTotal}MB" + '\n'
            print_text += f"Temperature: {gpu.temperature} °C" + '\n'
            print_text += "-" * 40 + '\n'

        logger.info(print_text)
        time.sleep(interval)
