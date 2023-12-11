import logging
import functools

def run_once(func):
    result = []
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not result:
            result.append(func(*args, **kwargs))
        return result[0]
    return wrapper
    
@run_once
def get_logger():
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('qllm')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger
