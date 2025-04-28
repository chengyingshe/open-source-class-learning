import os

def set_proxy(proxy_on=True):
    if proxy_on:
        os.environ['http_proxy'] = 'http://127.0.0.1:7897'
        os.environ['https_proxy'] = 'http://127.0.0.1:7897'
    else:
        keys = os.environ.keys()
        if 'http_proxy' in keys:
            os.environ.pop('http_proxy')
        if 'https_proxy' in keys:
            os.environ.pop('https_proxy')
