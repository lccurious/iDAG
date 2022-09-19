from typing import List
import os
import datetime
import traceback
import functools
import json
import socket
import requests
import time
import hmac
import hashlib
import base64
import urllib

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def dingtalk_sender(webhook_url: str,
                    user_mentions: List[str] = [],
                    secret: str = '',
                    proxy_addr: str = None,
                    keywords: List[str] = []):
    """
    DingTalk sender wrapper: execute func, send a DingTalk notification with the end status
    (sucessfully finished or crashed) at the end. Also send a DingTalk notification before
    executing func.
    `webhook_url`: str
        The webhook URL to access your DingTalk chatroom.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `user_mentions`: List[str] (default=[])
        Optional users phone number to notify.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `secret`: str (default='')
        DingTalk chatroom robot are set with at least one of those three security methods
        (ip / keyword / secret), the chatroom will only accect messages that:
            are from authorized ips set by user (ip),
            contain any keyword set by user (keyword),
            are posted through a encrypting way (secret).
        Vist https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq from more details.
    `keywords`: List[str] (default=[])
        see `secret`
    """
    msg_template_md = {
        "msgtype": "markdown",
        "markdown": {
            "title": "# Multisource: Art -> Clipart",
            "text": "",
        },
        "at": {
            "atMobiles": user_mentions,
            "isAtAll": False
        }
    }
    timeout_thresh = 5

    def _construct_encrypted_url():
        '''
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for details
        '''
        timestamp = round(datetime.datetime.now().timestamp() * 1000)
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        encrypted_url = webhook_url + '&timestamp={}'.format(timestamp) \
                        + '&sign={}'.format(sign)
        return encrypted_url

    def robust_post(url, msg):
        if proxy_addr is not None:
            proxies = {
                "http": f"{proxy_addr}",
                "https": f"{proxy_addr}"
            }
        try:
            requests.post(url, json=msg, timeout=timeout_thresh)
        except requests.exceptions.Timeout:
            requests.post(url, json=msg, timeout=timeout_thresh, proxies=proxies)
        except Exception as e:
            print("Post Failed {}: ".format(e), msg)

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True

            if master_process:
                contents = ['# Your training has started 🎬\n',
                            '- Machine name: %s' % host_name,
                            '- Main call: %s' % func_name,
                            '- Starting date: %s' % start_time.strftime(DATE_FORMAT)]
                contents.extend(['@{}'.format(i) for i in user_mentions])
                if len(keywords):
                    contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                msg_template_md['markdown']['title'] = 'Your training has started 🎬'
                msg_template_md['markdown']['text'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    robust_post(postto, msg=msg_template_md)
                else:
                    robust_post(webhook_url, msg=msg_template_md)

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ['# Your training is complete 🎉\n',
                                ' - Machine name: %s' % host_name,
                                ' - Main call: %s' % func_name,
                                ' - Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                ' - End date: %s' % end_time.strftime(DATE_FORMAT),
                                ' - Training duration: %s' % str(elapsed_time)]

                    try:
                        task_name = value['task']
                        # task_name = task_name.replace('/', '+')
                        # task_name = task_name.replace('->', '➡︎')
                        contents.append('\nMain call returned value:\n')
                        contents.append('## {}\n'.format(task_name))
                    except:
                        contents.append('Main call returned value: %s' % "ERROR - Couldn't str the returned value.")

                    contents.extend(['@{}'.format(i) for i in user_mentions])
                    if len(keywords):
                        contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                    msg_template_md['markdown']['title'] = 'Your training is complete 🎉'
                    msg_template_md['markdown']['text'] = '\n'.join(contents)
                    msg_template_md['markdown']['text'] += '\n\n' + value['content']
                    if secret:
                        postto = _construct_encrypted_url()
                        robust_post(postto, msg=msg_template_md)
                    else:
                        robust_post(webhook_url, msg=msg_template_md)
                        print(msg_template_md)

                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["# Your training has crashed ☠️\n",
                            '- Machine name: %s' % host_name,
                            '- Main call: %s' % func_name,
                            '- Starting date: %s' % start_time.strftime(DATE_FORMAT),
                            '- Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            '- Crashed training duration: %s\n\n' % str(elapsed_time),
                            "## Here's the error:\n",
                            '%s\n\n' % ex,
                            "> Traceback:",
                            '> %s' % traceback.format_exc()]
                contents.extend(['@{}'.format(i) for i in user_mentions])
                if len(keywords):
                    contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                msg_template_md['markdown']['text'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    robust_post(postto, msg=msg_template_md)
                else:
                    robust_post(webhook_url, msg=msg_template_md)
                    print(msg_template_md)

                raise ex

        return wrapper_sender

    return decorator_sender


if __name__ == '__main__':
    from prettytable import PrettyTable

    def to_markdown_table(pt):
        """
        Print a pretty table as a markdown table
        
        :param py:obj:`prettytable.PrettyTable` pt: a pretty table object.  Any customization
        beyond int and float style may have unexpected effects
        
        :rtype: str
        :returns: A string that adheres to git markdown table rules
        """
        _junc = pt.junction_char
        if _junc != "|":
            pt.junction_char = "|"
        markdown = [row[1:-1] for row in pt.get_string().split("\n")[1:-1]]
        pt.junction_char = _junc
        return "\n".join(markdown)

    @dingtalk_sender(webhook_url="https://oapi.dingtalk.com/robot/send?access_token=750084d30bb8d2d7b9fad6232266880d4481b9d18e0dfcf3a60075067945f53e", secret="SEC447e0856a08b1582ae88034b1adff53de3867b4448ddddb1ba3b98dbf2ff2beb", keywords=["test"])
    def pretty_table():
        x = PrettyTable()
        x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
        x.add_row(["Adelaide", 1295, 1158259, 600.5])
        x.add_row(["Brisbane", 5905, 1857594, 1146.4])
        x.add_row(["Darwin", 112, 120900, 1714.7])
        x.add_row(["Hobart", 1357, 205556, 619.5])
        x.add_row(["Sydney", 2058, 4336374, 1214.8])
        x.add_row(["Melbourne", 1566, 3806092, 646.9])
        x.add_row(["Perth", 5386, 1554769, 869.4])
        return {'task': 'pretty_table_test', 'content': to_markdown_table(x)}
    
    pretty_table()
