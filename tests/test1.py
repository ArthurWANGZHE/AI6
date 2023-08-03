import requests
import re
from bs4 import BeautifulSoup
from requests.exceptions import *
import random
import json
import time
import os
import sys


# 起点小说网站
url = 'https://www.qidian.com/free/all/'
# 伪装成edge浏览器
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.64'
}
# 发送请求
response = requests.get(url, headers=headers)
# 获取网页源码
html = response.text
# 使用BeautifulSoup解析网页源码
soup = BeautifulSoup(html, 'lxml')
# print(soup)
# 获取所有的小说网址
urls = []
for i in range(1,20):
    li_selector = f'li[data-rid="{i}"]'
    li_elements = soup.select(li_selector)
    for li_element in li_elements:
        target_divs = li_element.select('div',class_='book-mid-info')
        for target_div in target_divs:
            a_elements = target_div.find_all('a')
            for a_element in a_elements:
                url = a_element.get('href')
                urls.append(url)

# print(urls)

