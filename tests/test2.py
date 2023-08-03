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
url = 'https://www.qidian.com/book/1037342218/'
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
# 获取章节总数
target_div = soup.find('div', class_='catalog-volume')
h3_elements = target_div.find('h3')
text=h3_elements.get_text()
chapter_count = int(re.findall(r'\d+',text)[0])
# 获取章节网址
chapter_urls = []
class_name = "chapter-item"
for i in range(1, chapter_count+1):
    li_selector = f'li[data-rid="{i}"]'
    li_elements = soup.select(f'li.{class_name}[data-rid="{i}"]')
    for li_element in li_elements:
        a_elements = li_element.find('a')
        url = a_elements.get('href')
        chapter_urls.append(url)

# print(chapter_urls)
# 章节urls补充http
chapter_urls = ['https:'+chapter_url for chapter_url in chapter_urls]
# 获取每章节的内容
chapter_contents = []
for chapter_url in chapter_urls:
    response = requests.get(chapter_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    target_div = soup.find('div', class_='relative')
    p_elements = target_div.find_all('p')
    for p_element in p_elements:
        chapter_contents.append(p_element.get_text())
    time.sleep(random.randint(1, 3))
    print("爬完第{}章".format(chapter_urls.index(chapter_url)+1))

# 写入txt
with open('test.txt', 'w', encoding='utf-8') as f:
    for chapter_content in chapter_contents:
        f.write(chapter_content)
        f.write('\n')
    f.close()