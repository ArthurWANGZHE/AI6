import requests
import re
from bs4 import BeautifulSoup
import random
import time
# 小说网站
url = 'http://www.liulangcat.com/author.php?author=901'
# 伪装成浏览器
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
targe_ul = soup.find('ul',class_='item-bottom')
a_elements = targe_ul.find_all('a')
for a_element in a_elements:
    url = a_element.get('href')
    urls.append(url)
# 获取完整的小说网址
urls_ = ['https://www.kepub.net/book/'+url for url in urls]