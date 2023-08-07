import requests
import re
from bs4 import BeautifulSoup
import random
import time


# 起点小说网站
url__ = ['https://www.qidian.com/free/all/','https://www.qidian.com/free/all/page2/',
       'https://www.qidian.com/free/all/page3/',
       'https://www.qidian.com/free/all/page4/',
       'https://www.qidian.com/free/all/page5/']
# headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.64'
}
# 发送请求
urls = []
for url in url__:
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
T = len(urls)
print(T)
# 获取完整的小说网址
urls_ = ['https:'+url for url in urls]
# 分别爬取每本小说的章节
a_=1
e=0
for url in urls_:
    response = requests.get(url, headers=headers)
    html = response.text
    # 使用BeautifulSoup解析网页源码
    soup = BeautifulSoup(html, 'lxml')
    # print(soup)
    # 获取章节总数
    try:
        h3_elements = soup.find('h3',class_='volume-name')
        text = h3_elements.get_text()
        chapter_count = int(re.findall(r'\d+', text)[0])
        chapter_urls = []
        class_name = "chapter-item"
        # 获取章节网址
        for i in range(1, chapter_count + 1):
            li_selector = f'li[data-rid="{i}"]'
            li_elements = soup.select(f'li.{class_name}[data-rid="{i}"]')
            for li_element in li_elements:
                a_elements = li_element.find('a')
                url = a_elements.get('href')
                chapter_urls.append(url)
        chapter_urls = ['https:' + chapter_url for chapter_url in chapter_urls]
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
            print("正在爬第{}本".format(a_)+"爬完第{}章".format(chapter_urls.index(chapter_url) + 1)+
                  "一共有{}章".format(len(chapter_urls))+ "一共有{}本".format(T))
        # 写入txt
        with open('book{}.txt'.format(a_), 'w', encoding='utf-8') as f:
            for chapter_content in chapter_contents:
                f.write(chapter_content)
                f.write('\n')
            f.close()
        a_ = a_+1
    except:
        e=e+1
        continue

print("一共有{}本".format(T)+"一共有{}本爬取失败".format(e))
