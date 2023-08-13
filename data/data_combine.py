# This file is used to combine all the books into one file
for i in range(1,2):
    with open(f'book{i}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
# 去除空格和标点符号
        text.strip()
        text = text.replace(' ', '')
        text = text.replace('  ','')
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = text.replace('~', '')
        text = text.replace('·', '')
        text = text.replace('，', '')
        text = text.replace('。', '')
        text = text.replace('？', '')
        text = text.replace('！', '')
        text = text.replace('；', '')
        text = text.replace('：', '')
        text = text.replace('“', '')
        text = text.replace('”', '')
        text = text.replace('.', '')
        text = text.replace('…', '')
        text = text.replace('\\', '')
        text = text.replace('/', '')
        text = text.replace('《', '')
        text = text.replace('》', '')
        text = text.replace('【', '')
        text = text.replace('】', '')
        text = text.replace('（', '')
        text = text.replace('）', '')
        text = text.replace('1', '')
        text = text.replace('2', '')
        text = text.replace('3', '')
        text = text.replace('4', '')
        text = text.replace('5', '')
        text = text.replace('6', '')
        text = text.replace('7', '')
        text = text.replace('8', '')
        text = text.replace('9', '')
        text = text.replace('0', '')
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = text.replace('\t', '')
        text = text.replace(' ', '')
        text = ' '.join(text.split())

    with open('../Run6/book.txt', 'a', encoding='utf-8') as file:
        file.write(text)
    print(i)

print('done')



