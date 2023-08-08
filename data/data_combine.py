# This file is used to combine all the books into one file
for i in range(1,39):
    with open(f'book{i}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
# 去除空格和标点符号
        text = text.replace(' ', '')
        text = text.replace('\n', '')
        text = text.replace('\r', '')
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
        text = text.replace('《', '')
        text = text.replace('》', '')
        text = text.replace('【', '')
        text = text.replace('】', '')

    with open('../RUN/book0.txt', 'a', encoding='utf-8') as file:
        file.write(text)
    print(i)

print('done')



