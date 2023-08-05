# This file is used to combine all the books into one file
for i in range(1,10):
    with open(f'book{i}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    with open('book.txt', 'a', encoding='utf-8') as file:
        file.write(text)
    print(i)

print('done')



