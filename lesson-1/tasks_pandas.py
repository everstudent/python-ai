import pandas as pd
import numpy as np


# 1. task
authors = pd.DataFrame({
    'author_id':    [1, 2, 3],
    'author_name':  ['Тургенев', 'Чехов', 'Островский']
})

books = pd.DataFrame({
    'author_id':    [1, 1, 1, 2, 2, 3, 3],
    'book_title':   ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    'price':        [450, 300, 350, 500, 450, 370, 290]
})


print(authors)
print(books)


# 2. task
authors_price = pd.merge(authors, books, on='author_id', how='inner')
print(authors_price)


# 3. task
top = authors_price.sort_values(by='price', inplace=False, ascending=False)
top5 = top[0:5]

print(top5)


# 4. task
authors_stat = authors_price.groupby("author_name")
print(authors_stat.agg(mean_price=('price', np.mean), max_price=('price', np.max), min_price=('price', np.min)))


# 5. task
authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
print(authors_price)

book_info = pd.pivot_table(authors_price, values='price', index='author_name', columns=['cover'], aggfunc=sum, fill_value=0)
print(book_info)

book_info.to_pickle("book_info.pkl")
book_info2 = pd.read_pickle('book_info.pkl')

print(book_info2.equals(book_info))
