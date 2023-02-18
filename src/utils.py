import pandas as pd
import numpy as np


def prefilter_items(data, item_features=None, take_n_popular=5000,):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share_unique_users'] = popularity['share_unique_users'] / data['user_id'].nunique()
    # > 0.2
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data.loc[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)  < 0.02
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data.loc[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    # придётся убрать около млн строк - те чуть меньше половины

    # Уберем неинтересные для рекомендаций категории (department)
    if item_features is not None:
        department_size = (item_features.groupby('department')['item_id']
                                        .nunique()
                                        .sort_values(ascending=False)
                                        .reset_index()
                                        .rename(columns={'item_id': 'n_items'})
                          )
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments =  item_features.loc[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data.loc[~data['item_id'].isin(items_in_rare_departments)]
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # np.maximum сравнивает 2 массива и возвращает максимальное значение в ряду
    # те здесь вернёт quantity > 1 или 1
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 1.2]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 200]
    
    # ...

    # пользователи, купившие менее 5 уникальных товаров
    active = data.groupby('user_id')['item_id'].nunique().reset_index().rename(columns={'item_id': 'n_unique'})
    inactive_id = active.loc[active['n_unique'] < 5].user_id.tolist()
    data = data.loc[~data['item_id'].isin(inactive_id)]

    # оставляем только top-5000
    popularity = data.groupby('item_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'n_sold'})
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data
    

def postfilter_items(user_id, recommednations):
    pass
