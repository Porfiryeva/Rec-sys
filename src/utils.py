import pandas as pd


# параметры, которые принимает! продумать и дописать
def prefilter_items(data, item_features, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    tmp_data = data.copy()
    popularity = tmp_data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share_unique_users'] = popularity['share_unique_users'] / tmp_data['user_id'].nunique()

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    tmp_data = tmp_data.loc[~tmp_data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    tmp_data = tmp_data.loc[~tmp_data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекомендаций категории (department)

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    # Уберем слишком дорогие товары

    # ...

    # пользователи, купившие менее 5 уникальных товаров
    active = data.groupby('user_id')['item_id'].nunique().reset_index().rename(columns={'item_id': 'n_unique'})
    inactive_id = active.loc[active['n_unique'] < 5].user_id.tolist()
    tmp_data = tmp_data.loc[~tmp_data['item_id'].isin(inactive_id)]

    # оставляем только top-5000
    popularity = tmp_data.groupby('item_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'n_sold'})
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    tmp_data.loc[~tmp_data['item_id'].isin(top), 'item_id'] = 999999

    return tmp_data
    

def postfilter_items(user_id, recommednations):
    pass
