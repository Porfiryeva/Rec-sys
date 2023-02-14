import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    # data здесь - исходная таблица с данными (data_train)
    def __init__(self, data, weighting='tfidf'):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        assert weighting in ['tfidf', 'bm25', 'no'], 'Неверно указан метод'

        self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix)
        elif weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix)
        elif weighting != 'no':
            print('Error')

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data, values='quantity'):  # выбор столбца для значений

        # your_code
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values=values,
                                          aggfunc='count',
                                          fill_value=0
                                          )

        return user_item_matrix.astype(float)

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=2, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.02, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix.tocsr()))

        return model

    def get_recommendations(self, user, N=5):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=self.user_item_matrix.tocsr()[self.userid_to_id[user]],
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],
                                    recalculate_user=True)[0]]
        return res

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # принимает 1 пользователя
        popularity = (self.data.loc[self.data['user_id'] == user]
                      .groupby(['user_id', 'item_id'])['quantity']
                      .count()
                      .reset_index()
                      .sort_values('quantity', ascending=False)
                      )
        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.head(5).item_id.to_list()
        # если куплено менее 5 товаров
        # для самого частого рекомендуем столько, чтобы добрать до 5
        # чуть лучше метрика
        if len(popularity) < N:
            k = N - len(popularity)
            #             print(user)
            #             print(len(popularity))
            res = [self.id_to_itemid[rec]
                   for rec in self.model.similar_items(self.itemid_to_id[popularity[0]], N=2 + k)[0][1:]]
            res += [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[top_item], N=2)[0][1]]
                    for top_item in popularity[1:]]
        else:
            res = [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[top_item], N=2)[0][1]]
                   for top_item in popularity]

        # или просто отфильтровать пользователей с менее 5 уникальных покупок
#         res = [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[top_item], N=2)[0][1]]
#                        for top_item in popularity]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        # товары отсортированы по quantity для исходного товара
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        # упорядочиваем рекомендации, тк own_recommender выдаёт неупорядоченные
#         recs = self.own_recommender.recommend(userid=self.userid_to_id[user],
#                                               user_items=self.user_item_matrix.tocsr()[self.userid_to_id[user]],
#                                               N=N,
#                                               filter_already_liked_items=True,
#                                               filter_items=[self.itemid_to_id[999999]],
#                                               recalculate_user=True)
#         mask = recs[1].argsort()[::-1]  # получим индекс для сортировки
#         res = [self.id_to_itemid[rec] for rec in recs[0][mask]]
        #
        # на самом деле, возвращает упорядоченные в версии 0.6.2
        res = [self.id_to_itemid[rec]
               for rec in self.own_recommender.recommend(userid=self.userid_to_id[user],
                                                         user_items=self.user_item_matrix.tocsr()[
                                                             self.userid_to_id[user]],
                                                         N=N - 1,  # почему-то даёт рекомендаций на 1 больше
                                                         filter_already_liked_items=False,
                                                         filter_items=[self.itemid_to_id[999999]],
                                                         recalculate_user=True)[0]]

        if len(res) < N:
            print(user)
            print(len(res))
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res