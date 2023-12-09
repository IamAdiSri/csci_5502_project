import os
import pandas as pd
import numpy as np
import random

class Dataset:
    def __init__(self, dir):
        self.dataframe = self.load_dataframe(dir)

        self.user_item = {}
        self.item_user = {}

        self.user_count = 0
        self.item_count = 0

        self.train = []
        self.test = {}

        self.train_size = 0
        self.test_size = 0
        self.train_idx = None

    def load_dataframe(self, dir):
        data = pd.read_csv(os.path.join(dir, "u.data"), delimiter="\t", header=None)
        data.columns = ["user", "item", "score", "timestamp"]

        item = pd.read_csv(os.path.join(dir, "u.item"), delimiter="|", encoding = "ISO-8859-1", header=None)
        item.columns = ['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

        user = pd.read_csv(os.path.join(dir, "u.user"), delimiter="|", header=None)
        user.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

        data = data.join(user.set_index('user id'), on='user').join(item.set_index('movie id'), on='item')
        return(data)
    
    def gen_adjacency(self):
        for u, i in self.dataframe[['user', 'item']].to_numpy():
            # user and item IDs are 1-indexed but we want zero indexed
            u -= 1
            i -= 1

            self.user_item[u] = self.user_item.get(u, set())
            self.user_item[u].add(i)

            self.item_user[i] = self.item_user.get(i, set())
            self.item_user[i].add(u)

        self.user_count = len(self.user_item)
        self.item_count = len(self.item_user)

    def sample_positive(self, userid):
        return random.choice(tuple(self.user_item[userid]))

    def sample_negative(self, userid):
        i = random.choice(tuple(self.item_user.keys()))
        while i in self.user_item[userid]:
            i = random.choice(tuple(self.item_user.keys()))
        return i
        
    def make_train_test(self, neg_count=4):
        for u in self.user_item:
            leave_out = self.sample_positive(u)
            for i in self.user_item[u]:
                if i != leave_out:
                    self.train.append((u, i, 1))
            
            for i in self.user_item[u]:
                for _ in range(neg_count):
                    self.train.append((u, self.sample_negative(u), 0))

            self.test[u] = [leave_out, [self.sample_negative(u) for _ in range(100)]]
        self.train = np.array(self.train)

        self.train_size = len(self.train)
        self.test_size = len(self.test)
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)

    def train_generator(self, batch_size=32):
        for idx in range(0, self.train_size, batch_size):
            yield(self.train[self.train_idx[idx: idx + batch_size]])

    def test_generator(self):
        for u, [pi, nis] in self.test.items():
            yield u, pi, nis