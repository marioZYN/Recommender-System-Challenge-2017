from algorithms.Recommender import Recommender

class SLIM(Recommender):

    def __init__(self):

        super(SLIM, self).__init__()

    def fit(self):

        pass

    def get_user_history(self, user_index):

        return self.urm[user_index]
