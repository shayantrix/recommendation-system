import torch
import pandas as pd
from fastai.collab import *
from fastai.tabular.all import *
import gradio as gr

books = pd.read_csv(r'data\BookRating\Books.csv\Books.csv', delimiter=',', encoding='latin-1',
                    usecols=(0,1), names=('ISBN', 'bookTitle'), header=None)

users = pd.read_csv(r'data\BookRating\Users.csv\Users.csv', delimiter=',', encoding='latin-1', usecols=(0,))
users.columns = ['userID']

ratings = pd.read_csv(r'data\BookRating\Ratings.csv\Ratings.csv', delimiter=',', encoding='latin-1',
                      usecols=(0,1,2), names=('userID', 'ISBN', 'bookRating'), header=None)

# Remove the first row of data in ratings as it is junk data
ratings = ratings.iloc[1:]
ratings['bookRating'] = pd.to_numeric(ratings['bookRating'])

ratings = ratings.merge(books)

dls = CollabDataLoaders.from_df(ratings, user_name='userID', item_name='bookTitle', rating_name='bookRating', bs=64)

n_users = len(dls.classes['userID'])
n_books = len(dls.classes['bookTitle'])
n_factors = 5

users_factors = torch.randn(n_users, n_factors)
books_factors = torch.randn(n_books, n_factors)

class DotProduct(Module):
  def __init__(self, n_users, n_books, n_factors, y_range=(0, 10.1)):
    self.users_factors = Embedding(n_users, n_factors)
    self.books_factors = Embedding(n_books, n_factors)
    self.users_bias = Embedding(n_users, 1)
    self.books_bias = Embedding(n_books, 1)
    self.y_range = y_range

  def forward(self, x):
    users = self.users_factors(x[:, 0])
    books = self.books_factors(x[:, 1])
    res = ((users * books).sum(dim=1, keepdim=True))
    res += self.users_bias(x[:, 0]) + self.books_bias(x[:, 1])
    return sigmoid_range(res, *self.y_range)


# Weights
rating_counts = Counter(ratings['bookRating'])

# Give them weights based on their appearence count in the dataset
total = sum(rating_counts.values())
weights = {k: total / (v * len(rating_counts)) for k, v in rating_counts.items()}

weight_tensor = torch.tensor([weights[i] for i in range(11)], dtype=torch.float32)

weight_tensor /= weight_tensor.sum()  # Normalize to sum to 1

# Loss function
class WeightedMSELoss:
    def __init__(self, weights):
        # defining weights for each rating number
        # weights: tensor of shape (n_ratings,)
        self.weights = weights
    def __call__(self, input, target):
        # input: model predictions (batch_size, 1)
        # target: true ratings (batch_size)
        device = target.device
        weights = self.weights.to(device)
        # I had a problem with device as I am training on Cuda and there was a conflict in the process
        target_long = target.long().squeeze()
        diff = (input.squeeze() - target.squeeze()) ** 2

        # Apply weights based on the portion of ratings
        batch_weights = weights[target_long]
        weighted_diff = diff * batch_weights

        # return the weighted mean squared of differences
        return weighted_diff.mean()

learn = load_learner('data/recommendation_model.pkl')


def recommend_books(bookTitle):
    book_factors = learn.model.i_weight.weight
    idx = dls.classes['bookTitle'].o2i[bookTitle]
    distances = nn.CosineSimilarity(dim=1)(book_factors, book_factors[idx][None])
    idx = distances.argsort(descending=True)[1:6]
    return dls.classes['bookTitle'][idx]

iface = gr.Interface(
    fn=recommend_books,
    inputs=gr.Textbox(label="Book Title"),
    outputs="text",
    title="Book Recommendation System",
    description="Enter a book title to receive book recommendations."
)
iface.launch()