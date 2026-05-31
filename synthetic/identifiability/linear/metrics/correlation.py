import numpy as np
import scipy as sp
import scipy.stats as stats
from .munkres import Munkres
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]


    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    X_train, X_test, y_train, y_test = train_test_split(x_sort.T, y.T,random_state=1, train_size=0.5)
    X_pred = np.zeros(y_test.shape)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    regr_list = []
    for i in range(X_train.shape[1]):
        X_train_list.append(X_train[:,i])
        y_train_list.append(y_train[:,i])
        X_test_list.append(X_test[:,i])
        regr_tmp = MLPRegressor(random_state=1, max_iter=500).fit(np.reshape(X_train_list[i], (-1,1)), y_train_list[i])
        regr_list.append(regr_tmp)
        X_pred[:,i] = regr_tmp.predict(np.reshape(X_test_list[i], (-1,1)))

    if method=='Pearson':
        corr_sort = np.corrcoef(y_test.T, X_pred.T)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y_test, X_pred)
        corr_sort = corr_sort[0:dim, dim:]
        
    # X_train0 = X_train[:,0]
    # X_train1 = X_train[:,1]
    # y_train0 = y_train[:,0]
    # y_train1 = y_train[:,1]
    # X_test0 = X_test[:,0]
    # X_test1 = X_test[:,1]

    # regr0 = MLPRegressor(random_state=1, max_iter=500).fit(np.reshape(X_train0, (-1,1)), y_train0)
    # regr1 = MLPRegressor(random_state=1, max_iter=500).fit(np.reshape(X_train1, (-1,1)), y_train1)

    # X_pred[:,0] = regr0.predict(np.reshape(X_test0, (-1,1)))
    # X_pred[:,1] = regr1.predict(np.reshape(X_test1, (-1,1)))

    # # Re-calculate correlation --------------------------------
    # if method=='Pearson':
    #     corr_sort = np.corrcoef(y_test.T, X_pred.T)
    #     corr_sort = corr_sort[0:dim,dim:]
    # elif method=='Spearman':
    #     corr_sort, pvalue = stats.spearmanr(y_test, X_pred)
    #     corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

def calculate_mcc(Z_true, Z_pred, correlation_fn="Pearson"):
  """Computes score based on both training and testing codes and factors."""
  mus_train, ys_train = Z_true.T, Z_pred.T
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc