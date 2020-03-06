# x, y 数据的样本， d指向的哪一列（对于哪个特征进行划分呢）， v是基于哪个值来进行划分（哪个候选点来进行划分）
def split(x, y, d, v):
    index_l = (x[:, d] <= v)
    index_r = (x[:, d] > v)
    # 使用fancy indexing来进行取值
    return x[index_l], x[index_r], y[index_l], y[index_r]


def try_split(x, y):
    best_ent = float('inf')  # 定义我们最好的值是无穷大
    best_d, best_v = -1, -1  # 看我们最终要基于哪一列进行划分，且它对应信息熵最小的划分value是多少
    for d in range(x.shape[1]):  # 基于我们的列数进行探索
        sorted_index = np.argsort(x[:, d]) # 对我们的第d列进行下标的排序
        for i in range(len(x) -1): # 对第d列的每行数据进行处理
            v = (x[sorted_index[i], d] + x[sorted_index[i+1], d]) / 2 # 取平均值
            x_l, x_r, y_l, y_r = split(x, y, d, v)
            ent = Ent(y_l) + Ent(y_r)
            if ent < best_ent: # 进行迭代处理
                best_ent, best_d, best_v = ent, d, v
    return best_ent, best_d, best_v