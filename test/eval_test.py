import numpy as np

def test_eval_r1_unit():
    # 构造特征数量和维度
    N = 4   # 样本数
    C = 3   # 特征维度

    # 构造文本与动作特征：让它们一一对应（相似度最大）
    np.random.seed(0)
    text_feat = np.random.randn(N, C)
    text_feat /= np.linalg.norm(text_feat, axis=1, keepdims=True)

    motion_feat = text_feat + np.random.normal(0, 0.01, size=(N, C))  # 加一点噪声
    motion_feat /= np.linalg.norm(motion_feat, axis=1, keepdims=True)

    # 构造 dataset_pair: 每个文本对应第 i 个动作
    dataset_pair = {i: [i] for i in range(N)}

    # 相似度矩阵（文本 → 动作）
    sims_t2m = 100 * text_feat.dot(motion_feat.T)   # 或去掉 100，看单位影响

    ranks = np.zeros(N)
    for index, score in enumerate(sims_t2m):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # 计算 R@1
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    print(f"R@1 = {r1} (unit: percentage)")

    # 测试小数形式
    r1_frac = len(np.where(ranks < 1)[0]) / len(ranks)
    print(f"R@1 = {r1_frac} (unit: fraction)")

test_eval_r1_unit()
