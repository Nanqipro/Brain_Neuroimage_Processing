import numpy as np
import argparse
import os
import time
import datetime
from TraceContrast_model import TraceContrast
import datautils
from utils import init_dl_program, name_with_datetime
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy.io as scio


# 导出点云数据为PLY格式
def export_ply_with_label(out, points, colors):
    """
    将点云数据及其颜色标签导出为PLY格式文件
    
    参数:
    -------
    out : str
        输出PLY文件路径
    points : np.ndarray
        点云坐标数据，形状为(N, 3)
    colors : np.ndarray
        点云颜色数据，形状为(N, 3)，值范围为[0,1]
    """
    with open(out, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex ' + str(points.shape[0]) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            cur_color = colors[i, :]
            f.write('%f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2],
                int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)
            ))


# 创建定期保存模型检查点的回调函数
def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    """
    返回一个回调函数，用于定期保存模型检查点
    
    参数:
    -------
    save_every : int
        保存频率
    unit : str
        频率单位，'epoch'或'iter'
        
    返回:
    -------
    function
        回调函数
    """
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


if __name__ == '__main__':

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='用于保存模型、输出和评估指标的文件夹名称')
    parser.add_argument('scn_data_path', help='用于加载输入数据的文件路径')
    parser.add_argument('task', help='用于datautils的任务名称')

    # 可选参数
    parser.add_argument('--gpu', type=int, default=0, help='用于训练和推理的GPU编号（默认为0）')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小（默认为64）')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率（默认为0.001）')
    parser.add_argument('--repr-dims', type=int, default=16, help='表征维度（默认为16）')
    parser.add_argument('--max-train-length', type=int, default=10000, help='对于长度大于<max_train_length>的序列，将被裁剪成多个长度小于<max_train_length>的序列（默认为10000）')
    parser.add_argument('--iters', type=int, default=None, help='迭代次数')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--save-every', type=int, default=None, help='每<save_every>次迭代/轮次保存一次检查点')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--max-threads', type=int, default=None, help='此进程允许使用的最大线程数')
    parser.add_argument('--irregular', type=float, default=0, help='缺失观测的比例（默认为0）')
    args = parser.parse_args()
    
    print("Arguments:", str(args))
    
    # 初始化深度学习环境
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')

    # 加载SCN数据
    train_data, poi = datautils.load_SCN(args.scn_data_path, args.task)

    # 设置模型配置
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    # 如果指定了保存频率，添加回调函数
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    # 创建运行目录
    run_dir = 'training/' + args.task + '_' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 记录开始时间
    t = time.time()
    
    # 创建并训练模型
    model = TraceContrast(
        input_dims=train_data.shape[-1],  # 输入维度为神经元数
        device=device,
        **config
    )
    # 训练模型
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )

    # 保存训练好的模型
    model.save(f'{run_dir}/model.pkl')

    # 计算并输出训练时间
    t = time.time() - t
    print(f"\n训练时间: {datetime.timedelta(seconds=t)}\n")

    # 使用模型编码器获取神经元活动的嵌入表征
    train_repr = model.encode(train_data)

    # 对嵌入进行归一化处理
    from sklearn.preprocessing import normalize
    embeddings = np.reshape(train_repr, (train_repr.shape[0],train_repr.shape[1]*train_repr.shape[2]))
    embeddings = normalize(embeddings)

    num_nodes = train_data.shape[0]

    # 保存嵌入结果
    scio.savemat(f'./{run_dir}/embedding.mat', {'emb':embeddings})

    # 使用t-SNE进行降维可视化
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(embeddings)
    # 对t-SNE结果进行归一化，范围缩放到[0,1]
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    # 保存t-SNE结果
    scio.savemat(f'./{run_dir}/tsne.mat', {'tsne':x_norm})

    # 进行无监督聚类，尝试1-5个簇的聚类效果
    for num_classes in range(1,6):
        if num_classes == 1:
            # 如果只有1个簇，所有样本归为同一类
            y = np.zeros((embeddings.shape[0],),dtype=int)
        else:
            # 使用K-means聚类
            kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
            y = kmeans.labels_
        # 保存聚类结果
        scio.savemat(f'./{run_dir}/class_order_{num_classes}.mat', {'order':y})

        # 计算每个类别在t-SNE第一维上的平均位置
        class_mean = np.zeros((num_classes,1))
        for k in range(num_classes):
            class_order = np.where(y==k)
            tmp_pos = x_norm[class_order,0]
            class_mean[k] = np.mean(tmp_pos)
        # 根据位置对类别进行排序
        new_mean = np.argsort(class_mean, axis=0)

        # 绘制t-SNE可视化图
        plt.figure(figsize=(6, 6))
        for i in range(x_norm.shape[0]):
            for j in range(num_classes):
                if new_mean[j]==y[i]:
                    color_order = j
            plt.scatter(
                x_norm[i, 0], x_norm[i, 1], marker='.', color=plt.cm.Set1(color_order)
            )
        # 设置标题
        if num_classes==1:
            plt.title(f'{num_classes} Cluster', size = 20, fontweight='bold')
        else:
            plt.title(f'{num_classes} Clusters', size = 20, fontweight='bold')
        plt.yticks(fontproperties = 'Arial', size = 20, fontweight='bold')
        plt.xticks(fontproperties = 'Arial', size = 20, fontweight='bold')
        # 保存图像
        plt.savefig(f'./{run_dir}/tsne_{num_classes}.eps', dpi=400)
        plt.close()

        # 生成点云数据
        points = poi.cpu().numpy()  # 使用神经元的三维空间坐标
        colors = np.zeros_like(points)
        # 为不同类别分配不同颜色
        for i in range(colors.shape[0]):
            # 根据聚类标签分配颜色
            if y[i] < 9:
                colors[i] = np.array(plt.cm.Set1(y[i])[:3])
            elif y[i] >= 9 and y[i] < 17:
                colors[i] = np.array(plt.cm.Set2(y[i]-9)[:3])
            else:
                colors[i] = np.array(plt.cm.Set3(y[i]-17)[:3])

        # 导出点云数据为PLY格式
        export_ply_with_label(f'./{run_dir}/{num_classes}_clusters.ply', points, colors)


    print("完成。")
