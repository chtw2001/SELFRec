import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import faiss
# paper: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW'21


class NCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['NCL'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-tau'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.hyper_layers = int(args['-hyper_layers'])
        self.alpha = float(args['-alpha'])
        self.proto_reg = float(args['-proto_reg'])
        self.k = int(args['-num_clusters'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = InfoNCE(user_emb[user_idx],user2centroids,self.ssl_temp)
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = InfoNCE(item_emb[item_idx],item2centroids,self.ssl_temp)
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb, context_item_emb = torch.split(context_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb, initial_item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        context_user_emb = context_user_emb[user]
        initial_user_emb = initial_user_emb[user]
        ssl_loss_user = InfoNCE(context_user_emb,initial_user_emb,self.ssl_temp)
        context_item_emb = context_item_emb[item]
        initial_item_emb = initial_item_emb[item]
        ssl_loss_item = InfoNCE(context_item_emb,initial_item_emb,self.ssl_temp)
        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            if epoch >= 20:
                self.e_step()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb, emb_list  = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,user_idx,pos_idx)
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + ssl_loss

                if epoch<20: #warm_up
                    optimizer.zero_grad()
                    warm_up_loss.backward()
                    optimizer.step()
                    if n % 100 == 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item())
                else:
                    # Backward and optimize
                    proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                    batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + ssl_loss + proto_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100==0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'proto_loss', proto_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings, all_embeddings
