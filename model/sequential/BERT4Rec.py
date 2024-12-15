import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.sampler import next_batch_sequence
from util.loss_torch import l2_reg_loss
from util.structure import PointWiseFeedForward
from math import floor
import random


# Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM'19

class BERT4Rec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERT4Rec, self).__init__(conf, training_set, test_set)
        args =self.config['BERT4Rec']
        block_num = int(args['n_blocks'])
        drop_rate = float(args['drop_rate'])
        head_num = int(args['n_heads'])
        self.aug_rate = float(args['mask_rate']) # 0.5
        self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                # seq -> sequence item list,
                # pos -> seq별 item 개수 만큼 리스트 ex) 5 -> [1, 2, 3, 4, 5], 
                # seq_len -> 배치 사이즈 별로 sequence 개수
                seq, pos, y, neg_idx, seq_len = batch
                aug_seq, masked, labels = self.item_mask_for_bert(seq, seq_len, self.aug_rate, self.data.item_num+1)
                seq_emb = model.forward(aug_seq, pos)
                # item mask
                rec_loss = self.calculate_loss(seq_emb,masked,labels)
                batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'rec_loss:', rec_loss.item())
            model.eval()
            self.fast_evaluation(epoch)

    def item_mask_for_bert(self,seq,seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i]*mask_ratio),1))
            masked[i, to_be_masked] = 1
            labels += list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        # labels -> mask된 원래 seq의 값
        # masked -> 어떤 좌표에 마스크 된건지 나타냄
        # augmented_seq -> mask된 좌표의 값이 item_num+1이 됨
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        seq_emb = seq_emb[masked>0].view(-1, self.emb_size)
        # 유사도/선호도 출력
        logits = torch.mm(seq_emb, self.model.item_emb.t())
        # mask된 값을 예측하고 그 값과의 차이를 loss로 사용
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())/labels.shape[0]
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            for i,length in enumerate(seq_len):
                if length == self.max_len:
                    seq[i,:length-1] = seq[i,1:]
                    pos[i,:length-1] = pos[i,1:]
                    pos[i, length-1] = length
                    seq[i, length-1] = self.data.item_num+1
                else:
                    pos[i, length] = length+1
                    seq[i,length] = self.data.item_num+1
            seq_emb = self.model.forward(seq,pos)
            # 마지막 item에 배열의 크기와 같은 값을 넣고 그 값을 예측하기?
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            score = torch.matmul(torch.cat(last_item_embeddings,0), self.model.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class BERT_Encoder(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate):
        super(BERT_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks # 2
        self.head_num = n_heads # 1
        self.drop_rate = drop_rate # 0.2
        self.max_len = max_len
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num+2, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len+2, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        # 2개의 layer
        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            # new_fwd_layer
            # (torch.nn.Linear(hidden_units, hidden_units),
            #  torch.nn.GELU(),
            #  torch.nn.Linear(hidden_units, hidden_units),
            #  torch.nn.Dropout(p=dropout_rate))
            # + residual connection
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate,'gelu')
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seq, pos):
        seq_emb = self.item_emb[seq]
        seq_emb *= self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        # seq_emb -> item 정보
        # pos_emb -> 순서 정보
        # 두 개 섞으면 괜찮나?
        seq_emb += pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        # padding이 연산에 영향을 미치지 않도록
        timeline_mask = torch.BoolTensor(seq == 0).cuda() # 0이면 True Tensor
        seq_emb *= ~timeline_mask.unsqueeze(-1)
        # tl = seq_emb.shape[1]
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            # 층 정규화
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            # attention layer
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=None)
            # 원본과 결합
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            # 층 정규화
            seq_emb = self.forward_layer_norms[i](seq_emb)
            # Dense1, activation, Dense2, Dropout, residual connection
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb *= ~timeline_mask.unsqueeze(-1)
        # 최종 출력 층 정규화
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb