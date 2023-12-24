import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity

from module.model import MaskedMultimodalAutoencoder
from module.submodule import SupportEncoder, LayerNormalization
from module.spectral_norm import spectral_norm
from module.utils import *

from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
import json

class Extractor(nn.Module):
    """
    Matching metric based on KB Embeddings
    """
    def __init__(self, embed_dim, num_symbols, embed=None):
        super(Extractor, self).__init__()
        self.embed_dim = int(embed_dim)
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(self.embed_dim, int(self.embed_dim/2))
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim/2))
        self.fc2 = nn.Linear(self.embed_dim, int(self.embed_dim/2))

        self.dropout = nn.Dropout(0.2)
        self.dropout_e = nn.Dropout(0.2)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.reshape_layer = nn.Linear(d_model, self.embed_dim)
        self.support_encoder = SupportEncoder(self.embed_dim, 2*self.embed_dim, dropout=0.2)
        #self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        entities = connections[:,:,1].squeeze(-1)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 50, embed_dim)
        concat_embeds = ent_embeds

        out = self.gcn_w(concat_embeds)
        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def entity_encoder(self, entity1, entity2):
        entity1 = self.dropout_e(entity1)
        entity2 = self.dropout_e(entity2)
        entity1 = self.fc1(entity1)
        entity2 = self.fc2(entity2)
        entity = torch.cat((entity1, entity2), dim=-1)
        return entity.tanh() # (batch, embed_dim)

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta
        
        query_e1 = self.symbol_emb(query[:,0]) # (batch, embed_dim)
        query_e2 = self.symbol_emb(query[:,1]) # (batch, embed_dim)
        query_e = self.entity_encoder(query_e1, query_e2)

        support_e1 = self.symbol_emb(support[:,0]) # (batch, embed_dim)
        support_e2 = self.symbol_emb(support[:,1]) # (batch, embed_dim)
        support_e = self.entity_encoder(support_e1, support_e2)

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)
        
        query_neighbor = torch.cat((query_left, query_e,  query_right), dim=-1) # tanh
        support_neighbor = torch.cat((support_left, support_e, support_right), dim=-1) # tanh

        support = self.reshape_layer(support_neighbor)
        query = self.reshape_layer(query_neighbor)

        support_g = self.support_encoder(support) # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)

        # cosine similarity
        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return query_g, matching_scores

    def update(self, embed):
        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
        self.symbol_emb.weight.requires_grad = False

class Discriminator(nn.Module):
    def __init__(self, dropout=0.3):
        super(Discriminator, self).__init__()

        fc_middle = nn.Linear(200, 200)
        self.fc_middle = spectral_norm(fc_middle)

        fc_TF = nn.Linear(200, 1) # True or False
        self.fc_TF = spectral_norm(fc_TF)

        self.layer_norm = LayerNormalization(200)

    def forward(self, ep_vec, centroid_matrix):

        middle_vec = F.leaky_relu(self.fc_middle(ep_vec))
        middle_vec = self.layer_norm(middle_vec)

        centroid_matrix = F.leaky_relu(self.fc_middle(centroid_matrix))
        centroid_matrix = self.layer_norm(centroid_matrix)

        # determine True or False
        logit_TF = self.fc_TF(middle_vec)

        # determine label
        class_scores = torch.matmul(middle_vec, centroid_matrix.t())

        return middle_vec, logit_TF, class_scores

class ZSLmodule(nn.Module):
    def __init__(self, args, data_path, r2id, e2id, device, dataset, pretrain_margin=3.0):
        super(ZSLmodule, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.data_path = data_path
        self.train_tasks = json.load(open(os.path.join(self.data_path, "train_tasks_zsl.json")))
        self.test_tasks = json.load(open(os.path.join(self.data_path, "test_tasks_zsl.json")))
        self.rel2id = r2id
        self.ent2id = e2id
        self.device = device
        self.prertain_margin = pretrain_margin
        self.rel2candidates = json.load(
            open(os.path.join(self.data_path, "rel2candidates_all.json"))
        )
        self.e1rel_e2 = json.load(open(os.path.join(self.data_path, "e1rel_e2_all.json")))
        
        noises = Variable(torch.randn(self.test_sample, self.noise_dim)).to(device)
        self.test_noises = 0.1 * noises
        self.meta = not self.no_meta
        self.label_num = len(self.train_tasks.keys())

        batch_rels = torch.arange(0, len(self.rel2id))
        batch_data = dataset.generate_batch([], batch_rels)
        des_tokens, des_pad_masks = batch_data['rel_des'], batch_data['rel_des_padding_mask']
        self.des_tokens = des_tokens
        self.des_pad_masks = des_pad_masks

        self.rela2label = dict()
        rela_sorted = sorted(list(self.train_tasks.keys()))
        for i, rela in enumerate(rela_sorted):
            self.rela2label[rela] = int(i)

        ent_embs = torch.rand((dataset.num_nodes, self.emb_dim))
        rel_embs = torch.rand((dataset.num_relations, self.emb_dim))

        self.load_embed(ent_embs, rel_embs)
        self.num_symbols = len(self.symbol2id.keys()) - 1  #
        self.pad_id = self.num_symbols

        self.Extractor = Extractor(
            self.emb_dim, self.num_symbols, embed=self.symbol2vec
        )
        self.Extractor.to(device)
        self.Extractor.apply(weights_init)
        self.E_parameters = filter(
            lambda p: p.requires_grad, self.Extractor.parameters()
        )
        self.optim_E = torch.optim.Adam(self.E_parameters, lr=self.lr_E)

        self.Discriminator = Discriminator()
        self.Discriminator.to(device)
        self.Discriminator.apply(weights_init)
        self.D_parameters = filter(
            lambda p: p.requires_grad, self.Discriminator.parameters()
        )
        self.optim_D = torch.optim.Adam(self.D_parameters, lr=self.lr_D, betas=(0.5, 0.9))
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
            self.optim_D, milestones=[20000], gamma=0.2
        )

        self.num_ents = len(self.ent2id.keys())
        print("##BUILDING CONNECTION MATRIX")
        degrees = self.build_connection(max_=self.max_neighbor)
    
    def save(self, generate_model):
        torch.save(generate_model.state_dict(), os.path.join(self.save_path, "Generator"))
        torch.save(self.Discriminator.state_dict(), os.path.join(self.save_path, "Discriminator"))

    def load_embed(self, ent_embs, rel_embs):
        symbol_id = {}
        print("##LOADING PRE-TRAINED EMBEDDING")
        ent_embed = ent_embs.numpy()
        rel_embed = rel_embs.numpy()
        i = 0
        embeddings = []
        for key in self.rel2id.keys():
            if key not in ["", "OOV"]:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(rel_embed[self.rel2id[key], :]))

        for key in self.ent2id.keys():
            if key not in ["", "OOV"]:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(ent_embed[self.ent2id[key], :]))

        symbol_id["PAD"] = i
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        embeddings = np.array(embeddings)

        self.symbol2id = symbol_id
        self.symbol2vec = embeddings
    
    def update_embed(self, ent_embs, rel_embs):
        self.load_embed(ent_embs, rel_embs)
        self.Extractor.update(self.symbol2vec)

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        for rel in self.train_tasks.keys():
            for tri in self.train_tasks[rel]:
                e1, rel, e2 = tri
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                # self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))
                self.e1_rele2[e2].append((self.symbol2id[rel], self.symbol2id[e1]))
        for rel in self.test_tasks.keys():
            for tri in self.test_tasks[rel]:
                e1, rel, e2 = tri
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                # self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))
                self.e1_rele2[e2].append((self.symbol2id[rel], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))
        return degrees
    
    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(
                np.stack([self.connections[_, :, :] for _ in left], axis=0)
            )
        ).to(self.device)
        left_degrees = Variable(
            torch.FloatTensor([self.e1_degrees[_] for _ in left])
        ).to(self.device)
        right_connections = Variable(
            torch.LongTensor(
                np.stack([self.connections[_, :, :] for _ in right], axis=0)
            )
        ).to(self.device)
        right_degrees = Variable(
            torch.FloatTensor([self.e1_degrees[_] for _ in right])
        ).to(self.device)
        return (left_connections, left_degrees, right_connections, right_degrees)
    
    def pretrain_Extractor(self):
        pretrain_losses = deque([], 100)
        i = 0
        for data in Extractor_generate(
            self.data_path,
            self.pretrain_batch_size,
            self.symbol2id,
            self.ent2id,
            self.e1rel_e2,
            self.pretrain_few,
            self.pretrain_subepoch,
        ):
            i += 1

            (
                support,
                query,
                false,
                support_left,
                support_right,
                query_left,
                query_right,
                false_left,
                false_right,
            ) = data

            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).to(self.device)
            query = Variable(torch.LongTensor(query)).to(self.device)
            false = Variable(torch.LongTensor(false)).to(self.device)

            _, query_scores = self.Extractor(
                query, support, query_meta, support_meta
            )
            _, false_scores = self.Extractor(
                false, support, false_meta, support_meta
            )

            margin_ = query_scores - false_scores
            pretrain_loss = F.relu(self.pretrain_margin - margin_).mean()

            self.optim_E.zero_grad()
            pretrain_loss.backward()

            pretrain_losses.append(pretrain_loss.item())

            if i % self.pretrain_loss_every == 0:
                print(
                    "Step: %d, Feature Extractor Pretraining loss: %.2f"
                    % (i, np.mean(pretrain_losses))
                )

            self.optim_E.step()

            if i > self.pretrain_times:
                break
        self.save_pretrain()

    def train(self, generate_model):
        print("\n##START ADVERSARIAL TRAINING...")
        self.pretrain_Extractor()
        self.centroid_matrix = torch.zeros((len(self.train_tasks), self.emb_dim))
        self.centroid_matrix = self.centroid_matrix.to(self.device)
        
        grad_list = ['generate_fc_layer.weight_orig','generate_fc_layer.bias','des_rel_map_layer1.weight_orig', 'des_rel_map_layer1.bias', 
                     'des_rel_map_layer2.weight_orig', 'des_rel_map_layer2.bias', 'layer_norm.a_2', 'layer_norm.b_2']
        for name, param in generate_model.named_parameters():
            if name in grad_list:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.G_parameters = filter(
            lambda p: p.requires_grad, generate_model.parameters()
        )
        self.optim_G = torch.optim.Adam(self.G_parameters, lr=self.lr_maximum, betas=(0.5, 0.9))
        self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            self.optim_G, milestones=[4000], gamma=0.2
        )

        for relname in self.train_tasks.keys():
            query, query_left, query_right, label_id = centroid_generate(
                relname,
                self.symbol2id,
                self.ent2id,
                self.train_tasks,
                self.rela2label,
            )
            query_meta = self.get_meta(query_left, query_right)
            query = Variable(torch.LongTensor(query)).to(self.device)
            query_ep, _ = self.Extractor(query, query, query_meta, query_meta)
            self.centroid_matrix[label_id] = query_ep.data.mean(dim=0)
        self.centroid_matrix = Variable(self.centroid_matrix)
        
        D_every = self.D_epoch * self.loss_every
        D_losses = deque([], D_every)
        D_real_losses, D_real_class_losses, D_fake_losses, D_fake_class_losses = (
            deque([], D_every),
            deque([], D_every),
            deque([], D_every),
            deque([], D_every),
        )
        G_every = self.G_epoch * self.loss_every
        G_losses = deque([], G_every)
        G_fake_losses, G_class_losses, G_VP_losses, G_real_class_losses = (
            deque([], G_every),
            deque([], G_every),
            deque([], G_every),
            deque([], G_every),
        )
        G_data = train_generate_decription(
            self.data_path,
            self.G_batch_size,
            self.symbol2id,
            self.ent2id,
            self.e1rel_e2,
            self.rel2id,
            self.gan_batch_rela,
            self.rela2label,
            self.des_tokens,
            self.des_pad_masks
        )

        generate_model.zero_grad()
        self.Discriminator.zero_grad()

        for epoch in range(self.train_times):
            # train Discriminator
            self.Discriminator.train()
            generate_model.eval()
            for _ in range(self.D_epoch):  # D_epoch = 5
                ### Discriminator real part
                (
                    D_des_tokens,
                    D_des_pad_mask,
                    query,
                    query_left,
                    query_right,
                    D_false,
                    D_false_left,
                    D_false_right,
                    D_labels,
                ) = G_data.__next__()

                # real part
                query_meta = self.get_meta(query_left, query_right)
                query = Variable(torch.LongTensor(query)).to(self.device)
                D_real, _ = self.Extractor(query, query, query_meta, query_meta)

                # fake part
                noises = Variable(torch.randn(len(query), self.noise_dim)).to(self.device)
                D_fake = generate_model.generate(D_des_tokens.to(self.device), D_des_pad_mask.to(self.device), noises)

                # neg part
                D_false_meta = self.get_meta(D_false_left, D_false_right)
                D_false = Variable(torch.LongTensor(D_false)).to(self.device)
                D_neg, _ = self.Extractor(D_false, D_false, D_false_meta, D_false_meta)

                # generate Discriminator part vector
                centroid_matrix_ = (
                    self.centroid_matrix
                )  # gaussian_noise(self.centroid_matrix)
                _, D_real_decision, D_real_class = self.Discriminator(
                    D_real.detach(), centroid_matrix_
                )
                _, D_fake_decision, D_fake_class = self.Discriminator(
                    D_fake.detach(), centroid_matrix_
                )
                _, _, D_neg_class = self.Discriminator(
                    D_neg.detach(), self.centroid_matrix
                ) 

                # real adversarial training loss
                loss_D_real = -torch.mean(D_real_decision)

                # adversarial training loss
                loss_D_fake = torch.mean(D_fake_decision)

                # real classification loss
                D_real_scores = D_real_class[range(len(query)), D_labels]
                D_neg_scores = D_neg_class[range(len(query)), D_labels]
                D_margin_real = D_real_scores - D_neg_scores
                loss_rela_class = F.relu(self.pretrain_margin - D_margin_real).mean()

                # fake classification loss
                D_fake_scores = D_fake_class[range(len(query)), D_labels]
                D_margin_fake = D_fake_scores - D_neg_scores
                loss_fake_class = F.relu(self.pretrain_margin - D_margin_fake).mean()

                grad_penalty = calc_gradient_penalty(
                    self.Discriminator,
                    D_real.data,
                    D_fake.data,
                    len(query),
                    self.centroid_matrix,
                    self.device
                )

                loss_D = (
                    loss_D_real
                    + 0.5 * loss_rela_class
                    + loss_D_fake
                    + grad_penalty
                    + 0.5 * loss_fake_class
                )

                # D_real_losses, D_real_class_losses, D_fake_losses, D_fake_class_losses
                D_losses.append(loss_D.item())
                D_real_losses.append(loss_D_real.item())
                D_real_class_losses.append(loss_rela_class.item())
                D_fake_losses.append(loss_D_fake.item())
                D_fake_class_losses.append(loss_fake_class.item())

                loss_D.backward()
                self.optim_D.step()
                self.scheduler_D.step()
                
                generate_model.zero_grad()
                self.Discriminator.zero_grad()
            # train Generator
            self.Discriminator.eval()
            generate_model.train()
            for _ in range(self.G_epoch):  # G_epoch = 1
                (
                    G_des_tokens,
                    G_des_pad_mask,
                    query,
                    query_left,
                    query_right,
                    G_false,
                    G_false_left,
                    G_false_right,
                    G_labels,
                ) = G_data.__next__()

                # G sample
                noises = Variable(torch.randn(len(query), self.noise_dim)).to(self.device)
                G_sample = generate_model.generate(G_des_tokens.to(self.device), G_des_pad_mask.to(self.device), noises)  # to train G

                # real data
                query_meta = self.get_meta(query_left, query_right)
                query = Variable(torch.LongTensor(query)).to(self.device)
                G_real, _ = self.Extractor(query, query, query_meta, query_meta)

                # This negative for classification loss
                G_false_meta = self.get_meta(G_false_left, G_false_right)
                G_false = Variable(torch.LongTensor(G_false)).to(self.device)
                G_neg, _ = self.Extractor(
                    G_false, G_false, G_false_meta, G_false_meta
                )  # just use Extractor to generate ep vector

                # generate Discriminator part vector
                centroid_matrix_ = self.centroid_matrix
                _, G_decision, G_class = self.Discriminator(G_sample, centroid_matrix_)
                _, _, G_real_class = self.Discriminator(
                    G_real.detach(), centroid_matrix_
                )
                _, _, G_neg_class = self.Discriminator(G_neg.detach(), centroid_matrix_)

                # adversarial training loss
                loss_G_fake = -torch.mean(G_decision)

                # G sample classification loss
                G_scores = G_class[range(len(query)), G_labels]
                G_neg_scores = G_neg_class[range(len(query)), G_labels]
                G_margin_ = G_scores - G_neg_scores
                loss_G_class = F.relu(self.pretrain_margin - G_margin_).mean()

                # real classification loss
                G_real_scores = G_real_class[range(len(query)), G_labels]
                G_margin_real = G_real_scores - G_neg_scores
                loss_rela_class_ = F.relu(self.pretrain_margin - G_margin_real).mean()
                
                # Visual Pivot Regularization
                count = 0
                loss_VP = Variable(torch.Tensor([0.0])).to(self.device)
                for i in range(len(self.train_tasks.keys())):
                    sample_idx = (np.array(G_labels) == i).nonzero()[0]
                    count += len(sample_idx)
                    if len(sample_idx) == 0:
                        loss_VP += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        loss_VP += (
                            (G_sample_cls.mean(dim=0) - self.centroid_matrix[i])
                            .pow(2)
                            .sum()
                            .sqrt()
                        )
                assert count == len(query)
                loss_VP *= float(1.0 / self.gan_batch_rela)

                # Generator loss function
                loss_G = (
                    loss_G_fake + loss_G_class + 3.0 * loss_VP
                ) 

                # G_fake_losses, G_class_losses, G_VP_losses
                G_losses.append(loss_G.item())
                G_fake_losses.append(loss_G_fake.item())
                G_class_losses.append(loss_G_class.item())
                G_real_class_losses.append(loss_rela_class_.item())
                G_VP_losses.append(loss_VP.item())

                loss_G.backward()
                self.optim_G.step()
                self.scheduler_G.step()

                generate_model.zero_grad()
                self.Discriminator.zero_grad()
            
            if epoch % self.loss_every == 0 and epoch != 0:
                D_screen = [
                    np.mean(D_real_losses),
                    np.mean(D_real_class_losses),
                    np.mean(D_fake_losses),
                    np.mean(D_fake_class_losses),
                ]
                G_screen = [
                    np.mean(G_fake_losses),
                    np.mean(G_class_losses),
                    np.mean(G_real_class_losses),
                    np.mean(G_VP_losses),
                ]
                print(
                    "Epoch: %d, D_loss: %.2f [%.2f, %.2f, %.2f, %.2f], G_loss: %.2f [%.2f, %.2f, %.2f, %.2f]"
                    % (
                        epoch,
                        np.mean(D_losses),
                        D_screen[0],
                        D_screen[1],
                        D_screen[2],
                        D_screen[3],
                        np.mean(G_losses),
                        G_screen[0],
                        G_screen[1],
                        G_screen[2],
                        G_screen[3],
                    )
                )
        
        self.save(generate_model)
        hits10_test, hits5_test, mrr_test = self.eval(generate_model, mode="test", meta=self.meta)
    
    def eval(self, generate_model, mode="test", meta=True, load_pretrain=False):
        if load_pretrain:
            self.load_pretrain()
            self.load()
        generate_model.eval()
        self.Discriminator.eval()
        self.Extractor.eval()
        symbol2id = self.symbol2id

        # logging.info('EVALUATING ON %s DATA' % mode.upper())
        print("##EVALUATING ON %s DATA" % mode.upper())

        test_candidates = json.load(
            open(self.data_path + '/' + mode + "_candidates.json")
        )

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in test_candidates.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            des_token = self.des_tokens[self.rel2id[query_]]
            des_tokens = des_token.unsqueeze(0).expand(self.test_sample, -1).to(self.device)
            des_pad_mask = self.des_pad_masks[self.rel2id[query_]]
            des_pad_masks = des_pad_mask.unsqueeze(0).expand(self.test_sample, -1).to(self.device)
            relation_vecs = generate_model.generate(des_tokens.to(self.device), des_pad_masks.to(self.device), self.test_noises)
            relation_vecs = relation_vecs.detach().cpu().numpy()

            for e1_rel, tail_candidates in test_candidates[query_].items():
                head, rela, _ = e1_rel.split("\t")

                true = tail_candidates[0]
                query_pairs = []
                query_pairs.append([symbol2id[head], symbol2id[true]])

                if meta:
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[head])
                    query_right.append(self.ent2id[true])

                for tail in tail_candidates[1:]:
                    query_pairs.append([symbol2id[head], symbol2id[tail]])
                    if meta:
                        query_left.append(self.ent2id[head])
                        query_right.append(self.ent2id[tail])

                query = Variable(torch.LongTensor(query_pairs)).to(self.device)

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    candidate_vecs, _ = self.Extractor(
                        query, query, query_meta, query_meta
                    )

                    candidate_vecs.detach()
                    candidate_vecs = candidate_vecs.data.cpu().numpy()

                    scores = cosine_similarity(candidate_vecs, relation_vecs)

                    scores = scores.mean(axis=1)

                    assert scores.shape == (len(query_pairs),)

                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            print(
                "{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}".format(
                    mode + query_,
                    np.mean(hits10_),
                    np.mean(hits5_),
                    np.mean(hits1_),
                    np.mean(mrr_),
                )
            )

        print("############   " + mode + "    #############")
        print("HITS10: {:.3f}".format(np.mean(hits10)))
        print("HITS5: {:.3f}".format(np.mean(hits5)))
        print("HITS1: {:.3f}".format(np.mean(hits1)))
        print("MAP: {:.3f}".format(np.mean(mrr)))
        print("###################################")

        return np.mean(hits10), np.mean(hits5), np.mean(mrr)

    def load(self, generate_model):
        generate_model.load_state_dict(torch.load(os.path.join(self.save_path, "Generator")))
        self.Discriminator.load_state_dict(torch.load(os.path.join(self.save_path, "Discriminator")))
    
    def save_pretrain(self):
        torch.save(self.Extractor.state_dict(), os.path.join(self.save_path, "Extractor"))

    def load_pretrain(self):
        self.Extractor.load_state_dict(torch.load(os.path.join(self.save_path, "Extractor")))

    def generate_entity_pair_emb(self, relations, generate_model):
        self.load_pretrain()
        #self.load(generate_model)
        symbol2id = self.symbol2id
        entity_pair_embs = []
        rels = []
        target_idx = []
        for rel in relations:

            # des_token = self.des_tokens[self.rel2id[rel]]
            # des_tokens = des_token.unsqueeze(0).expand(self.test_sample, -1).to(self.device)
            # des_pad_mask = self.des_pad_masks[self.rel2id[rel]]
            # des_pad_masks = des_pad_mask.unsqueeze(0).expand(self.test_sample, -1).to(self.device)
            # relation_vecs = generate_model.generate(des_tokens.to(self.device), des_pad_masks.to(self.device), self.test_noises)
            # relation_vecs = relation_vecs.detach().cpu().numpy()

            related_triples = self.test_tasks[rel]
            entity_pairs = [[symbol2id[tri[0]], symbol2id[tri[2]]] for tri in related_triples]
            query_left = [self.ent2id[tri[0]] for tri in related_triples]
            query_right = [self.ent2id[tri[0]] for tri in related_triples]
            query_meta = self.get_meta(query_left, query_right)
            query = Variable(torch.LongTensor(entity_pairs)).cuda()
            with torch.no_grad():
                entity_pair_emb, _ = self.Extractor(
                    query, query, query_meta, query_meta
                )
            print(entity_pair_emb.shape)
            entity_pair_embs.append(entity_pair_emb.cpu())
            rels += [rel for i in range(len(related_triples))]
            #target_idx.append(len(rels))
            #entity_pair_embs.append(relation_vecs.mean(dim=0).unsqueeze(dim=0))
            #rels.append(rel)

        return entity_pair_embs, rels, target_idx