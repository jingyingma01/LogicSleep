import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from utils.dataset import GraphData
import utils.util as util

class Trainer:
    def __init__(self, args, net, G_data):
        self.config = args
        self.lambda_crf = self.config.lambda_crf
        self.lambda_logic = self.config.lambda_logic
        self.cuda = self.config.cuda
        self.net = net
        self.feat_dim = G_data.feat_dim
        self.init(G_data.train_gs, G_data.test_gs, G_data.target_gs)
        self.device = util.set_device(self.config)
        self.net.to(self.device)
        self.wdb = self.config.wdb
        self.sch = self.config.sch
        self.num_classes = args.num_class
        self.overlap = self.config.overlap
        self.log_file = 'logs//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
        self.model_file = 'models//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.pth'

    def init(self, train_gs, test_gs, target_gs):
        print('#train: %d, #test: %d #target: %d' % (len(train_gs), len(test_gs), len(target_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        target_data = GraphData(target_gs, self.feat_dim)
        self.train_d = train_data.loader(self.config.batch, True)
        self.test_d = test_data.loader(self.config.batch, False)
        self.target_d = target_data.loader(self.config.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr = self.config.lr, amsgrad = True,
            weight_decay = self.config.weightDecay)
        if self.config.sch == 1:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size = self.config.lrStepSize,
                                                       gamma = self.config.lrGamma)
        elif self.config.sch == 2:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode = 'min',
                                                                  factor = self.config.lrFactor,
                                                                  patience = self.config.lrPatience,
                                                                  verbose = True)

    def run_epoch(self, epoch, data, model, optimizer):
        losses, n_samples = [], 0
        labels = []
        preds = []
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            _, As_seqs, hs_seqs, ys_seqs = batch
            B, L = ys_seqs.shape
            BL = B * L
            center_mask_one, pair_mask_one = util.make_masks_one(self.overlap, L)
            center_mask_B = util.expand_mask(center_mask_one.to(ys_seqs.device), B)
            pair_mask_B = util.expand_mask(pair_mask_one.to(ys_seqs.device), B)

            gs = As_seqs.reshape(BL, As_seqs.size(-2), As_seqs.size(-1)).float()
            hs = hs_seqs.reshape(BL, hs_seqs.size(-2), hs_seqs.size(-1)).float()

            gs, hs, ys_seqs = map(util.to_cuda, [gs, hs, ys_seqs])
            logits, embedding, salient_spacial, salient_node, ts_trasaction = model(gs, hs)
            S = logits.size(-1)
            logits_seq = logits.view(B, L, S)

            ce_idx = center_mask_B.nonzero(as_tuple = False)
            logits_ce = logits_seq[ce_idx[:, 0], ce_idx[:, 1], :]
            labels_ce = ys_seqs[ce_idx[:, 0], ce_idx[:, 1]]
            loss_ce = F.cross_entropy(logits_ce, labels_ce)

            logits_crf = logits_seq[:, (self.config.overlap-1):-(self.config.overlap-1), :]
            ys_crf = ys_seqs[:, (self.config.overlap-1):-(self.config.overlap-1)]
            mask_crf = torch.ones_like(ys_crf, dtype=torch.bool)
            loss_crf = self.net.crf_nll(logits_crf, ys_crf, mask_crf)

            probs = F.softmax(logits_seq, dim = -1)
            loss_logic = util.dl2_logic_loss(probs, pair_mask_B)

            loss = loss_ce + self.lambda_crf * loss_crf + self.lambda_logic * loss_logic

            with (torch.no_grad()):
                paths = self.net.crf_decode(logits_crf, mask_crf)
                preds_seq = torch.full_like(ys_seqs, fill_value = -1)
                for b, path in enumerate(paths):
                    preds_seq[b, (self.config.overlap-1) : -(self.config.overlap-1)] = \
                        torch.tensor(path, device = ys_seqs.device)

            valid_mask = (preds_seq != -1)

            valid_count = int(valid_mask.sum().item())
            losses.append(loss * valid_count)
            n_samples += valid_count

            y_valid = ys_seqs[valid_mask]
            p_valid = preds_seq[valid_mask]
            labels.append(y_valid.detach().cpu())
            preds.append(p_valid.detach().cpu())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = sum(losses) / n_samples
        concatenated_label = torch.cat(labels, dim=0)
        one_dimensional_label = concatenated_label.view(-1)
        label_list = one_dimensional_label.tolist()
        concatenated_pred = torch.cat(preds, dim=0)
        one_dimensional_pred = concatenated_pred.view(-1)
        pred_list = one_dimensional_pred.tolist()

        acc, f1_cls, f1 = util.compute_metrics(label_list, pred_list)

        return (avg_loss.item(), acc, f1_cls, f1)

    def train(self):
        max_train_acc = 0.0
        max_test_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f max %.5f\n'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f\n'
        target_str = 'Target epoch %d: loss %.5f acc %.5f\n'
        stage_acc_str = 'Wake F1 %.5f N1 F1 %.5f N2 F1 %.5f N3 F1 %.5f REM F1 %.5f\n'
        for e_id in range(self.config.num_epochs):
            self.net.train()
            loss, acc, class_f1, f1 = self.run_epoch(e_id, self.train_d, self.net, self.optimizer)
            max_train_acc = max(max_train_acc, acc)
            print(train_str % (e_id, loss, acc, max_train_acc))
            print(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))
            if e_id == 0:
                with open(self.log_file, 'a+') as f:
                    f.write(str(self.config))
                    f.write('\n')
            with open(self.log_file, 'a+') as f:
                f.write(train_str % (e_id, loss, acc, max_train_acc))
                f.write(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))
                f.write("\n")
            self.net.eval()
            with torch.no_grad():
                loss, acc, class_f1, f1 = self.run_epoch(e_id, self.test_d, self.net, None)
                if self.config.sch == 1:
                    self.scheduler.step()
                elif self.config.sch == 2:
                    self.scheduler.step(loss)
            max_test_acc = max(max_test_acc, acc)
            print(test_str % (e_id, loss, acc, max_test_acc))
            print(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))

            with open(self.log_file, 'a+') as f:
                f.write(test_str % (e_id, loss, acc, max_test_acc))
                f.write(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))
                f.write("\n")
            if acc == max_test_acc:
                torch.save(self.net, self.model_file)
                self.net.eval()
                with torch.no_grad():
                    loss, acc, class_f1, f1 = self.run_epoch(e_id, self.target_d, self.net, None)
                    if self.config.sch == 1:
                        self.scheduler.step()
                    elif self.config.sch == 2:
                        self.scheduler.step(loss)
                print(target_str % (e_id, loss, acc))
                print(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))

                with open(self.log_file, 'a+') as f:
                    f.write(target_str % (e_id, loss, acc))
                    f.write(stage_acc_str % (class_f1[0], class_f1[1], class_f1[2], class_f1[3], class_f1[4]))
                    f.write("\n")