import torch
import argparse
import numpy as np
from tqdm import tqdm
from main import set_random
from trainer import Trainer
from utils.data_loader import FileLoader
import utils.tta as tta
import utils.util as util
class_names = ["W", "N1", "N2", "N3", "REM"]

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--cuda', default = 0, type = int, help = 'CUDA device number')
    parser.add_argument('--seed', type = int, default = 0, help = 'seed')
    parser.add_argument('--data', default = 'ISRUC_S3', help = 'data folder name')
    parser.add_argument('--num_node', type = int, default = 10, help = 'num of channels')
    parser.add_argument('--fold', type=int, default=0, help='fold (0..10)')
    parser.add_argument('--batch', type=int, default = 16, help='batch size')
    parser.add_argument('--lr', type=float, default = 1e-5, help='learning rate')
    parser.add_argument('--drop_n', type=float, default = 0.3, help='drop net')
    parser.add_argument('--drop_c', type=float, default = 0.5, help='drop output')
    parser.add_argument('--lambda_logic', type=float, default=0.01, help='Weight λ for Logic (DL2) loss term')

    parser.add_argument('--tta', action = 'store_true', default = False, help = 'enable TTA')
    parser.add_argument('--model_paths', type=str,
                        default='model.pth', help='tta models')

    parser.add_argument('--num_patch', type = int, default = 5, help='Number of Patch')
    parser.add_argument('--feat_dim', type = int, default = 600, help='Feature Dim')
    parser.add_argument('--norm', type = str, default = 'Batch', help='Batch/Layer/Group')
    parser.add_argument('--window', type = int, default = 16, help = 'window size')
    parser.add_argument('--overlap', type = int, default = 2, help = 'overlap size')
    parser.add_argument('--act_n', type = str, default = 'ELU', help = 'network act')
    parser.add_argument('--act_c', type = str, default = 'ELU', help = 'output act')
    parser.add_argument('--gcn_h', type = str, default = '1024 512 256 256', help = 'GCN hidden layer')
    parser.add_argument('--l_n', type = int, default = 3, help = 'The layer of Unet')
    parser.add_argument('--ks', type = str, default = '0.9 0.8 0.7')
    parser.add_argument('--cs', type = str, default = '0.5 0.5 0.5')
    parser.add_argument('--sch', type = int, default = 2, help = 'scheduler')
    parser.add_argument('--chs', type = str, default = '32 64 128 256')
    parser.add_argument('--kernal', type = str, default = '15 9 7 3', help = 'kernal')
    parser.add_argument('--delta_t', type = float, default = 0.8, help='Adjacency Time Matrix')
    parser.add_argument('--delta_p', type = float, default = 0.9, help='Adjacency Position Matrix')
    parser.add_argument('--num_class', type = int, default = 5, help = 'Number of Classification')
    parser.add_argument('--weightDecay', type = float, default = 0.005)
    parser.add_argument('--lrStepSize', type = int, default = 10)
    parser.add_argument('--lrGamma', type = float, default = 0.1)
    parser.add_argument('--lrFactor', type = float, default = 0.5)
    parser.add_argument('--lrPatience', type = int, default = 10)
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = get_args()
    set_random(args.seed)
    model = torch.load(args.model_paths)
    fold = args.fold

    subject_list = []
    if args.data == 'ISRUC_S1':
        data_num = 100
    elif args.data == 'MASS_SS3':
        data_num = 62
    elif args.data == 'ISRUC_S3':
        data_num = 10
    for i in range(1, data_num + 1):
        if fold == 10:
            subject_list.append(i)
        else:
            if i % 10 == fold:
                subject_list.append(i)

    if args.tta:
        model = tta.configure_model(model)
        params, param_names = tta.collect_params(model)
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay = 0)
        adapt_model = tta.TTA(model, optimizer, steps = 1, lambda_logic = args.lambda_logic)
    all_bacc = []
    all_f1 = []
    all_f1_cls = []
    for fold in tqdm(subject_list):
        args.fold = fold
        if args.tta:
            adapt_model.begin_subject()

        G_data = FileLoader(args).load_data(True)
        trainer = Trainer(args, model, G_data)
        torch.cuda.set_device(args.cuda)
        n_samples = 0
        labels = []
        preds = []

        if args.tta:
            adapt_model.optimizer.zero_grad(set_to_none=True)
            for batch in trainer.test_d:
                _, As_seqs, hs_seqs, ys_seqs = batch
                B, L = ys_seqs.shape
                BL = B * L
                center_mask_one, pair_mask_one = util.make_masks_one(args.overlap, L)
                center_mask_B = util.expand_mask(center_mask_one.to(ys_seqs.device), B)
                pair_mask_B = util.expand_mask(pair_mask_one.to(ys_seqs.device), B)
                gs = As_seqs.reshape(BL, As_seqs.size(-2), As_seqs.size(-1)).float()
                hs = hs_seqs.reshape(BL, hs_seqs.size(-2), hs_seqs.size(-1)).float()
                gs, hs, ys_seqs = map(util.to_cuda, [gs, hs, ys_seqs])
                _, loss = adapt_model([gs, hs], B, L, center_mask_B, pair_mask_B, overlap=args.overlap,
                                accumulate_only = True)
            adapt_model.optimizer.step()

        model.eval()
        for batch in tqdm(trainer.test_d):
            _, As_seqs, hs_seqs, ys_seqs = batch
            B, L = ys_seqs.shape
            BL = B * L
            center_mask_one, pair_mask_one = util.make_masks_one(args.overlap, L)
            center_mask_B = util.expand_mask(center_mask_one.to(ys_seqs.device), B)
            pair_mask_B = util.expand_mask(pair_mask_one.to(ys_seqs.device), B)

            gs = As_seqs.reshape(BL, As_seqs.size(-2), As_seqs.size(-1)).float()
            hs = hs_seqs.reshape(BL, hs_seqs.size(-2), hs_seqs.size(-1)).float()

            gs, hs, ys_seqs = map(util.to_cuda, [gs, hs, ys_seqs])
            with torch.no_grad():
                logits, embedding, salient_spacial, salient_node, ts_trasaction = model(gs, hs)
            S = logits.size(-1)
            logits_seq = logits.view(B, L, S)

            logits_crf = logits_seq[:, (args.overlap-1) : -(args.overlap-1), :]
            ys_crf = ys_seqs[:, (args.overlap-1) : -(args.overlap-1)]
            mask_crf = torch.ones_like(ys_crf, dtype = torch.bool)

            paths = model.crf_decode(logits_crf, mask_crf)
            preds_seq = torch.full_like(ys_seqs, fill_value=-1)
            for b, path in enumerate(paths):
                preds_seq[b, (args.overlap-1) : -(args.overlap-1)] = torch.tensor(path, device=ys_seqs.device)

            probs_crf_in = torch.softmax(logits_crf, dim=-1)
            inner_preds = preds_seq[:, args.overlap: -args.overlap]
            conf_inner = probs_crf_in.gather(-1, inner_preds.unsqueeze(-1)).squeeze(-1)
            conf_seq = torch.full(ys_seqs.shape, fill_value=-1.0, device=ys_seqs.device)
            conf_seq[:, args.overlap: -args.overlap] = conf_inner

            valid_mask = (preds_seq != -1)
            valid_count = int(valid_mask.sum().item())
            n_samples += valid_count
            y_valid = ys_seqs[valid_mask]
            p_valid = preds_seq[valid_mask]
            labels.append(y_valid.detach().cpu())
            preds.append(p_valid.detach().cpu())
            c_valid = conf_seq[valid_mask]

        concatenated_label = torch.cat(labels, dim=0)
        one_dimensional_label = concatenated_label.view(-1)
        label_list = one_dimensional_label.tolist()
        concatenated_pred = torch.cat(preds, dim=0)
        one_dimensional_pred = concatenated_pred.view(-1)
        pred_list = one_dimensional_pred.tolist()

        bacc, f1_cls, f1 = util.compute_metrics(label_list, pred_list)

        all_bacc.append(bacc)
        all_f1.append(f1)
        all_f1_cls.append(np.asarray(f1_cls, dtype=float))

    mean_bacc = sum(all_bacc) / len(all_bacc)
    mean_f1 = sum(all_f1) / len(all_f1)
    all_f1_cls = np.stack(all_f1_cls, axis=0)
    per_class_f1_mean = np.nanmean(all_f1_cls, axis=0)

    print(f"Mean BACC (unweighted): {mean_bacc:.4f}")
    print(f"Mean F1 (unweighted):   {mean_f1:.4f}")
    for i in range(5):
        print(
            f"{class_names[i]:>6}: {per_class_f1_mean[i]:.4f}"
        )