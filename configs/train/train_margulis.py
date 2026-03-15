import os
import json
import torch
import argparse
import numpy as np
import logging
from datetime import datetime
from graphqec.qecc.ldpc_code.margulis_code import MargulisCode
from graphqec.decoder.nn import get_model


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    save_dir = cfg['training']['save_dir']
    logger = setup_logging(save_dir)
    logger.info(f'Config: {config_path}')
    logger.info(f'Started at: {datetime.now()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    torch.manual_seed(cfg['training']['seed'])

    code = MargulisCode.from_file(cfg['code']['npz_path'])
    logger.info(f'Code loaded: [[{2*code.n_half}, {code.blue_print.Lx.shape[0]}, ~20]]')

    tanner_graph = code.get_tanner_graph().to(device)
    model_cfg = dict(cfg['model'])
    name = model_cfg.pop('name')
    decoder = get_model(name=name, tanner_graph=tanner_graph, **model_cfg).to(device)
    logger.info(f'Decoder built: {sum(p.numel() for p in decoder.parameters()):,} parameters')

    # Get split sizes directly from decoder
    num_init_check = int(decoder.num_init_check)
    num_cycle_check = int(decoder.num_cycle_check)
    logger.info(f'num_init_check={num_init_check}, num_cycle_check={num_cycle_check}')

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=(cfg['training']['beta1'], cfg['training']['beta2'])
    )

    # Resume from checkpoint if exists
    start_epoch = 0
    best_acc = 0.0
    latest_ckpt = os.path.join(save_dir, 'latest.pt')
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        decoder.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        logger.info(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}')

    error_rates = cfg['training']['error_rates']
    num_cycles = cfg['training']['num_cycles']
    batch_size = cfg['dataloader']['batch_size']
    num_epochs = cfg['training']['num_epochs']

    # Save results as CSV
    results_path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            header = 'epoch,loss,' + ','.join([f'acc_p{p}' for p in error_rates]) + ',mean_acc\n'
            f.write(header)

    for epoch in range(start_epoch, num_epochs):
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0

        for p in error_rates:
            for r in num_cycles:
                dem = code.get_dem(num_cycle=r, physical_error_rate=p)
                sampler = dem.compile_sampler(seed=cfg['training']['seed'] + epoch)
                syndromes, obs_flips, _ = sampler.sample(batch_size * 10)

                for i in range(0, len(syndromes), batch_size):
                    syn_batch = torch.tensor(
                        syndromes[i:i+batch_size], dtype=torch.long
                    ).to(device)
                    obs_batch = torch.tensor(
                        obs_flips[i:i+batch_size], dtype=torch.float32
                    ).to(device)

                    # Split exactly as _decode() does
                    encoding = syn_batch[:, :num_init_check]
                    cycle = syn_batch[:, num_init_check:-num_init_check].reshape(
                        syn_batch.shape[0], -1, num_cycle_check
                    )
                    readout = syn_batch[:, -num_init_check:]

                    optimizer.zero_grad()
                    logits = decoder((encoding, cycle, readout))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, obs_batch
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation
        decoder.eval()
        val_accs = []
        for p in error_rates:
            dem = code.get_dem(num_cycle=8, physical_error_rate=p)
            sampler = dem.compile_sampler()
            syn_val, obs_val, _ = sampler.sample(1000)
            with torch.no_grad():
                preds = decoder.decode(syn_val, batch_size=10)
            acc = np.mean(preds == obs_val)
            val_accs.append(acc)

        mean_acc = np.mean(val_accs)

        logger.info(f'Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val acc: {mean_acc:.4f}')
        for p, acc in zip(error_rates, val_accs):
            logger.info(f'  p={p:.3f}: {acc:.4f}')

        # Save CSV row
        with open(results_path, 'a') as f:
            row = f'{epoch+1},{avg_loss:.6f},' + ','.join([f'{a:.6f}' for a in val_accs]) + f',{mean_acc:.6f}\n'
            f.write(row)

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'model_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': mean_acc,
            'best_acc': max(mean_acc, best_acc),
        }
        torch.save(ckpt, latest_ckpt)
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
            logger.info(f'  New best saved!')

    logger.info(f'Training done. Best val acc: {best_acc:.4f}')
    logger.info(f'Checkpoints and logs saved to {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train/margulis240.json')
    args = parser.parse_args()
    main(args.config)
