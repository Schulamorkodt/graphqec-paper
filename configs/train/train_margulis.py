import os
import json
import torch
import argparse
import numpy as np
from graphqec.qecc.ldpc_code.margulis_code import MargulisCode
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.decoder.nn.dataloader import get_incremental_dataloader

def main(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(cfg['training']['seed'])

    code = MargulisCode.from_file(cfg['code']['npz_path'])
    print(f'Code loaded: [[{2*code.n_half}, {code.blue_print.Lx.shape[0]}, ~20]]')

    tanner_graph = code.get_tanner_graph().to(device)
    model_cfg = dict(cfg['model'])
    name = model_cfg.pop('name')
    decoder = build_neural_decoder(tanner_graph, {'name': name, **model_cfg}).to(device)
    print(f'Decoder built: {sum(p.numel() for p in decoder.parameters()):,} parameters')

    save_dir = cfg['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=(cfg['training']['beta1'], cfg['training']['beta2'])
    )

    error_rates = cfg['training']['error_rates']
    num_cycles = cfg['training']['num_cycles']
    batch_size = cfg['dataloader']['batch_size']
    num_epochs = cfg['training']['num_epochs']
    best_acc = 0.0

    for epoch in range(num_epochs):
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
                        syndromes[i:i+batch_size], dtype=torch.float32
                    ).to(device)
                    obs_batch = torch.tensor(
                        obs_flips[i:i+batch_size], dtype=torch.float32
                    ).to(device)

                    optimizer.zero_grad()
                    logits = decoder(syn_batch)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, obs_batch
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

        avg_loss = epoch_loss / num_batches

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
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val acc: {mean_acc:.4f}')
        for p, acc in zip(error_rates, val_accs):
            print(f'  p={p:.3f}: {acc:.4f}')

        ckpt = {
            'epoch': epoch,
            'model_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': mean_acc,
        }
        torch.save(ckpt, os.path.join(save_dir, 'latest.pt'))
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
            print(f'  New best saved!')

    print(f'Training done. Best val acc: {best_acc:.4f}')
    print(f'Checkpoints saved to {save_dir}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train/margulis240.json')
    args = parser.parse_args()
    main(args.config)
