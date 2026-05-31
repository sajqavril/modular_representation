import argparse
import inspect

import torch
try:
    import ipdb as pdb
except ImportError:
    pdb = None
import os
import pwd
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from modules.instantaneous2 import InstantaneousProcess
from tools.utils import load_yaml
from datasets.sim_dataset import StationaryDataset
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    from lightning.pytorch.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('configs',
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")
    print("######### Seed #########")
    print(args.seed)
    print("########################")

    pl.seed_everything(args.seed)

    data = StationaryDataset(dataset=cfg['DATASET'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = InstantaneousProcess(input_dim=cfg['VAE']['INPUT_DIM'],
                               z_dim=cfg['VAE']['LATENT_DIM'], 
                               z_dim_fix=cfg['VAE']['LATENT_DIM_FIX'],
                               z_dim_change=cfg['VAE']['LATENT_DIM_CHANGE'],
                               lag=cfg['VAE']['LAG'],
                               nclass=cfg['VAE']['NCLASS'],
                               hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                               embedding_dim=cfg['VAE']['EMBED_DIM'],
                               lr=cfg['VAE']['LR'],
                               beta=cfg['VAE']['BETA'],
                               gamma=cfg['VAE']['GAMMA'],
                               theta=cfg['VAE']['THETA'],
                               decoder_dist=cfg['VAE']['DEC']['DIST'],
                               correlation=cfg['MCC']['CORR'],
                               enable_flexible_sparsity=cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'],
                               w_hist=cfg['VAE']['FLEXIBLE_SPARTSITY']['HIST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               w_inst=cfg['VAE']['FLEXIBLE_SPARTSITY']['INST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               layer_dims=cfg['VAE'].get('LAYER_DIMS', None),
                               )

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp, 'seed_%d' % args.seed)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    run_name = cfg['WANDB']['LOG_NAME']
    seed_tag = 'seed%d' % args.seed
    if seed_tag not in run_name:
        run_name = '%s_%s' % (run_name, seed_tag)
    logger = WandbLogger(project=cfg['WANDB']['PROJ_NAME'], name=run_name)

    trainer_kwargs = dict(default_root_dir=log_dir,
                          check_val_every_n_epoch=cfg['MCC']['FREQ'],
                          max_epochs=cfg['VAE']['EPOCHS'],
                          callbacks=[checkpoint_callback],
                          logger=logger)
    if torch.cuda.is_available():
        trainer_params = inspect.signature(pl.Trainer.__init__).parameters
        print('######### Device #########')
        if 'gpus' in trainer_params:
            trainer_kwargs['gpus'] = 1
            print('Using CUDA via PyTorch Lightning gpus=1')
        else:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = 1
            print('Using CUDA via PyTorch Lightning accelerator=gpu, devices=1')
        print('CUDA_VISIBLE_DEVICES=%s' % os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL'))
        print('PyTorch Lightning version=%s' % pl.__version__)
        print('##########################')
    else:
        raise RuntimeError('CUDA is not available; refusing to run CPU training.')

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='I'
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)