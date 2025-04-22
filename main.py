import argparse
from smiles_generator.train import train_vae_cli
from smiles_generator.rl_finetune import finetune_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De Novo Compound Generator CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # VAE training
    p1 = subparsers.add_parser("train_vae", help="Train the SMILES VAE")
    p1.add_argument("--data",   type=str, required=True, help="Path to SMILES data file")
    p1.add_argument("--epochs", type=int, default=10,     help="Number of training epochs")

    # RL fine-tuning
    p2 = subparsers.add_parser("finetune", help="Fine-tune with RL (composite reward)")
    p2.add_argument("--rewards", type=str, default="qed,logp,sa",
                    help="Comma-separated reward metrics: qed, logp, sa")
    p2.add_argument("--steps",   type=int, default=10000, help="Number of RL steps")

    args = parser.parse_args()
    if args.mode == "train_vae":
        train_vae_cli(data_path=args.data, epochs=args.epochs)
    elif args.mode == "finetune":
        reward_list = args.rewards.split(',')
        finetune_cli(reward_list=reward_list, num_steps=args.steps)