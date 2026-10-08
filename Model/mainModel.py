"""Entry point: fine-tune PressureVisionNet on the processed forearm-press
sessions, then evaluate the best checkpoint on the held-out test sessions.

Run with the trajPipeline conda env, from the repo root, after
preprocessing/dataProcessor.py --all:

    ~/anaconda3/envs/trajPipeline/bin/python Model/mainModel.py
    ~/anaconda3/envs/trajPipeline/bin/python Model/mainModel.py --seed 7
    ~/anaconda3/envs/trajPipeline/bin/python Model/mainModel.py --init_weights imagenet

--seed overrides model_training.training.seed and --init_weights overrides
model_training.model.init_weights; either one gives the run its own
checkpoint folder (<checkpoint_dir>/<init>_seed_<seed>/) so runs don't
overwrite each other.
"""

import argparse                                                                #For the command-line overrides
import os                                                                      #For building the per-run checkpoint dir
import time                                                                    #For timing the whole run

import dataset                                                                 #Config loading
import evaluate                                                                #Test-set evaluation
import train                                                                   #The training loop itself


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PressureVisionNet on finger-on-forearm presses")
    parser.add_argument("--seed", type=int, default=None, help="override model_training.training.seed")
    parser.add_argument("--init_weights", choices=["pressurevision", "imagenet"], default=None,
                        help="override model_training.model.init_weights")
    parser.add_argument("--max_epochs", type=int, default=None, help="override model_training.training.max_epochs")
    parser.add_argument("--skip_test", action="store_true", help="do not evaluate on the test split afterwards")
    args = parser.parse_args()                                                #Reading the command line

    run_start = time.perf_counter()                                           #Marking when the run started
    print("=" * 70)
    print("[main] PressureVisionNet force fine-tuning - training run")
    print("=" * 70)

    config = dataset.load_config()                                            #Loading config.yaml
    model_config = config["model_training"]
    training_config = model_config["training"]
    if args.seed is not None:                                                 #Seed given on the command line
        training_config["seed"] = args.seed
    if args.init_weights is not None:                                         #Starting weights given on the command line
        model_config["model"]["init_weights"] = args.init_weights
    if args.max_epochs is not None:
        training_config["max_epochs"] = args.max_epochs
    if args.seed is not None or args.init_weights is not None:                #Giving this run its own checkpoint folder
        run_name = f"{model_config['model']['init_weights']}_seed_{training_config['seed']}"
        training_config["checkpoint_dir"] = os.path.join(training_config["checkpoint_dir"], run_name)

    print(f"[main] Starting weights: {model_config['model']['init_weights']}  Seed: {training_config['seed']}  "
          f"Max epochs: {training_config['max_epochs']}  Batch size: {model_config['loader']['batch_size']}")
    best_val_mae_n, best_path = train.train(config)                           #Dataset build + model build + training loop

    if not args.skip_test:                                                    #Scoring the best checkpoint on unseen sessions
        print("[main] Evaluating the best checkpoint on the test split")
        test_metrics = evaluate.evaluate(best_path, split="test")
    elapsed_min = (time.perf_counter() - run_start) / 60.0

    print("=" * 70)
    print(f"[main] Run finished in {elapsed_min:.1f} minutes")
    print(f"[main] Best validation MAE: {best_val_mae_n:.3f} N   Checkpoint: {best_path}")
    if not args.skip_test:
        print(f"[main] Test MAE: {test_metrics['overall']['mae_n']:.3f} N   "
              f"(trivial baseline {test_metrics['trivial_baseline_mean_train_force']['mae_n']:.3f} N)")
    print("=" * 70)


if __name__ == "__main__":                                                    #Standard entry-point guard
    main()                                                                    #Running the pipeline
