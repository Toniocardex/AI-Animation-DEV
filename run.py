"""
run.py
Entrypoint principale - esegui i comandi da qui
"""
import argparse
import sys
from pathlib import Path


def _import_catalog_main():
    """Import compatibile sia con package sia con file flat."""
    try:
        from tools.catalog_assets import main as catalog_main
    except ModuleNotFoundError:
        from tools_catalog_assets import main as catalog_main
    return catalog_main


def _import_build_dataset():
    """Import compatibile sia con package sia con file flat."""
    try:
        from src.dataset.quick_build import build_dataset
    except ModuleNotFoundError:
        from src_dataset_quick_build import build_dataset
    return build_dataset


def _import_train():
    """Import compatibile sia con package sia con file flat."""
    try:
        from src.training.quick_train import train
    except ModuleNotFoundError:
        from src_training_quick_train import train
    return train


def _import_generate_main():
    """Import compatibile sia con package sia con file flat."""
    try:
        from src.inference.quick_generate import main as generate_main
    except ModuleNotFoundError:
        from src_inference_quick_generate import main as generate_main
    return generate_main


def _import_train_vqvae():
    try:
        from src.training.vqvae_train import train_vqvae
    except ModuleNotFoundError:
        from src_training_vqvae_train import train_vqvae
    return train_vqvae


def _import_train_prior():
    try:
        from src.training.prior_train import train_prior
    except ModuleNotFoundError:
        from src_training_prior_train import train_prior
    return train_prior


def _import_generate_vqvae():
    try:
        from src.inference.vqvae_generate import generate_vqvae_sample
    except ModuleNotFoundError:
        from src_inference_vqvae_generate import generate_vqvae_sample
    return generate_vqvae_sample


def ensure_directories():
    """Crea le cartelle necessarie."""
    dirs = [
        "data/raw/licensed",
        "data/processed",
        "data/final/train",
        "data/final/val",
        "data/final/test",
        "data/processed",
        "configs",
        "checkpoints",
        "outputs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def cmd_catalog(args):
    """Cataloga gli asset."""
    print("Catalogo asset dal pack...\n")
    catalog_main = _import_catalog_main()
    catalog_main()


def cmd_build(args):
    """Costruisce il dataset."""
    print("Costruzione dataset...\n")
    build_dataset = _import_build_dataset()
    build_dataset()


def cmd_train(args):
    """Addestra il modello."""
    print("Inizio training...\n")
    train = _import_train()
    tb_dir = getattr(args, "tensorboard_dir", "runs/pixelart_train")
    if getattr(args, "no_tensorboard", False):
        tb_dir = None
    train(
        loss_preset=args.preset,
        num_epochs=args.epochs,
        cross_ref_prob=getattr(args, "cross_ref_prob", 0.22),
        cross_pose_prob=getattr(args, "cross_pose_prob", 0.22),
        resume=getattr(args, "resume", None),
        early_stopping_patience=getattr(args, "early_stopping_patience", 12),
        early_stopping_min_delta=getattr(args, "early_stopping_min_delta", 0.0),
        preview_every=getattr(args, "preview_every", 5),
        checkpoint_backup_every=getattr(args, "checkpoint_backup_every", 10),
        plateau_patience=getattr(args, "plateau_patience", 5),
        csv_log_path=getattr(args, "csv_log", "checkpoints/training_metrics.csv"),
        tensorboard_dir=tb_dir,
    )


def cmd_train_vqvae(args):
    train_vqvae = _import_train_vqvae()
    train_vqvae(
        split_dir=args.split_dir,
        palette_path=args.palette,
        rebuild_palette=args.rebuild_palette,
        num_colors=args.num_colors,
        num_embeddings=args.num_embeddings,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
    )


def cmd_train_prior(args):
    train_prior = _import_train_prior()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_prior(
        vqvae_ckpt=args.vqvae_ckpt,
        palette_path=args.palette,
        split_dir=args.split_dir,
        num_layers=args.layers,
        n_head=args.n_head,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
    )


def cmd_generate_vqvae(args):
    gen = _import_generate_vqvae()
    gen(
        out_path=args.out,
        vqvae_ckpt=args.vqvae_ckpt,
        prior_ckpt=args.prior_ckpt,
        prior_meta=args.prior_meta,
        palette_path=args.palette,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


def cmd_generate(args):
    """Genera sprite da un checkpoint."""
    print("Generazione sprite...\n")
    generate_main = _import_generate_main()
    use_random = getattr(args, "random_input", False) and not args.input
    motion_from = getattr(args, "motion_from", None)
    if getattr(args, "poses_from_input", False):
        if motion_from is not None:
            print(
                "[WARN] --poses-from-input ignorato: e' gia' impostato --motion-from.\n"
            )
        elif args.input:
            motion_from = args.input
            print(
                "[OK] Pose temporali dalla stessa strip di --input "
                f"(ref = colonna {args.input_frame}).\n"
            )
        else:
            print(
                "[WARN] --poses-from-input richiede --input. "
                "Genero senza motion-from.\n"
            )
    if getattr(args, "no_palette_json", False):
        palette_json = None
    else:
        palette_json = getattr(args, "palette_json", None)
        if palette_json is None and Path("configs/palette.json").is_file():
            palette_json = "configs/palette.json"

    generate_main(
        checkpoint=args.checkpoint,
        source_image=args.input,
        frame_count=args.frame_count,
        animation=args.animation,
        input_frame=args.input_frame,
        chain_frames=not getattr(args, "no_chain_frames", False),
        motion_from=motion_from,
        enhance_output=not getattr(args, "no_enhance", False),
        palette_cap=not getattr(args, "no_palette_cap", False),
        palette_levels=getattr(args, "palette_levels", 4),
        palette_json=palette_json,
        alpha_hard=not getattr(args, "soft_alpha", False),
        random_source=use_random,
    )


def main():
    ensure_directories()

    parser = argparse.ArgumentParser(
        description="AI Animation Dev - Generatore Sprite Pixel Art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Comandi disponibili:

  python run.py catalog    Cataloga gli asset dal pack
  python run.py build      Costruisce il dataset
  python run.py train      Addestra il modello
  python run.py generate   Genera sprite

Flusso di lavoro consigliato:
  1. python run.py catalog
  2. python run.py build
  3. python run.py train
  4. python run.py generate
     Con movimento da strip (pose che scorre) + ref stile tuo:
       python run.py generate --input mostro.png --motion-from data/raw/licensed/.../idle.png --animation idle
     Stessa strip per ref (colonna --input-frame) e per le pose:
       python run.py generate --input data/raw/licensed/.../Receptacle_idle.png --poses-from-input --animation idle

  Pipeline VQ-VAE + Prior (pixel categorici, separata dall'UNet):
    python run.py train-vqvae
    python run.py train-prior
    python run.py generate-vqvae
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Comando: catalog
    subparsers.add_parser(
        "catalog",
        help="Cataloga gli asset",
    )

    # Comando: build
    subparsers.add_parser(
        "build",
        help="Costruisce il dataset",
    )

    # Comando: train
    train_parser = subparsers.add_parser(
        "train",
        help="Addestra il modello",
    )
    train_parser.add_argument(
        "--preset",
        default="multi_author_sharp",
        choices=[
            "multi_author_balanced",
            "multi_author_strict",
            "multi_author_permissive",
            "multi_author_sharp",
            "single_author",
        ],
        help="Preset loss per training multi-autore",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero di epoch di training",
    )
    train_parser.add_argument(
        "--cross-ref-prob",
        type=float,
        default=0.22,
        help="Train: prob. ref RGB da altro clip (stessa animazione). Val: sempre 0.",
    )
    train_parser.add_argument(
        "--cross-pose-prob",
        type=float,
        default=0.22,
        help="Train: prob. pose da altro clip (stessa animazione). Val: sempre 0.",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        metavar="PTH",
        help="Finetune: carica pesi da questo .pth (es. checkpoints/model_final.pth).",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop se val_loss non migliora per N epoche (0 = off).",
    )
    train_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Miglioramento minimo val_loss per resettare il contatore early stopping.",
    )
    train_parser.add_argument(
        "--preview-every",
        type=int,
        default=5,
        help="Ogni N epoche: preview GT|pred in outputs/previews/ (0 = off).",
    )
    train_parser.add_argument(
        "--checkpoint-backup-every",
        type=int,
        default=10,
        help="Backup checkpoints/checkpoint_epoch_N.pth ogni N epoche (0 = off).",
    )
    train_parser.add_argument(
        "--plateau-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau su val_loss; 0 = solo cosine.",
    )
    train_parser.add_argument(
        "--csv-log",
        default="checkpoints/training_metrics.csv",
        help="File CSV append-only con loss/ssim per epoca.",
    )
    train_parser.add_argument(
        "--tensorboard-dir",
        default="runs/pixelart_train",
        help="Directory log TensorBoard.",
    )
    train_parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Non scrivere log TensorBoard.",
    )

    # Comando: generate
    generate_parser = subparsers.add_parser(
        "generate",
        help="Genera sprite",
    )
    generate_parser.add_argument(
        "--checkpoint",
        default="checkpoints/model_final.pth",
        help="Checkpoint modello da usare in inferenza",
    )
    generate_parser.add_argument(
        "--input",
        default=None,
        help="PNG sorgente opzionale usato come input di generazione",
    )
    generate_parser.add_argument(
        "--random-input",
        action="store_true",
        help="Sceglie un asset casuale da data/raw/licensed (manifest): evita ref grigio senza --input",
    )
    generate_parser.add_argument(
        "--frame-count",
        type=int,
        default=8,
        help="Numero frame da generare",
    )
    generate_parser.add_argument(
        "--animation",
        default="idle",
        help="Condizionamento + metadata: idle, walk, run, attack, death, jump. "
        "Con --random-input viene sostituita dall'animazione del manifest/nome file.",
    )
    generate_parser.add_argument(
        "--input-frame",
        type=int,
        default=0,
        metavar="N",
        help="Se --input è una spritesheet orizzontale, indice del frame/colonna da usare (0=primo)",
    )
    generate_parser.add_argument(
        "--no-chain-frames",
        action="store_true",
        help="Disattiva la catena: stesso RGB sorgente per ogni frame (spesso tutte le celle uguali)",
    )
    generate_parser.add_argument(
        "--motion-from",
        default=None,
        metavar="PNG",
        help="Spritesheet orizzontale: sequenza di pose (silhouette per colonna); --input = ref RGB (aspetto)",
    )
    generate_parser.add_argument(
        "--poses-from-input",
        action="store_true",
        help="Usa la stessa immagine di --input come strip delle pose (ref resta la colonna --input-frame). "
        "Equivalente a --motion-from uguale a --input.",
    )
    generate_parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disattiva nitidezza/contrasto post-processing sui frame generati",
    )
    generate_parser.add_argument(
        "--no-palette-cap",
        action="store_true",
        help="Disattiva quantizzazione RGB finale; utile per debug (milioni di colori)",
    )
    generate_parser.add_argument(
        "--palette-levels",
        type=int,
        default=4,
        metavar="L",
        help="Livelli per canale RGB dopo generazione (L^3 colori max in griglia). "
        "3=27, 4=64 (default), 5=125. Ignorato con --no-palette-cap.",
    )
    generate_parser.add_argument(
        "--palette-json",
        default=None,
        metavar="JSON",
        help="Snap RGB al colore piu' vicino in questo palette.json (es. configs/palette.json). "
        "Se omesso e il file esiste, viene usato automaticamente.",
    )
    generate_parser.add_argument(
        "--no-palette-json",
        action="store_true",
        help="Non usare configs/palette.json neanche se presente (solo griglia L^3 o niente).",
    )
    generate_parser.add_argument(
        "--soft-alpha",
        action="store_true",
        help="Mantieni alpha continua (disattiva binarizzazione anti-smearing).",
    )
    generate_parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.5,
        help="Alpha hard: con mode=relative e' frazione del max alpha nel frame (0.5=meta'); "
        "con mode=absolute e' soglia su 0-255 scalata. Ignorato con --soft-alpha.",
    )
    generate_parser.add_argument(
        "--alpha-hard-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="relative: taglio rispetto al picco alpha (evita output tutto trasparente). "
        "absolute: soglia fissa come prima.",
    )

    vqvae_parser = subparsers.add_parser(
        "train-vqvae",
        help="Addestra VQ-VAE (palette categorica + codebook latente)",
    )
    vqvae_parser.add_argument("--split-dir", default="data/final/train")
    vqvae_parser.add_argument("--palette", default="configs/palette.json")
    vqvae_parser.add_argument(
        "--rebuild-palette",
        action="store_true",
        help="Ricalcola K-means anche se esiste configs/palette.json",
    )
    vqvae_parser.add_argument("--num-colors", type=int, default=64)
    vqvae_parser.add_argument("--num-embeddings", type=int, default=512)
    vqvae_parser.add_argument("--latent-dim", type=int, default=256)
    vqvae_parser.add_argument("--batch-size", type=int, default=8)
    vqvae_parser.add_argument("--epochs", type=int, default=80)
    vqvae_parser.add_argument("--lr", type=float, default=3e-4)
    vqvae_parser.add_argument("--image-size", type=int, default=256)

    prior_parser = subparsers.add_parser(
        "train-prior",
        help="Estrae codici VQ e addestra Transformer autoregressivo",
    )
    prior_parser.add_argument("--vqvae-ckpt", default="checkpoints/vqvae_final.pth")
    prior_parser.add_argument("--palette", default="configs/palette.json")
    prior_parser.add_argument("--split-dir", default="data/final/train")
    prior_parser.add_argument("--layers", type=int, default=6)
    prior_parser.add_argument("--n-head", type=int, default=8)
    prior_parser.add_argument("--n-embd", type=int, default=256)
    prior_parser.add_argument("--batch-size", type=int, default=64)
    prior_parser.add_argument("--epochs", type=int, default=100)
    prior_parser.add_argument("--lr", type=float, default=3e-4)
    prior_parser.add_argument("--image-size", type=int, default=256)

    gvqvae_parser = subparsers.add_parser(
        "generate-vqvae",
        help="Campiona dal Prior e decodifica con VQ-VAE",
    )
    gvqvae_parser.add_argument("--out", default="outputs/vqvae_prior_sample.png")
    gvqvae_parser.add_argument("--vqvae-ckpt", default="checkpoints/vqvae_final.pth")
    gvqvae_parser.add_argument("--prior-ckpt", default="checkpoints/prior_final.pth")
    gvqvae_parser.add_argument("--prior-meta", default="checkpoints/prior_meta.pt")
    gvqvae_parser.add_argument("--palette", default="configs/palette.json")
    gvqvae_parser.add_argument("--temperature", type=float, default=0.9)
    gvqvae_parser.add_argument("--top-k", type=int, default=50)
    gvqvae_parser.add_argument("--top-p", type=float, default=0.92)

    args = parser.parse_args()

    commands = {
        "catalog": cmd_catalog,
        "build": cmd_build,
        "train": cmd_train,
        "generate": cmd_generate,
        "train-vqvae": cmd_train_vqvae,
        "train-prior": cmd_train_prior,
        "generate-vqvae": cmd_generate_vqvae,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
