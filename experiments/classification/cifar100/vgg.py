from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import VGGBaseline
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.optim_recipes import get_procedure

if __name__ == "__main__":
    args = init_args(VGGBaseline, CIFAR100DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    if args.exp_name == "":
        args.exp_name = f"{args.version}-vgg{args.arch}-cifar100"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR100DataModule(**vars(args))

    # model
    model = VGGBaseline(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optim_recipe=get_procedure(f"vgg{args.arch}", "cifar100", args.version),
        style="cifar",
        **vars(args),
    )

    cli_main(model, dm, args.exp_dir, args.exp_name, args)
