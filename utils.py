import torch
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, folder + f"{epoch}_fake")
        save_image(x, folder + f"{epoch}_glasses")
        save_image(y, folder + f"{epoch}_no_glasses")
    gen.train()


def save_checkpoint(model, optimizer, epoch, filename):
    filename = str(epoch) + filename + "_cpt.pth.tar"
    print("=> Saving Checkpoint")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state(checkpoint["state_dict"])
    optimizer.load_state(checkpoint["state_dict"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
