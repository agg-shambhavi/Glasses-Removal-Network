import torch
import torch.nn as nn
import time
import torchvision.models.vgg as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_CYCLE = 10


def getVggFeatures(input, i):
    # function to get vgg features
    # phi according to research paper
    # i means the same as that of the paper
    layer_dict = {1: 5, 2: 10, 3: 17, 4: 24, 5: 30}
    vgg16 = models.vgg16(pretrained=True)
    output = vgg16.features[: layer_dict[i]](input)
    return output


def pcclT1(cycle_specs, img_glasses, l1):
    # need to convert b/w into 3 channels
    batch_size, c, w, h = cycle_specs.shape
    cycle_specs = cycle_specs.repeat(batch_size, 3, w, h)
    img_glasses = img_glasses.repeat(batch_size, 3, w, h)
    out = 0
    for i in range(1, 6):
        phi_gened = getVggFeatures(cycle_specs, i)
        phi_og = getVggFeatures(img_glasses, i)
        out += l1(phi_gened - phi_og)
    # i dont know if out has to be divided by 5 or not
    return out


def pcclT2(cycle_nospecs, img_noglasses, l1):
    # need to convert b/w into 3 channels
    batch_size, c, w, h = cycle_nospecs.shape
    out = 0
    cycle_nospecs = cycle_nospecs.repeat(batch_size, 3, w, h)
    img_noglasses = img_noglasses.repeat(batch_size, 3, w, h)
    out = 0
    for i in range(1, 6):
        phi_gened = getVggFeatures(cycle_nospecs, i)
        phi_og = getVggFeatures(img_noglasses, i)
        out += l1(phi_gened - phi_og)
    # i dont know if out has to be divided by 5 or not
    return out


def trainFn(
    disc_G, disc_N, gen_G, gen_N, loader, optim_g, optim_d, l1, mse, val_loader, epoch
):
    # loop = tqdm(loader, leave=True, position=0)

    for idx, (img_glasses, img_noglasses) in enumerate(loader):
        img_glasses = img_glasses.to(DEVICE)
        img_noglasses = img_noglasses.to(DEVICE)

        # Train Discriminators
        start_time = time.time()

        fake_nospecs = gen_N(img_glasses)
        disc_ns_real = disc_N(img_noglasses)
        disc_ns_fake = disc_N(fake_nospecs.detach())
        disc_ns_real_loss = mse(disc_ns_real, torch.ones_like(disc_ns_real))
        disc_ns_fake_loss = mse(disc_ns_fake, torch.zeros_like(disc_ns_fake))
        disc_ns_loss = disc_ns_real_loss + disc_ns_fake_loss

        fake_specs = gen_G(img_noglasses)
        disc_s_real = disc_G(img_glasses)
        disc_s_fake = disc_G(fake_specs.detach())
        disc_s_real_loss = mse(disc_s_real, torch.ones_like(disc_s_real))
        disc_s_fake_loss = mse(disc_s_fake, torch.zeros_like(disc_s_fake))
        disc_s_loss = disc_s_real_loss + disc_s_fake_loss

        end_time = time.time()
        print(f"{end_time - start_time} Discriminators training")

        # grand total
        D_loss = (disc_ns_loss + disc_s_loss) / 2

        # backprop and update weights of discriminator

        start_time = time.time()

        optim_d.zero_grad()
        D_loss.backward()
        optim_d.step()

        end_time = time.time()
        print(f"{end_time - start_time} backprop discriminator")

        # Train Generators
        # adversarial loss

        start_time = time.time()

        disc_ns_fake = disc_N(fake_nospecs)
        disc_s_fake = disc_G(fake_specs)
        loss_g_s = mse(disc_s_fake, torch.ones_like(disc_s_fake))
        loss_g_ns = mse(disc_ns_fake, torch.ones_like(disc_ns_fake))

        end_time = time.time()
        print(f"{end_time - start_time} adversarial loss")

        # cycle loss

        start_time = time.time()

        cycle_specs = gen_G(fake_nospecs)
        cycle_nospecs = gen_N(fake_specs)
        specs_cycle_loss = l1(img_glasses, cycle_specs)
        nospecs_cycle_loss = l1(img_noglasses, cycle_nospecs)

        end_time = time.time()
        print(f"{end_time - start_time} cycle loss")

        ## Perceptual Loss
        loss_pccl = pcclT1(cycle_specs, img_glasses, l1) + pcclT2(
            cycle_nospecs, img_noglasses, l1
        )

        # grand total
        G_loss = (
            loss_g_s
            + loss_g_ns
            + (specs_cycle_loss * LAMBDA_CYCLE)
            + (nospecs_cycle_loss * LAMBDA_CYCLE)
            + loss_pccl
        )

        # backprop and update weights of discriminator

        start_time = time.time()

        optim_g.zero_grad()
        G_loss.backward()
        optim_g.step()

        end_time = time.time()
        print(f"{end_time - start_time} generator backprop")

        # if idx == 2000:
        #   print(f"Generator Loss: {G_loss} Discriminator Loss: {D_loss}")

        # if idx % 2 == 0:
        print(f"Generator Loss: {G_loss} Discriminator Loss: {D_loss}")

        # save_some_examples_g(gen_G, val_loader, epoch, idx, folder=resultGlasses)
        # save_some_examples_n(gen_N, val_loader, epoch, idx, folder=resultsNoGlasses)
