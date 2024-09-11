import os
import torch


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "C3_SL_checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

