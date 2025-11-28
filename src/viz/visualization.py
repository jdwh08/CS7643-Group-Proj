import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(net, test_loader, sample_idx = None):
    if sample_idx is None:
        sample_idx = np.random.randint(len(test_loader.dataset))
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    plt.title('Hand-Labeled Mask')
    target = test_loader.dataset[sample_idx][1]
    true_mask = target == 1
    target = true_mask * 1
    plt.imshow(target, cmap = 'Reds')
    with torch.no_grad():
        preds = trained_net(torch.unsqueeze(test_loader.dataset[sample_idx][0], dim = 0).to(device))
        pred_mask = np.argmax(preds['out'].cpu().numpy()[0], axis = 0)
    plt.figure()
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap = 'Reds')