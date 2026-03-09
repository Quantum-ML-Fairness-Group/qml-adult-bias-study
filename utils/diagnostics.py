import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_results(history_loss, history_acc, test_outputs, y_test, sens_test, bias_metrics):

    
    steps = np.arange(1, len(history_loss) + 1)
    
    if isinstance(test_outputs, torch.Tensor):
        test_outputs = test_outputs.detach().cpu()
        if test_outputs.max() > 1.0 or test_outputs.min() < 0.0:
            test_probs = torch.sigmoid(test_outputs).numpy().flatten()
        else:
            test_probs = test_outputs.numpy().flatten()
    else:
        test_probs = test_outputs.flatten()
        
    y_test = np.array(y_test).flatten()
    sens_test = np.array(sens_test).flatten()

    def smooth(y, w=100):
        if len(y) < w: return y
        return np.convolve(y, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # --- Training Loss ---
    axes[0, 0].plot(steps, history_loss, alpha=0.3, color="tab:blue", label="Raw")
    if len(history_loss) > 20:
        smoothed_loss = smooth(history_loss)
        axes[0, 0].plot(steps[len(steps)-len(smoothed_loss):], smoothed_loss, color="tab:blue", linewidth=2, label="Smoothed")
    axes[0, 0].set_title("Training Loss (Per Batch)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- Training Accuracy ---
    axes[0, 1].plot(steps, history_acc, alpha=0.3, color="tab:purple", label="Raw")
    if len(history_acc) > 20:
        smoothed_acc = smooth(history_acc)
        axes[0, 1].plot(steps[len(steps)-len(smoothed_acc):], smoothed_acc, color="tab:purple", linewidth=2, label="Smoothed")
    axes[0, 1].set_title("Training Accuracy (Per Batch)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # --- Prediction Distribution (by Truth) ---
    axes[0, 2].hist(test_probs[y_test==0], bins=30, alpha=0.5, label='Actual <=50k', color='tab:red', density=True)
    axes[0, 2].hist(test_probs[y_test==1], bins=30, alpha=0.5, label='Actual >50k', color='tab:green', density=True)
    axes[0, 2].set_title("Pred Probability Dist (By Label)", fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel("Probability Score")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # --- Bias Metrics (Bar Chart) ---
    metrics_list = ['DPD', 'TPR Diff', 'FPR Diff', 'EOD']
    values = [bias_metrics.get('dpd', 0), bias_metrics.get('tpr_diff', 0), 
              bias_metrics.get('fpr_diff', 0), bias_metrics.get('eod', 0)]
    colors = ['tab:orange', 'tab:cyan', 'tab:pink', 'tab:olive']
    
    axes[1, 0].bar(metrics_list, values, color=colors, alpha=0.8)
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].set_title("Fairness Metrics (Closer to 0 is better)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(min(min(values), -0.1) - 0.1, max(max(values), 0.1) + 0.1)
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # --- Score Distribution (by Sensitive Attribute) ---
    axes[1, 1].hist(test_probs[sens_test==0], bins=30, alpha=0.5, label='Group 0 (Female)', color='magenta', density=True)
    axes[1, 1].hist(test_probs[sens_test==1], bins=30, alpha=0.5, label='Group 1 (Male)', color='blue', density=True)
    axes[1, 1].set_title("Score Distribution (By Sex)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Probability Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # --- Summary Text ---
    axes[1, 2].axis('off')
    final_loss = np.mean(history_loss[-50:]) if len(history_loss) > 0 else 0
    final_acc = np.mean(history_acc[-50:]) if len(history_acc) > 0 else 0
    test_acc_calc = np.mean((test_probs > 0.5) == y_test)
    
    summary_text = (
        f" Summary Stats:\n\n"
        f"Train Steps: {len(history_loss)}\n"
        f"Final Train Loss: {final_loss:.4f}\n"
        f"Final Train Acc:  {final_acc:.4f}\n"
        f"---------------------------\n"
        f"Test Accuracy:    {test_acc_calc:.4f}\n"
        f"---------------------------\n"
        f"Bias (Fairness):\n"
        f"DPD: {values[0]:.4f}\n"
        f"EOD: {values[3]:.4f}"
    )
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=13, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.show()

# ==========================================
# Example usage in training loop (after testing and bias evaluation):
# ==========================================

# dpd, eod, tpr_diff, fpr_diff = ... 

# bias_dict = {
#     'dpd': dpd,
#     'eod': eod,
#     'tpr_diff': tpr_diff,
#     'fpr_diff': fpr_diff
# }

# visualize_results(
#     history_loss=history_loss, 
#     history_acc=history_acc, 
#     test_outputs=test_outputs, 
#     y_test=y_test_numpy, 
#     sens_test=sens_test_t, 
#     bias_metrics=bias_dict
# )