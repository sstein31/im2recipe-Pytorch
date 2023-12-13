

def make_train_val_curve(learning_curves_file: str, output_dir='plots/', title=None):
    """Make a learning curve from a learning curve file."""
    df = pd.read_json(learning_curves_file)
    
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(df['epoch'], df['combined_loss'], '-o', label='train', color='tab:blue', alpha=0.7)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Combined Train Loss', color='tab:blue')


    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['val_median_recall'], '-o', label='val', color='tab:red', alpha=0.7)
    ax2.set_ylabel('Val Median Recall', color='tab:red')

    # Add a little space at the bottom so the plot doesn't look so cramped.
    buffer1 = 0.05 * (df['combined_loss'].max())
    buffer2 = 0.05 * (df['val_median_recall'].max())
    ax1.set_ylim(0 - buffer1, df['combined_loss'].max() + buffer1)
    ax2.set_ylim(0 - buffer2, df['val_median_recall'].max() + buffer2)

    plt.grid()
    if title is None:
        title = 'train_val_loss'
    plt.title(title)
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'train_val_loss.png'))
    plt.close()
