from models.cnn_rnn import CNN_RNN_Model
from models.cnn import CNN
from models.vit import ViT

from models.mamba import cata_mamba
from models.mamba_v2 import cata_mamba_v2
from models.mamba_fc import cata_mamba_fc
from timesformer.models.vit import TimeSformer
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2, Tool_Recognition
from utils import *
import time

# LOGGING
import wandb
from calflops import calculate_flops

from prettytable import PrettyTable
import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Training Script")

# Define command-line arguments
# TODO: Load them from a config.json file later on
parser.add_argument(
    "--root_dir",
    type=str,
    help="Path containing the downloaded dataset folder Cataracts_Multitask",
    default="/l/users/dana.mohamed",
)
parser.add_argument(
    "--json_path",
    type=str,
    help="Path to the json file containing the dataset labels",
    default="/l/users/dana.mohamed/Cataracts_Multitask/labels_dataset_level.json",
)
parser.add_argument(
    "--path_pretrained_TimeSformer",
    type=str,
    help="Path to the pretrained TimeSformer model",
    default="/l/users/dana.mohamed/TimeSformer_divST_96x32_224_HowTo100M.pyth",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to save and load the model checkpoints",
    default="/l/users/dana.mohamed/checkpoints/",
)

parser.add_argument(
    "--qualitative_results_path",
    type=str,
    help="Path to save the qualitative results, including ribbon figure and confusion matrices",
    default="qualitative_results/",
)
parser.add_argument(
    "--num_classes", type=int, help="Number of classes, default=10", default=21
)
parser.add_argument(
    "--num_clips", type=int, help="Number of clips to sample from each video", default=8
)
parser.add_argument(
    "--clip_size", type=int, help="Number of frames in each clip", default=20
)
parser.add_argument(
    "--step_size",
    type=int,
    help="Number of frames to skip when sampling clips",
    default=2,
)

parser.add_argument(
    "--learning_rate", type=float, help="Learning rate for the optimizer", default=1e-3
)
parser.add_argument(
    "--scheduler",
    choices=["None", "StepLR", "Cosine"],
    help="Whether to use a scheduler for the optimizer",
    default="None",
)
parser.add_argument(
    "--momentum",
    type=float,
    help="Momentum for the optimizer (SGD only)",
    default=0.0,
)
parser.add_argument(
    "--clip-grad-norm",
    action="store_false",
    help="Clip the gradient norm to prevent exploding gradients. Default is True",
)
######################### Mamba specific parameters ############################
parser.add_argument(
    "--d_state", type=int, help="SSM state expansion factor", default=16
)
parser.add_argument("--d_conv", type=int, help="Local convolution width", default=4)
parser.add_argument("--expand", type=int, help="Block expansion factor", default=2)
parser.add_argument(
    "--mamba_num_blocks", type=int, help="Number of Mamba blocks", default=2
)
parser.add_argument(
    "--dilation_levels",
    type=int,
    help="Number of dilation levels in the 1D Conv in Cata-Mamba",
    default=3,
)

################################################################################
parser.add_argument(
    "--epochs", type=int, help="Number of epochs for training the model", default=50
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size for training the model", default=2
)
parser.add_argument(
    "--hidden_size", type=int, help="Hidden size for the RNN", default=512
)
parser.add_argument(
    "--loss_function",
    type=str,
    help="Loss function to use for training the model",
    default="CrossEntropyLoss",
)
parser.add_argument(
    "--weighted_loss",
    action="store_true",
    help="Whether to use weighted loss for the classes. Default is False.",
)
parser.add_argument(
    "--label_smoothing",
    type=float,
    help="Label smoothing factor for the loss function",
    default=0.0,
)
parser.add_argument(
    "--optimizer",
    type=str,
    help="Optimizer to use for training the model",
    default="Adam",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    help="Weight decay for the optimizer",
    default=1e-2,  # weight_decay 0.3 for ViT
)

parser.add_argument(
    "--task", type=str, help="Task to train the model on", default="Phase_detection"
)
parser.add_argument(
    "--dataset",
    choices=["1_Cataracts-21", "2_Cataracts-101", "9_CATARACTS"],
    help="Dataset to train the model on",
    default="2_Cataracts-101",
)
parser.add_argument(
    "--architecture",
    choices=["CNN_RNN", "CNN", "ViT", "Cata-Mamba", "Cata-Mamba-v2", "Cata-Mamba-fc", "TimeSformer"],
    help="Model to use for training",
    default="CNN_RNN",
)
parser.add_argument(
    "--cnn_model",
    choices=["resnet18", "resnet50", "resnet101"],
    help="CNN model to use for training",
    default="resnet50",
)
parser.add_argument(
    "--rnn_model",
    choices=["lstm", "gru"],
    help="RNN model to use for training",
    default="lstm",
)
parser.add_argument(
    "--num_layers_rnn", type=int, help="Number of layers for the RNN", default=1
)
parser.add_argument(
    "--bidirectional",
    action="store_true",
    help="Whether to use a bidirectional RNN. Default is False.",
)

parser.add_argument(
    "--resume_training",
    action="store_true",
    help="Whether to resume training from a checkpoint. Default is False.",
)

parser.add_argument(
    "--wandb_run_id",
    type=str,
    help="Used when resuming training for a specific model to resume the wandb run",
    default=None,
)
parser.add_argument(
    "--run_name",
    type=str,
    help="Name of the run to be displayed on Wandb",
    default=None,
)

parser.add_argument(
    "--seed",
    type=int,
    help="Seed for reproducibility",
    default=0,
)
parser.add_argument(
    "--create_qualitative_results",
    action="store_false",
    help="Whether to create qualitative results. Default is True.",
)
# If the flag is present in the command line, 
# log_results will be False; if the flag is omitted, it defaults to True.
parser.add_argument(
    "--log_results",
    action="store_false",
    help="Whether to log the results on Wandb. Default is True",
)
# Parse the command-line arguments
args = parser.parse_args()

# Set the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Set the device to GPU if available
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # check if NVIDIA device is visible to torch

# Parameters
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
hidden_size = args.hidden_size
bidirectional = args.bidirectional
num_layers_rnn = args.num_layers_rnn
cnn_model = args.cnn_model
rnn_model = args.rnn_model

# Directories
root_dir = args.root_dir
json_path = args.json_path
checkpoint_path = args.checkpoint_path
qualitative_results_path = args.qualitative_results_path


# Create a the directory to save the checkpoints
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    print("[INFO] created checkpoints directory")

if not os.path.exists(qualitative_results_path):
    os.makedirs(qualitative_results_path)
    print("[INFO] created qualitative results directory")

# Data related args
dataset = args.dataset
task = args.task
num_clips = args.num_clips
clip_size = args.clip_size
step_size = args.step_size

architecture = args.architecture
architectures = {
    "CNN_RNN": CNN_RNN_Model(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers_rnn,
        cnn=cnn_model,
        rnn=rnn_model,
        bidirectional=bidirectional,
    ),
    "CNN": CNN(cnn=cnn_model, num_classes=num_classes),
    "ViT": ViT(num_classes=num_classes),
    "Cata-Mamba": cata_mamba(
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_classes=num_classes,
        N=args.mamba_num_blocks,
        dilation_levels=args.dilation_levels,
        feature_extractor=cnn_model,
    ),
    "TimeSformer": TimeSformer(
        img_size=224,
        num_classes=num_classes,
        num_frames=96,
        attention_type="divided_space_time",
        pretrained_model=args.path_pretrained_TimeSformer,
    ),
    "Cata-Mamba-v2": cata_mamba_v2(
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_classes=num_classes,
        N=args.mamba_num_blocks,
        dilation_levels=args.dilation_levels,
        feature_extractor=cnn_model,
    ),
    "Cata-Mamba-fc": cata_mamba_fc(
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_classes=num_classes,
        N=args.mamba_num_blocks,
        feature_extractor=cnn_model,
    )
}
model = architectures[architecture]
model.to(DEVICE)

if args.loss_function == "CrossEntropyLoss":
    if args.weighted_loss and dataset == "2_Cataracts-101":
        class_weights = np.asarray(
            [
                0.21870906927865919,
                0.0480576346467981,
                0.20794132637585014,
                0.060416393951572174,
                0.07974307794890623,
                0.018958230588380327,
                0.045311303795470353,
                0.1879251900960704,
                0.09796812663905301,
                0.03496964667924019,
            ],
            dtype=np.float32,
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).float().to(DEVICE),
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

optimizers = {
    "Adam": optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    ),
    "SGD": optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    ),
    "AdamW": optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    ),
}
optimizer = optimizers[args.optimizer]

# Define the scheduler
schedulers = {
    "None": None,
    "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
    "Cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
}
scheduler = schedulers[args.scheduler]

if args.clip_grad_norm:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Logging
log_results = args.log_results

print(f"[INFO] Logging results: {log_results}")
# ####################################################################################################################################
if log_results:
    project_name = args.architecture
    if args.architecture == "Cata-Mamba-fc":
        project_name = "Cata-Mamba-v2"
    run_id = None
else:
    #generate random integer
    run_id = np.random.randint(0, 1000000)

# Add Wandb logging
if args.resume_training and args.wandb_run_id is not None:
    run_id = args.wandb_run_id
    wandb.init(project=project_name, id=run_id, resume="allow")

elif log_results:  # First time training
    if args.run_name:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=args.run_name,
            # track hyperparameters and run metadata
            config=args,
        )
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # track hyperparameters and run metadata
            config=args,
        )
    run_id = wandb.run.id
    print(f"[INFO] Run ID: {run_id}")
else:
    print("[INFO] Logging is disabled")

# number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(
    "[INFO] number of parameters in the model: {}".format(num_params)
    )

if log_results:
    wandb.log({"num_model_params": num_params})

# Transforms
transform = v2.Compose(
    [
        v2.Resize((250, 250)),  # Example resize, adjust as needed
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(5),
    ]
)

train_dataset = Tool_Recognition(
    root_dir,
    json_path,
    # dataset_name=dataset,
    split="Train",
    num_classes=num_classes,
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    replace_value=1,
    transform=transform,
)

val_dataset = Tool_Recognition(
    root_dir,
    json_path,
    # dataset_name=dataset,
    split="Validation",
    num_classes=num_classes,
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    replace_value=1,
    transform=transform,
)

test_dataset = Tool_Recognition(
    root_dir,
    json_path,
    # dataset_name=dataset,
    split="Test",
    num_classes=num_classes,
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    replace_value=1,
    transform=transform,
)

# Create a dataloader
if num_clips == -1:
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_shape = (batch_size, num_clips*clip_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("%s FLOPs:%s   MACs:%s   Params:%s \n" %(architecture, flops, macs, params))
if log_results:
    wandb.log({"FLOPs": flops})
    wandb.log({"MACs": macs})
    wandb.log({"Params": params})

best_val_acc = 0  # to keep track of best validation accuracy
best_val_epoch = -1
best_metrics = {}

# Capture the start time
start_time = time.time()

if args.resume_training:
    # Load the model
    ckpt = load_checkpoint(checkpoint_path + f"last_epoch_{run_id}.pth")

    start_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"[INFO] Resuming training from epoch {start_epoch+1}")
else:
    start_epoch = 0

for epoch in range(start_epoch, epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch + 1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    if log_results:
        wandb.log({"train_loss": loss})

    # acc = validate(model, val_loader, DEVICE)
    metrics = validate(model, val_loader, DEVICE)

    # Update the learning rate
    if scheduler:
        scheduler.step()

    acc = metrics["accuracy"]
    if log_results:
        wandb.log({"val_accuracy": acc})
        wandb.log({"val_precision": metrics["precision_micro"]})
        wandb.log({"val_recall": metrics["recall_micro"]})
        wandb.log({"val_f1_score": metrics["f1_score_micro"]})

    print(
        f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}"
    )
    save_checkpoint(
        model, optimizer, epoch, checkpoint_path + f"last_epoch_{run_id}.pth", scheduler
    )  # save checkpoint after each epoch
    if acc > best_val_acc:
        best_val_acc = acc
        best_metrics = metrics
        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_path + f"best_model_{run_id}.pth",
            scheduler,
            best=True,
        )
        best_val_epoch = epoch

# Capture the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time

# Log the total time taken
if log_results:
    wandb.log({"total_time": total_time})

# Print total time taken
print(f"[INFO] Total time taken: {total_time:.2f} seconds")

# print(f"[INFO] Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch+1}")
print("[INFO] Training complete")


# Unwanted scenario
if len(best_metrics) == 0:
    print("No best metrics found, using the last metrics")
    best_metrics = metrics

    # Get last checkpoint to test the model
    ckpt = load_checkpoint(checkpoint_path + f"last_epoch_{run_id}.pth")
else:
    # Get the best validation checkpoint to test the model
    ckpt = load_checkpoint(checkpoint_path + f"best_model_{run_id}.pth")


model.load_state_dict(ckpt["model_state_dict"])
per_class_metrics = False
test_metrics = validate(model, test_loader, DEVICE, per_class_metrics=per_class_metrics, inference_rate=True)

# Create a table with the results
print("VALIDATION SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", best_metrics["accuracy"]])
results.add_row(["Precision_micro", best_metrics["precision_micro"]])
results.add_row(["Recall_micro", best_metrics["recall_micro"]])
results.add_row(["F1-Score_micro", best_metrics["f1_score_micro"]])
results.add_row(["Jaccard_micro", best_metrics["jaccard_micro"]])
results.add_row(["Precision_macro", best_metrics["precision_macro"]])
results.add_row(["Recall_macro", best_metrics["recall_macro"]])
results.add_row(["F1-Score_macro", best_metrics["f1_score_macro"]])
results.add_row(["Jaccard_macro", best_metrics["jaccard_macro"]])
results.add_row(["Precision_weighted", best_metrics["precision_weighted"]])
results.add_row(["Recall_weighted", best_metrics["recall_weighted"]])
results.add_row(["F1-Score_weighted", best_metrics["f1_score_weighted"]])
results.add_row(["Jaccard_weighted", best_metrics["jaccard_weighted"]])
print(results)

if log_results:
    # Log the validation results on Wandb - VALIDATION DATA
    table = wandb.Table(columns=["Metrics_Validation_Set", "Value"])
    table.add_data("Accuracy", best_metrics["accuracy"])
    table.add_data("Precision_micro", best_metrics["precision_micro"])
    table.add_data("Recall_micro", best_metrics["recall_micro"])
    table.add_data("F1-Score_micro", best_metrics["f1_score_micro"])
    table.add_data("Jaccard_micro", best_metrics["jaccard_micro"])
    table.add_data("Precision_macro", best_metrics["precision_macro"])
    table.add_data("Recall_macro", best_metrics["recall_macro"])
    table.add_data("F1-Score_macro", best_metrics["f1_score_macro"])
    table.add_data("Jaccard_macro", best_metrics["jaccard_macro"])
    table.add_data("Precision_weighted", best_metrics["precision_weighted"])
    table.add_data("Recall_weighted", best_metrics["recall_weighted"])
    table.add_data("F1-Score_weighted", best_metrics["f1_score_weighted"])
    table.add_data("Jaccard_weighted", best_metrics["jaccard_weighted"])
    wandb.log({"validation_results_table": table}, commit=False)


    # Log the test results on Wandb - TEST DATA
    table = wandb.Table(columns=["Metrics_Test_Set", "Value"])
    table.add_data("Inference_rate", test_metrics["inference_rate"])
    table.add_data("Accuracy", test_metrics["accuracy"])
    table.add_data("Precision_micro", test_metrics["precision_micro"])
    table.add_data("Recall_micro", test_metrics["recall_micro"])
    table.add_data("F1-Score_micro", test_metrics["f1_score_micro"])
    table.add_data("Jaccard_micro", test_metrics["jaccard_micro"])
    table.add_data("Precision_macro", test_metrics["precision_macro"])
    table.add_data("Recall_macro", test_metrics["recall_macro"])
    table.add_data("F1-Score_macro", test_metrics["f1_score_macro"])
    table.add_data("Jaccard_macro", test_metrics["jaccard_macro"])
    table.add_data("Precision_weighted", test_metrics["precision_weighted"])
    table.add_data("Recall_weighted", test_metrics["recall_weighted"])
    table.add_data("F1-Score_weighted", test_metrics["f1_score_weighted"])
    table.add_data("Jaccard_weighted", test_metrics["jaccard_weighted"])
    wandb.log({"test_results_table": table}, commit=False)


    if per_class_metrics:
        columns_per_class = ["Metric"] + [f"Phase_{i}" for i in range(num_classes)]
        table = wandb.Table(columns=columns_per_class)
        table.add_data(*(["Jaccard"] + [str(j) for j in test_metrics["jaccard_per_class"]]))
        table.add_data(
            *(["Precision"] + [str(p) for p in test_metrics["precision_per_class"]])
        )
        table.add_data(*(["Recall"] + [str(r) for r in test_metrics["recall_per_class"]]))
        table.add_data(*(["F1-Score"] + [str(f) for f in test_metrics["f1_per_class"]]))
        wandb.log({"test_results_table_per_class": table}, commit=False)

# Print the results
# Create a table with the results
fps = test_metrics["inference_rate"]
print(f"Inference rate: {fps} FPS")
print("*" * 50)
print("TEST SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", test_metrics["accuracy"]])
results.add_row(["Precision_micro", test_metrics["precision_micro"]])
results.add_row(["Recall_micro", test_metrics["recall_micro"]])
results.add_row(["F1-Score_micro", test_metrics["f1_score_micro"]])
results.add_row(["Jaccard_micro", test_metrics["jaccard_micro"]])
results.add_row(["Precision_macro", test_metrics["precision_macro"]])
results.add_row(["Recall_macro", test_metrics["recall_macro"]])
results.add_row(["F1-Score_macro", test_metrics["f1_score_macro"]])
results.add_row(["Jaccard_macro", test_metrics["jaccard_macro"]])
results.add_row(["Precision_weighted", test_metrics["precision_weighted"]])
results.add_row(["Recall_weighted", test_metrics["recall_weighted"]])
results.add_row(["F1-Score_weighted", test_metrics["f1_score_weighted"]])
results.add_row(["Jaccard_weighted", test_metrics["jaccard_weighted"]])
results.add_row(['Inference_rate', test_metrics["inference_rate"]])

print(results)

if log_results:
    print("Run ID: ", wandb.run.id)
    print("Run URL: ", wandb.run.get_url())
    wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.run.summary["test_precision_micro"] = test_metrics["precision_micro"]
    wandb.run.summary["test_recall_micro"] = test_metrics["recall_micro"]
    wandb.run.summary["test_f1_score_micro"] = test_metrics["f1_score_micro"]
    wandb.run.summary["test_jaccard_micro"] = test_metrics["jaccard_micro"]
    wandb.run.summary["test_precision_macro"] = test_metrics["precision_macro"]
    wandb.run.summary["test_recall_macro"] = test_metrics["recall_macro"]
    wandb.run.summary["test_f1_score_macro"] = test_metrics["f1_score_macro"]
    wandb.run.summary["test_jaccard_macro"] = test_metrics["jaccard_macro"]
    wandb.run.summary["test_precision_weighted"] = test_metrics["precision_weighted"]
    wandb.run.summary["test_recall_weighted"] = test_metrics["recall_weighted"]
    wandb.run.summary["test_f1_score_weighted"] = test_metrics["f1_score_weighted"]
    wandb.run.summary["test_jaccard_weighted"] = test_metrics["jaccard_weighted"]

# Save qualitative results
create_qualitative_results = args.create_qualitative_results

if create_qualitative_results:
    # # The ribbon diagram
    # qualitative_results_path_ribbon = os.path.join(
    #     qualitative_results_path, f"ribbon_diagram_{run_id}.png"
    # )
    # batch = next(iter(test_loader))
    # make_qualitative_results(
    #     {f"{architecture}": model},
    #     next(iter(test_loader)),
    #     qualitative_results_path_ribbon,
    #     num_classes,
    #     DEVICE,
    # )

    # Confusion matrix
    cf_matrix = test_metrics["confusion_matrix"]
    confusion_path = os.path.join(
        qualitative_results_path, f"confusion_matrix_{run_id}.png"
    )

    save_confusion_matrix(cf_matrix, confusion_path)
    
    if log_results:
        # wandb.log({"qualitative_results": wandb.Image(qualitative_results_path_ribbon)})
        wandb.log({"confusion_matrix": wandb.Image(confusion_path)})

if log_results:
    wandb.finish()
