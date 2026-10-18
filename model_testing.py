import torch

import torch.nn as nn

import torch.nn.functional as F

# =========================================================

# MODEL A: STANDARD CNN

# =========================================================

# Model A uses normal convolution layers.
class ModelA(nn.Module):

    """
    Standard CNN:
    Conv3x3 -> ReLU -> MaxPool repeated three times.
    Fully connected classifier at the end.
    """

    def __init__(self, nc=10):

        super().__init__()

        self.features = nn.Sequential(

            # Input: 3 x 64 x 64

            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),  # Output: 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),  # Output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),  # Output: 128 x 8 x 8

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, nc)

        )

    def forward(self, x):

        return self.classifier(self.features(x))

# =========================================================

# DEPTHWISE SEPARABLE CONVOLUTION

# =========================================================

# This block uses fewer calculations than a standard convolution.
class DSConv(nn.Module):

    """

    Depthwise 3x3 convolution followed by pointwise 1x1

    convolution. Both use BatchNorm and ReLU6.

    """

    def __init__(self, cin, cout):

        super().__init__()

        # Depthwise convolution

        self.dw = nn.Conv2d(

            cin, cin,

            kernel_size=3,

            padding=1,

            groups=cin,

            bias=False

        )

        self.bn1 = nn.BatchNorm2d(cin)

        # Pointwise convolution

        self.pw = nn.Conv2d(

            cin, cout,

            kernel_size=1,

            bias=False

        )

        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):

        x = F.relu6(self.bn1(self.dw(x)))

        x = F.relu6(self.bn2(self.pw(x)))

        return x

# =========================================================

# MODEL B: LIGHTWEIGHT CNN

# =========================================================

# Model B is designed to be smaller and more lightweight.
class ModelB(nn.Module):

    """

    Lightweight CNN using depthwise separable convolutions

    and global average pooling.

    """

    def __init__(self, nc=10):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(

                3, 16,

                kernel_size=3,

                padding=1,

                bias=False

            ),

            nn.BatchNorm2d(16),

            nn.ReLU6()

        )

        self.body = nn.Sequential(

            DSConv(16, 32),

            nn.MaxPool2d(2),

            DSConv(32, 64),

            nn.MaxPool2d(2),

            DSConv(64, 128),

            nn.MaxPool2d(2),

            DSConv(128, 128)

        )

        # Global average pooling reduces 128 x 8 x 8 to 128.

        self.head = nn.Linear(128, nc)

    def forward(self, x):

        x = self.body(self.stem(x))

        x = x.mean(dim=(2, 3))

        return self.head(x)

# =========================================================

# LOAD THE EXISTING TRAINED MODELS

# =========================================================

# Use the GPU if it is available otherwise use the CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

modelA = ModelA(nc=10)

modelB = ModelB(nc=10)

modelA.load_state_dict(

    torch.load(

        "results/ModelA.pth",

        map_location=DEVICE,

        weights_only=True

    )

)

modelB.load_state_dict(

    torch.load(

        "results/ModelB.pth",

        map_location=DEVICE,

        weights_only=True

    )

)

modelA = modelA.to(DEVICE).eval()

modelB = modelB.to(DEVICE).eval()

print("Device:", DEVICE)

print("Existing trained models loaded successfully.")

# =========================================================

# HELPER FUNCTIONS

# =========================================================

# Count the parameters in a model or layer.
def count_parameters(module, trainable_only=False):

    """

    Count parameters in a model or layer.

    """

    return sum(

        p.numel()

        for p in module.parameters()

        if not trainable_only or p.requires_grad

    )

# Estimate how much space the model parameters use.
def model_size_kb(model):

    """

    Approximate parameter storage in KiB.

    This does not include all runtime memory requirements.

    """

    size_bytes = sum(

        p.numel() * p.element_size()

        for p in model.parameters()

    )

    return size_bytes / 1024

def conv_output_size(input_size, kernel_size, stride=1, padding=0):

    """

    Calculate output width or height for a convolution or pooling layer.

    """

    return (

        input_size + 2 * padding - kernel_size

    ) // stride + 1

# Print the main details of each layer.
def print_layer_details(model):

    """

    Display individual layers, parameter counts and trainability.

    """

    print("\nINDIVIDUAL LAYER DETAILS")

    print("-" * 100)

    print(

        f"{'Layer':<35}"

        f"{'Type':<22}"

        f"{'Parameters':>15}"

        f"{'Trainable':>15}"

    )

    print("-" * 100)

    for name, module in model.named_modules():

        # Avoid counting parent containers such as Sequential twice.

        if name == "":

            continue

        if len(list(module.children())) > 0:

            continue

        total = count_parameters(module)

        trainable = count_parameters(module, trainable_only=True)

        if isinstance(module, nn.Conv2d):

            layer_type = "Conv2d"

            details = (

                f"{name} "

                f"[{module.in_channels}->{module.out_channels}, "

                f"k={module.kernel_size}]"

            )

        elif isinstance(module, nn.Linear):

            layer_type = "Linear"

            details = (

                f"{name} "

                f"[{module.in_features}->{module.out_features}]"

            )

        elif isinstance(module, nn.BatchNorm2d):

            layer_type = "BatchNorm2d"

            details = f"{name} [{module.num_features} channels]"

        elif isinstance(module, nn.ReLU):

            layer_type = "ReLU"

            details = name

        elif isinstance(module, nn.ReLU6):

            layer_type = "ReLU6"

            details = name

        elif isinstance(module, nn.MaxPool2d):

            layer_type = "MaxPool2d"

            details = name

        elif isinstance(module, nn.Dropout):

            layer_type = "Dropout"

            details = name

        elif isinstance(module, nn.Flatten):

            layer_type = "Flatten"

            details = name

        else:

            layer_type = module.__class__.__name__

            details = name

        print(

            f"{details:<35}"

            f"{layer_type:<22}"

            f"{total:>15,}"

            f"{trainable:>15,}"

        )

# =========================================================

# PRINT CONVOLUTION PARAMETERS

# =========================================================

# Show the settings used by convolution layers.
def print_conv_details(model):

    """

    Report kernel size, filters, stride, padding, groups,

    bias usage and number of parameters for convolution layers.

    """

    print("\nCONVOLUTION LAYER CONFIGURATION")

    print("-" * 100)

    for name, layer in model.named_modules():

        if isinstance(layer, nn.Conv2d):

            print(f"\nLayer: {name}")

            print(f"  Input channels : {layer.in_channels}")

            print(f"  Output channels: {layer.out_channels}")

            print(f"  Kernel size    : {layer.kernel_size}")

            print(f"  Stride         : {layer.stride}")

            print(f"  Padding        : {layer.padding}")

            print(f"  Groups         : {layer.groups}")

            print(f"  Bias enabled   : {layer.bias is not None}")

            print(

                f"  Parameters     : "

                f"{count_parameters(layer):,}"

            )

            if layer.groups == layer.in_channels:

                print("  Type           : Depthwise convolution")

            elif layer.kernel_size == (1, 1):

                print("  Type           : Pointwise convolution")

            else:

                print("  Type           : Standard convolution")

# =========================================================

# FULLY CONNECTED LAYER DETAILS

# =========================================================

# Show the details of fully connected layers.
def print_fc_details(model):

    """

    Print input units, output units and parameter counts

    for every fully connected layer.

    """

    print("\nFULLY CONNECTED LAYERS")

    print("-" * 80)

    found = False

    for name, layer in model.named_modules():

        if isinstance(layer, nn.Linear):

            found = True

            print(f"\nLayer: {name}")

            print(f"  Input units : {layer.in_features:,}")

            print(f"  Output units: {layer.out_features:,}")

            print(f"  Weights     : {layer.in_features * layer.out_features:,}")

            print(f"  Biases      : {layer.out_features if layer.bias is not None else 0:,}")

            print(f"  Total       : {count_parameters(layer):,}")

    if not found:

        print("No fully connected layers found.")

# =========================================================

# OUTPUT SHAPE DETAILS

# =========================================================

# Run a sample image through the model and print each output shape.
def print_output_shapes(model):

    """

    Display the output tensor shape of each layer for a

    batch of one 64 x 64 RGB image.

    Models remain in evaluation mode.

    """

    print("\nOUTPUT SHAPES")

    print("-" * 80)

    handles = []

    def make_hook(name):

        def hook(module, inputs, output):

            if isinstance(output, torch.Tensor):

                print(f"{name:<35} {tuple(output.shape)}")

        return hook

    for name, module in model.named_modules():

        if name and len(list(module.children())) == 0:

            handles.append(

                module.register_forward_hook(make_hook(name))

            )

    dummy = torch.zeros(1, 3, 64, 64, device=DEVICE)

    with torch.inference_mode():

        model(dummy)

    for handle in handles:

        handle.remove()

# =========================================================

# COMPLETE MODEL REPORT

# =========================================================

# Print a full report for one model.
def model_report(model, name):

    total = count_parameters(model)

    trainable = count_parameters(model, trainable_only=True)

    non_trainable = total - trainable

    size = model_size_kb(model)

    print("\n\n" + "=" * 100)

    print(f"{name} - NETWORK ARCHITECTURE REPORT")

    print("=" * 100)

    print("\nGENERAL SUMMARY")

    print("-" * 60)

    print(f"Total parameters       : {total:,}")

    print(f"Trainable parameters   : {trainable:,}")

    print(f"Non-trainable parameters: {non_trainable:,}")

    print(f"Parameter storage      : {size:.2f} KiB")

    print(f"Parameter storage      : {size / 1024:.4f} MiB")

    print_layer_details(model)

    print_conv_details(model)

    print_fc_details(model)

    print_output_shapes(model)

# =========================================================

# GENERATE REPORTS FOR BOTH TRAINED MODELS

# =========================================================

# Compare the two already-trained models.
model_report(modelA, "MODEL A")

model_report(modelB, "MODEL B")

# =========================================================

# FINAL COMPARISON

# =========================================================

A_total = count_parameters(modelA)

B_total = count_parameters(modelB)

A_trainable = count_parameters(modelA, trainable_only=True)

B_trainable = count_parameters(modelB, trainable_only=True)

A_size = model_size_kb(modelA)

B_size = model_size_kb(modelB)

print("\n\n" + "=" * 100)

print("MODEL A VS MODEL B: FINAL COMPARISON")

print("=" * 100)

print(f"{'Metric':<35}{'Model A':>20}{'Model B':>20}")

print("-" * 75)

print(f"{'Total parameters':<35}{A_total:>20,}{B_total:>20,}")

print(f"{'Trainable parameters':<35}{A_trainable:>20,}{B_trainable:>20,}")

print(f"{'Parameter storage (KiB)':<35}{A_size:>20.2f}{B_size:>20.2f}")

if A_total > 0:

    reduction = (A_total - B_total) / A_total * 100

    print(f"\nParameter reduction in Model B: {reduction:.2f}%")

    if B_total < A_total:

        print("Model B uses fewer parameters than Model A.")

    elif B_total > A_total:

        print("Model B uses more parameters than Model A.")

    else:

        print("Both models have the same number of parameters.")

if A_size > 0:

    size_reduction = (A_size - B_size) / A_size * 100

    print(f"Parameter-storage reduction: {size_reduction:.2f}%")

print("\nAnalysis completed. No training was performed.")
