import torch
import torch.nn as nn
import json
import logging
import random
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_all_seeds(seed: int = 42):
    """Set seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Load configuration
with open('fedmriapp/models/model_config.json') as f:
    model_config = json.load(f)

class CustomCNN(nn.Module):
    def __init__(self, seed: int = 42):
        super(CustomCNN, self).__init__()
        
        # Set seed for reproducible initialization
        set_all_seeds(seed)
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Calculate the size of flattened features
        self.flatten_size = self._get_flatten_size(model_config['imgHeight'], model_config['imgWidth'])
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights with fixed seed
        self._initialize_weights(seed)
        logger.info(f"CustomCNN initialized with seed {seed}")
        
    def _get_flatten_size(self, height, width):
        h = height // 8
        w = width // 8
        return 32 * h * w
    
    def _initialize_weights(self, seed: int):
        set_all_seeds(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CustomCNNLight(nn.Module):
    def __init__(self, seed: int = 42):
        super(CustomCNNLight, self).__init__()
        
        # Set seed for reproducible initialization
        set_all_seeds(seed)
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Calculate the size of flattened features
        self.flatten_size = self._get_flatten_size(model_config['imgHeight'], model_config['imgWidth'])

        # Simplified Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights with fixed seed
        self._initialize_weights(seed)
        logger.info(f"CustomCNNLight initialized with seed {seed}")
    
    def _get_flatten_size(self, height, width):
        h = height // 4
        w = width // 4
        return 16 * h * w
    
    def _initialize_weights(self, seed: int):
        set_all_seeds(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class TorchModel(nn.Module):
    def __init__(self, input_shape=(244, 244, 1), num_classes=4, seed=42):
        super(TorchModel, self).__init__()
        
        # In PyTorch, input shape is expected as (channels, height, width)
        # So we need to adapt from TF's (height, width, channels)
        in_channels = input_shape[2]
        
        # Convolutional layers
        self.features = nn.Sequential(
            # First conv block: 32 filters of 3x3, followed by ReLU and MaxPool
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block: 64 filters of 3x3, followed by ReLU and MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block: 64 filters of 3x3, followed by ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
        )
        
        # Adding a functional forward pass to correctly compute the feature dimensions
        self.flattened_size = self._get_conv_output(input_shape)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
    
    def _get_conv_output(self, shape):
        # Create a dummy input tensor
        if len(shape) == 3:  # If shape is (H, W, C)
            # Convert to (C, H, W) for PyTorch
            dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        else:  # If shape is already (C, H, W)
            dummy_input = torch.zeros(1, *shape)
            
        # Forward pass through features to get output size
        output = self.features(dummy_input)
        n_size = output.data.view(1, -1).size(1)
        return n_size
    
    def forward(self, x):
        # PyTorch uses (batch_size, channels, height, width) format
        # If input is in TF format (batch_size, height, width, channels),
        # we need to permute it
        if x.ndim == 4 and x.shape[1] == x.shape[3]:  # If channels is the last dimension
            x = x.permute(0, 3, 1, 2)
            
        x = self.features(x)
        x = self.classifier(x)
        return x

# class EnhancedBrainModel(nn.Module):
#     def __init__(self, img_height=model_config['imgHeight'], img_width=model_config['imgWidth'], num_classes=4, seed=42):
#         super(EnhancedBrainModel, self).__init__()
#         set_all_seeds(seed)
        
#         # Enhanced Convolutional Blocks
#         self.conv_blocks = nn.Sequential(
#             # Block 1 - Capture basic textures
#             nn.Sequential(
#                 nn.Conv2d(1, 32, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Dropout(0.25)
#             ),
            
#             # Block 2 - Detect complex patterns
#             nn.Sequential(
#                 nn.Conv2d(32, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Dropout(0.4)
#             ),
            
#             # Block 3 - Identify tumor-specific features
#             nn.Sequential(
#                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                 #nn.BatchNorm2d(128),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Dropout(0.5)
#             ),
            
#             # Block 4 - Final feature refinement
#             nn.Sequential(
#                 nn.Conv2d(128, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Dropout(0.5)
#             )
#         )

#         # Calculate flattened features size
#         self.flatten_size = self._calculate_flatten_size(img_height, img_width)
        
#         # Enhanced Classifier with residual-like connections
#         self.classifier = nn.Sequential(
#             nn.Linear(self.flatten_size, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#         # Initialize weights
#         self._initialize_weights(seed)

#     def _calculate_flatten_size(self, height, width):
#         # Each MaxPool2d(2) reduces spatial dimensions by half (4 blocks → 2^4 = 16)
#         h = height // 16
#         w = width // 16
#         return 256 * h * w

#     def _initialize_weights(self, seed):
#         set_all_seeds(seed)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv_blocks(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

def get_custom_model(seed: int = 42):
    """
    Creates and returns a custom model with reproducible initialization.
    
    Args:
        seed (int): Seed for reproducible initialization. Default is 42.
    
    Returns:
        CustomCNNLight: Initialized model
    """
    try:
        # Set seeds for reproducible model creation
        # set_all_seeds(seed)
        
        # Create model
        #model = TorchModel(seed=seed)
        model = CustomCNNLight(seed=seed)
        
        # Move to device
        device = torch.device(model_config['device'])
        model = model.to(device)
        
        logger.info(f"Successfully created custom model with seed {seed}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create custom model: {str(e)}")
        raise

# Set initial seeds when module is imported
set_all_seeds(42)