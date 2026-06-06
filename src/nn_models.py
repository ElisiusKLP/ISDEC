# Note: Neural networks were implemented, but not used in the final analysis.
from flax import linen as nn

class EEGNet(nn.Module):
    n_classes: int
    channels: int
    samples: int

    F1: int = 8
    D: int = 2
    F2: int = 16
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, train=True):

        # -----------------------------------
        # Block 1: Temporal Convolution
        # -----------------------------------

        x = nn.Conv(
            features=self.F1,
            kernel_size=(1, 64),
            padding='SAME',
            use_bias=False
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)

        # -----------------------------------
        # Spatial Depthwise Convolution
        # -----------------------------------

        x = nn.Conv(
            features=self.F1 * self.D,
            kernel_size=(self.channels, 1),
            feature_group_count=self.F1,
            use_bias=False
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.elu(x)

        x = nn.avg_pool(
            x,
            window_shape=(1, 4),
            strides=(1, 4)
        )

        x = nn.Dropout(rate=self.dropout_rate)(
            x,
            deterministic=not train
        )

        # Separable Convolution
        x = nn.Conv(
            features=self.F2,
            kernel_size=(1, 16),
            padding='SAME',
            use_bias=False
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.elu(x)

        x = nn.avg_pool(
            x,
            window_shape=(1, 8),
            strides=(1, 8)
        )

        x = nn.Dropout(rate=self.dropout_rate)(
            x,
            deterministic=not train
        )

        # -----------------------------------
        # Classification
        # -----------------------------------

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.n_classes)(x)

        return x

class CNN(nn.Module):
    n_classes: int
    resolution: int
    @nn.compact
    def __call__(self, x, train=True):
        

        # Expect input `x` with shape (batch, H, W, C), default H=W=256
        # First hidden layer: Conv2D (3x3) -> ReLU -> MaxPool(2x2)
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            padding='VALID',
            use_bias=True
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='VALID'
        )

        # Second hidden layer: Conv2D (4x4) -> ReLU -> MaxPool(2x2)
        x = nn.Conv(
            features=216,
            kernel_size=(4, 4),
            padding='VALID',
            use_bias=True
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='VALID'
        )

        # Flatten and fully-connected layers
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(48)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        # Output layer: sigmoid for binary, softmax for multiclass
        if self.n_classes == 1:
            x = nn.Dense(1)(x)
            x = nn.sigmoid(x)
        else:
            x = nn.Dense(self.n_classes)(x)
            x = nn.softmax(x)

        return x
__all__ = ["EEGNet", "CNN"]
