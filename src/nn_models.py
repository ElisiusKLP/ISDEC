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

        # -----------------------------------
        # Separable Convolution
        # -----------------------------------

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
__all__ = ["EEGNet"]
