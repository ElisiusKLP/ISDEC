import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from typing import Any


class TrainState(train_state.TrainState):
    batch_stats: Any


class FlaxTrainer:
    """Minimal Flax training helper for classification models.

    Keeps a `TrainState` with `params` and `batch_stats` and exposes
    small APIs used by the sklearn-like adapter below.
    """

    def __init__(self, model: nn.Module, input_shape, learning_rate: float = 1e-3, seed: int = 734):
        self.model = model
        self.learning_rate = float(learning_rate)
        self.rng = jax.random.PRNGKey(int(seed))

        dummy_x = jnp.zeros((1,) + tuple(input_shape))
        variables = self.model.init(self.rng, dummy_x, train=True)
        params = variables.get("params")
        batch_stats = variables.get("batch_stats", {})

        tx = optax.adam(self.learning_rate)
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, batch_stats=batch_stats
        )

        def loss_fn(params, batch_stats, x, y, rng):
            vars = {"params": params, "batch_stats": batch_stats}
            (logits, new_vars) = self.model.apply(
                vars, x, train=True, mutable=["batch_stats"], rngs={"dropout": rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss, new_vars

        def _train_step(state, x, y, rng):
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, new_vars), grads = grad_fn(state.params, state.batch_stats, x, y, rng)
            state = state.apply_gradients(grads=grads)
            # new_vars is a dict like {'batch_stats': {...}}
            state = state.replace(batch_stats=new_vars.get("batch_stats", {}))
            return state, loss

        self._train_step = jax.jit(_train_step)

    def train_on_batch(self, x: jnp.ndarray, y: jnp.ndarray):
        print(f"Training on x shape {x.shape} y shape {y.shape}")
        self.rng, step_rng = jax.random.split(self.rng)
        self.state, loss = self._train_step(self.state, x, y, step_rng)
        return float(loss)

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        print(f"Predicting on x shape {x.shape}")
        vars = {"params": self.state.params, "batch_stats": self.state.batch_stats}
        out = self.model.apply(vars, x, train=False, mutable=False)
        # model.apply may return logits or (logits, aux); handle both
        logits = out[0] if isinstance(out, tuple) else out
        return jnp.argmax(logits, axis=-1)


class FlaxSKLearnLikeModel:
    """Very small sklearn-like wrapper around `FlaxTrainer`.

    Provides `fit(X, y)` and `predict(X)` so it can be used where a
    simple `.fit`/`.predict` model object is expected.
    """

    def __init__(self, model: nn.Module, input_shape, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
        self.trainer = FlaxTrainer(model, input_shape, learning_rate=lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

    def _iter_minibatches(self, Xj: jnp.ndarray, yj: jnp.ndarray):
        n = Xj.shape[0]
        self.trainer.rng, perm_rng = jax.random.split(self.trainer.rng)
        perm = np.array(jax.random.permutation(perm_rng, n), dtype=int)
        for i in range(0, n, self.batch_size):
            idx = perm[i : i + self.batch_size]
            yield Xj[idx], yj[idx]

    def train_epoch(self, Xj: jnp.ndarray, yj: jnp.ndarray):
        for xb, yb in self._iter_minibatches(Xj, yj):
            self.trainer.train_on_batch(xb, yb)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xj = jnp.array(X)
        yj = jnp.array(y).astype(jnp.int32)
        for _ in range(self.epochs):
            self.train_epoch(Xj, yj)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xj = jnp.array(X)
        n = Xj.shape[0]
        preds = []
        for i in range(0, n, self.batch_size):
            xb = Xj[i : i + self.batch_size]
            p = self.trainer.predict(xb)
            preds.append(np.array(p))
        return np.concatenate(preds)


__all__ = ["FlaxTrainer", "FlaxSKLearnLikeModel"]
