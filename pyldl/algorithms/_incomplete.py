from typing import Optional

import numpy as np

import tensorflow as tf
import keras
from qpsolvers import solve_qp
from scipy.spatial.distance import cdist
from pyldl.algorithms.utils import svt, proj
from pyldl.algorithms.base import BaseADMM, BaseIncomLDL
from sklearn.neighbors import NearestNeighbors

from pyldl.algorithms import LDSVR, SA_IIS, SA_BFGS, AA_KNN, AA_BP


class IncomLDL(BaseADMM, BaseIncomLDL):
    """:class:`IncomLDL <pyldl.algorithms.IncomLDL>` is proposed in paper :cite:`2017:xu`.
    """

    def _update_W(self):
        G = -np.eye(self._n_outputs, dtype=np.float64)
        h = np.zeros((self._n_outputs, 1), dtype=np.float64)
        A = np.ones((1, self._n_outputs), dtype=np.float64)
        b = np.array([1.], dtype=np.float64)

        M = np.zeros_like(self._y)

        for i in range(self._X.shape[0]):
            P = np.diag((1 + self._rho) * self._mask[i] + self._rho * (1 - self._mask[i])).astype(np.float64)
            ql = (self._V[i] - self._y[i] * self._mask[i] - self._rho * self._Z[i]) * self._mask[i]
            qr = (self._V[i] - self._rho * self._Z[i]) * (1 - self._mask[i])
            q = np.transpose(ql + qr).astype(np.float64)
            M[i] = solve_qp(P, q, G, h, A, b, solver='quadprog')

        self._W = np.linalg.pinv(np.transpose(self._X) @ self._X) @ np.transpose(self._X) @ M

    def _update_Z(self):
        A = self._X @ self._W + self._V / self._rho
        tau = self._alpha / self._rho
        self._Z = svt(A, tau)

    def fit(self, X, y, mask, alpha=1e-3, max_iterations=100, **kwargs):
        self._alpha = alpha
        return super().fit(X, y, mask=mask, max_iterations=max_iterations, **kwargs)

    def predict1(self, X):
        sa_bfgs = SA_BFGS()
        sa_bfgs.fit(self._X, self._X @ self._W)
        return sa_bfgs.predict(X)


class WInLDL(BaseADMM, BaseIncomLDL):
    """:class:`WInLDL <pyldl.algorithms.WInLDL>` is proposed in paper :cite:`2024:li`.
    """

    def _update_W(self):
        self._W = np.linalg.solve(
            np.transpose(self._X) @ self._X + 1e-5 * np.eye(self._n_features),
            np.transpose(self._X) @ (self._Z - self._V / self._rho)
        )

    def _update_Z(self):
        self._update_Q()
        Y = (self._X @ self._W) * (1 - self._mask) + self._y
        numerator = self._rho * self._X @ self._W + self._V + self._Q * self._Q * Y
        denominator = self._Q * self._Q + self._rho
        self._Z = proj(numerator / denominator)

    def _update_Q(self):
        a = 1 + self._current_iteration / self._max_iterations
        self._Q2 = np.power(a, np.tile(self._avg, (self._y.shape[0], 1))) * (1 - self._mask)
        self._Q = self._Q1 + self._Q2

    def _before_train(self):
        self._avg = np.sum(self._y, axis=0) / np.count_nonzero(self._y, axis=0)
        self._Q1 = np.exp2(1 - self._y) * self._mask

    def fit(self, X, y, mask, rho=2., **kwargs):
        return super().fit(X, y, mask=mask, rho=rho, **kwargs)

    def predict1(self, X):
        sa_bfgs = SA_BFGS()
        sa_bfgs.fit(self._X, self._X @ self._W)
        return sa_bfgs.predict(X)


class HidLDL(BaseIncomLDL):

    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.EPS_ERR = 1e-3
        self._olds = {}


    @property
    def params(self):
        return [self._Dg, self._Z]

    def _err(self):
        err = 0.
        for i, c in enumerate(self.params):
            err = np.maximum(err, np.abs(c - self._olds[i]).max())
        return err

    def _restore(self):
        for i, c in enumerate(self.params):
            self._olds[i] = c

    def _converged(self):
        return self._err() <= self.EPS_ERR


    def fit(self, X, y, mask, alpha=1e-3, sigma=1., rho=2., max_iterations=100):
        super().fit(X, y, mask=mask)
        self._alpha = alpha
        self._sigma = sigma
        self._rho2 = 2.
        self._rho = rho
        self._max_iterations = max_iterations
        self._before_train()
        self._train()

    def _train(self):
        for i in range(self._max_iterations):
            self._current_iteration = i + 1
            self._restore()
            self._update_Dg()
            self._update_Z()
            self._update_Z2()
            self._update_V()
            self._update_V2()
            if self._converged():
                print(f"Converged in {self._current_iteration} iterations.")
                break

    def _before_train(self):
      
        self._Z = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)
        self._V = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)
        self._V2 = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)

        self._Z2 = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)

        self._Dg = tf.Variable(self._y, trainable=True, dtype=tf.float32)

        self._nn = NearestNeighbors(n_neighbors=self._n_outputs + 1)
        self._nn.fit(self._X)
        graph = self._nn.kneighbors_graph()

        A = tf.exp(-(cdist(self._X, self._X) ** 2) / (2 * self._sigma ** 2))
        A = tf.cast(A * graph.toarray(), dtype=tf.float32)
        A_hat = tf.linalg.diag(tf.reduce_sum(A, axis=1))

        self._G = tf.cast(A_hat - A, dtype=tf.float32)
        self._Dg_optimizer = keras.optimizers.SGD()
        return self

    def _E_loss(self):

        lap = tf.linalg.trace(tf.transpose(self._Dg) @ self._G @ self._Dg)

        linear_loss = tf.linalg.trace(tf.transpose(self._V) @ (self._Dg - self._Z))

        qua_loss = 0.5 * self._rho * tf.reduce_sum((self._Dg - self._Z) ** 2)

        linear_loss2 = tf.linalg.trace(tf.transpose(self._V2) @ (self._Dg - self._Z2))

        qua_loss2 = 0.5 * self._rho * tf.reduce_sum((self._Dg - self._Z2) ** 2)

        return 0.5 *lap + linear_loss + qua_loss + linear_loss2 + qua_loss2

    def _update_V(self):
        self._V = self._V + self._rho * (self._Dg - self._Z)

    def _update_V2(self):
        self._V2 = self._V2 + self._rho2 * (self._Dg - self._Z2)

    def _update_Dg(self):
        with tf.GradientTape() as tape:
            E_loss = self._E_loss()
        E_gradients = tape.gradient(E_loss, self._Dg)
        self._Dg_optimizer.apply_gradients([(E_gradients, self._Dg)])
        self._Dg.assign(tf.maximum(self._Dg, 0))
        row_sums = tf.reduce_sum(self._Dg, axis=1)
        # 将行和为0的行设置为1/n
        n = self._Dg.shape[1]
        self._Dg.assign(tf.where(tf.expand_dims(row_sums, axis=1) == 0, tf.ones_like(self._Dg) / n, self._Dg))

        # 对所有行进行归一化
        row_sums = tf.reduce_sum(self._Dg, axis=1)
        self._Dg.assign(self._Dg / row_sums[:, tf.newaxis])

    def _update_Z(self):
        t1 = self._Dg + self._V / self._rho
        t1 = np.asarray(t1)
        M = np.zeros_like(self._y)
        for i in range(self._X.shape[0]):
            zero_indices = np.where(self._mask[i] == 0)[0]
            M[i, zero_indices] = t1[i, zero_indices]
            qr = - self._rho * self._Dg[i] - self._V[i]  # 一次项
            qr = np.asarray(qr)
            ql_scalar = 0.5 * self._rho
            # 创建一个与 qr 相同维度的全1向量
            ones_vector = np.ones_like(qr)
            # 将标量 ql 扩展成与 qr 相同维度的向量
            ql_vector = ql_scalar * ones_vector  # 二次项
            one_indices = np.where(self._mask[i] == 1)[0]
            y_one = self._y[i, one_indices]
            qr_one = qr[one_indices]
            ql_one = ql_vector[one_indices]
            b = np.sum(qr_one * y_one)
            a = np.sum(ql_one * y_one * y_one)
            k = - b / (2 * a) if a != 0 else 0
            M[i, one_indices] = k * y_one
        self._Z = M

    def _update_Z2(self):
        A = self._Dg + self._V2 / self._rho2
        tau = self._alpha / self._rho2
        self._Z2 = svt(A, tau)

    def predict(self, X):
        return self._Dg

    def predict1(self, X):
        sa_bfgs = SA_BFGS()
        sa_bfgs.fit(self._X, np.asarray(self._Dg))
        return sa_bfgs.predict(X)


class HidLDL_noCons(BaseIncomLDL):
    
    #for ablation test

    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.EPS_ABS = 1e-4
        self.EPS_REL = 1e-3
        self.EPS_ERR = 1e-3
        self._olds = {}

    @property
    def constraint(self):
        return [[self._Z, self._Dg]]

    @property
    def params(self):
        return [self._Dg, self._Z]

    @property
    def Vs(self):
        return [self._V]

    def _primal_residual(self):
        return np.array([np.linalg.norm(c[0] - c[1], 'fro') for c in self.constraint])

    def _dual_residual(self):
        return np.array(
            [np.linalg.norm(self._rho * (c[0] - self._olds[i]), 'fro') for i, c in enumerate(self.constraint)])

    def _primal_eps(self):
        return np.sqrt(self._X.shape[0]) * self.EPS_ABS + self.EPS_REL * np.array([np.maximum(
            np.linalg.norm(c[0], 'fro'), np.linalg.norm(c[1], 'fro')
        ) for c in self.constraint])

    def _dual_eps(self):
        return np.sqrt(self._n_outputs) * self.EPS_ABS + self.EPS_REL * np.array([
            np.linalg.norm(v, 'fro') for v in self.Vs
        ])

    def _err(self):
        err = 0.
        for i, c in enumerate(self.params):
            err = np.maximum(err, np.abs(c - self._olds[i]).max())
        return err

    def _restore2(self):
        for i, c in enumerate(self.params):
            self._olds[i] = c

    def _converged2(self):
        return self._err() <= self.EPS_ERR

    def _restore(self):
        for i, c in enumerate(self.constraint):
            self._olds[i] = c[0]

    def _converged(self):
        return np.all(self._primal_residual() <= self._primal_eps()) and np.all(
            self._dual_residual() <= self._dual_eps())

    def fit(self, X, y, mask, alpha=1e-3, sigma=1., rho=1., max_iterations=100):
        super().fit(X, y, mask=mask)
        self._alpha = alpha
        self._sigma = sigma
        self._rho = rho
        self._max_iterations = max_iterations
        self._before_train()
        self._train()

    def _train(self):
        for i in range(self._max_iterations):
            self._current_iteration = i + 1
            self._restore()
            self._update_Dg()
            self._update_Z()
            self._update_V()
            if self._converged():
                print(f"Converged in {self._current_iteration} iterations.")
                break

    def _before_train(self):
        #ADMM初始化
        # 创建形状为 (self._X.shape[0], self._n_outputs) 的 float32 类型数组
        self._Z = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)
        self._V = np.ones((self._X.shape[0], self._n_outputs), dtype=np.float32)

        self._Dg = tf.Variable(self._Z, trainable=True, dtype=tf.float32)

        self._nn = NearestNeighbors(n_neighbors=self._n_outputs + 1)
        self._nn.fit(self._X)
        graph = self._nn.kneighbors_graph()

        A = tf.exp(-(cdist(self._X, self._X) ** 2) / (2 * self._sigma ** 2))
        A = tf.cast(A * graph.toarray(), dtype=tf.float32)
        A_hat = tf.linalg.diag(tf.reduce_sum(A, axis=1))

        self._G = tf.cast(A_hat - A, dtype=tf.float32)
        self._Dg_optimizer = keras.optimizers.SGD()
        return self

    def _E_loss(self):

        lap = tf.linalg.trace(tf.transpose(self._Dg) @ self._G @ self._Dg)

        linear_loss = tf.linalg.trace(tf.transpose(self._V) @ (self._Dg - self._Z))

        qua_loss = 0.5 * self._rho * tf.reduce_sum((self._Dg - self._Z) ** 2)

        return 0.5 * lap + linear_loss + qua_loss

    def _update_Dg(self):
        with tf.GradientTape() as tape:
            E_loss = self._E_loss()
        E_gradients = tape.gradient(E_loss, self._Dg)
        self._Dg_optimizer.apply_gradients([(E_gradients, self._Dg)])
        self._Dg.assign(tf.maximum(self._Dg, 0))
        row_sums = tf.reduce_sum(self._Dg, axis=1)
        # 将行和为0的行设置为1/n
        n = self._Dg.shape[1]
        self._Dg.assign(tf.where(tf.expand_dims(row_sums, axis=1) == 0, tf.ones_like(self._Dg) / n, self._Dg))

        # 对所有行进行归一化
        row_sums = tf.reduce_sum(self._Dg, axis=1)
        self._Dg.assign(self._Dg / row_sums[:, tf.newaxis])

    def _update_V(self):
        self._V = self._V + self._rho * (self._Dg - self._Z)

    def _update_Z(self):
        A = self._Dg + self._V / self._rho
        tau = self._alpha / self._rho
        self._Z = svt(A, tau)

    def predict(self, X):
        return self._Dg

    def predict1(self, X):
        sa_bfgs = SA_BFGS()
        sa_bfgs.fit(self._X, np.asarray(self._Dg))
        return sa_bfgs.predict(X)
