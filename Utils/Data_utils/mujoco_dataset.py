import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class MuJoCoDataset(Dataset):
    def __init__(
        self, 
        window=128, 
        num=30000, 
        dim=12, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        scalar=None,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(MuJoCoDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        
        self.window, self.var_num = window, dim
        self.auto_norm = neg_one_to_one
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)
        self.pred_len, self.missing_ratio = predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        self.rawdata, self.scaler = self._generate_random_trajectories(n_samples=num, seed=seed)
        if scalar is not None:
            self.scaler = scalar

        self.period, self.save2npy = period, save2npy
        self.samples = self.normalize(self.rawdata)
        self.sample_num = self.samples.shape[0]

        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e
        
        env = suite.load('hopper', 'stand')
        physics = env.physics

		# Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)

            for t in range(self.window):
                data[i, t, :self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2:] = physics.data.qvel
                physics.step()

		# Restore RNG.
        np.random.set_state(st0)

        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))
        return data, scaler

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        if self.save2npy:
            np.save(os.path.join(self.dir, f"mujoco_ground_truth_{self.window}_{self.period}.npy"), sq)

            if self.auto_norm:
                np.save(os.path.join(self.dir, f"mujoco_norm_truth_{self.window}_{self.period}.npy"), unnormalize_to_zero_to_one(data))
            else:
                np.save(os.path.join(self.dir, f"mujoco_norm_truth_{self.window}_{self.period}.npy"), data)

        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"mujoco_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
