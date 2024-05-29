## 🚄 Get Started

Please download **dataset.zip**, then unzip and copy it to the location indicated by '`Place the dataset here`'.

> 🔔 **dataset.zip** can be downloaded [here](https://drive.google.com/file/d/11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6/view?usp=sharing).

## 🔨 Build Dataloader

### 🕶️ Pipeline Overview

The figure below shows everything that happens after calling *build_dataloader*

<div align="center">
<img src="../figures/Flowchart.svg" width=100% />
</div>

Below we show the specific meaning of the saved file:

  - `{name}_norm_truth_{length}_train.npy` - The saved data is used for training model, which has been normalized into [0, 1].
  - `{name}_norm_truth_{length}_test.npy` - The saved data is used for inference, which has been normalized into [0, 1].
  - `{name}_ground_truth_{length}_train.npy` - The saved data is used for training model, however, it is raw data that has not been normalized.
  - `{name}_ground_truth_{length}_test.npy` - The saved data is used for inference, however, it is raw data that has not been normalized.
  - `{name}_masking_{length}.npy` - The saved mask-seqences indicate the generation target of imputation or forecasting.

> 🔔 Note that the generated time series `ddpm_fake_{name}.npy` or `ddpm_{mode}_{name}.npy` are also normalized into [0, 1]. You can restore them by adding following codes in **main.py**:
```
line 86: samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
line 93: samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
```

### 📝 Custom Dataset

Real-world sequences (or any self-prepared sequences) need to be configured via the following:

* Create and check the settings in your **.yaml file** and modify it such as *seq_length* and *feature_size* etc. if necessary.
* Convert the real-world time series into **.csv file**, then put it to the repo like our template datasets.
* Make sure that non-numeric rows and columns are not included in the training data, or you may need to modify codes in **./Utils/Data_utils/real_datasets.py**.
  - Remove the header if it exists:
    ```
    line 132: df = pd.read_csv(filepath, header=0)
    # set `header=None` if it does not exsit.
    ```
  - Delete rows and columns, here using the first column as an example:
    ```
    line 133: df.drop(df.columns[0], axis=1, inplace=True)
    ```

> 🔔 Please set `use_ff=False` at line 54 in **./Models/interpretable_diffusion/gaussian_diffusion.py** if your temporal data is highly irregular.