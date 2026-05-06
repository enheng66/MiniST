# MiniST
## 1. Dataset
In our experiments, we use the SD, GBA, GLA, and CA datasets, with the data spanning the entire year of 2019. You can download the datasets from the [LargeST](https://github.com/liuxu77/LargeST/blob/main) repository. After downloading the archive.zip file, place it in the `MiniST/data/ca` directory and unzip it.

First, you should navigate to the `MiniST/data/ca` directory and run the Jupyter notebook file `process_ca_his.ipynb` to generate the cleaned traffic flow data. Then, go to the `data` directory and execute the `generate_data_for_training.py` script using a command like 
```
python generate_data_for_training.py --dataset ca --years 2019
```
to prepare the data for model training.

To generate the SD sub-dataset, please first run through all the cells in the provided Jupyter notebook generate_sd_dataset.ipynb located in the MiniST/data/sd folder. Then, use the following command to generate traffic flow data for model training:
```
python generate_data_for_training.py --dataset sd --years 2019
```
You can use the same procedure to generate the GBA and GLA datasets.


## 2.Model Running
To run  the model, you may enter the folder MiniST and directly execute the Python file in the terminal.
short-term forecasting:
```
python experiments/miniST/main.py --device cuda:0 --dataset sd --years 2019 --model_name miniST  --bs 32 --max_epochs 200 --seq_len 720 --horizon 12
```
long-term forecasting:
 ```
python experiments/miniST/main.py --device cuda:0 --dataset ca --years 2019 --model_name miniST  --bs 32 --max_epochs 200 --seq_len 720 --horizon 96
```
