# diff_tim
example of training for electricity data (on datatype you can select also traffic or lorenz if you want to try with other data types but you will also have to change the pkl files content
pip install -r requirements.txt
!python exe_forecasting.py \
  --datatype electricity \
  --modelfolder "" \
  --data_pkl_path ./data/electricity_nips/data.pkl \
  --meanstd_pkl_path ./data/electricity_nips/meanstd.pkl \
  --history_len 512 \
  --pred_len 64 \
  --epochs 50 \
  --batch_size 8 \
  --nsample 20 \
  --eval_batch_size 1

  
  

