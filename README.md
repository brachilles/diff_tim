# diff_tim
example of unguided sampling for model trained on traffic data  

pip install -r requirements.txt
!python exe_forecasting.py \
  --datatype traffic \
  --modelfolder "" \
  --data_pkl_path ./data/electricity_nips/data.pkl \
  --meanstd_pkl_path ./data/electricity_nips/meanstd.pkl \
  --history_len 512 \
  --pred_len 64 \
  --epochs 50 \
  --batch_size 8 \
  --nsample 20 \
  --eval_batch_size 1

  
  

