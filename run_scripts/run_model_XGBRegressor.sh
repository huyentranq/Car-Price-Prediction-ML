echo -e "\n--- [C] KỊCH BẢN: XGBRegressor - Không TUNING---"
python main_script.py \
    --model_name "XGBRegressor" \
    --tuner 0 \
    --run_id "005" \
    --savedir_models "outputs/models" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         


echo -e "\n--- [C] KỊCH BẢN: XGBRegressor - Có TUNING---"
python main_script.py \
    --model_name "XGBRegressor" \
    --tuner 1 \
    --run_id "006" \
    --savedir_models "outputs/models" \
  