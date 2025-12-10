echo -e "\n--- [B] KỊCH BẢN CƠ SỞ: LGBMRegressor - Không TUNING ---"
python main_script.py \
    --model_name "LGBMRegressor" \
    --tuner 0 \
    --run_id "003" \
    --savedir_models "outputs/models" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     


echo -e "\n--- [B] KỊCH BẢN CƠ SỞ: LGBMRegressor - Có TUNING ---"
python main_script.py \
    --model_name "LGBMRegressor" \
    --tuner 1 \
    --run_id "004" \
    --savedir_models "outputs/models" \
