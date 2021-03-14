set -ex
python main.py --exper_name "secure-deep-hiding" --epochs 80 --use_key --num_secrets 1 --channel_secret 3 --channel_key 3
