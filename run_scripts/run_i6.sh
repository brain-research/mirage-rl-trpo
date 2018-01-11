python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1339 --control-variate-lr 1e-3 --log-file logs/v-lbfgs-y-disc-1339_0.json --v-optimizer lbfgs --use-disc-avg-v --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1339 --control-variate-lr 1e-3 --log-file logs/v-adam-y-disc-1339_1.json --v-optimizer adam --use-disc-avg-v --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1339 --control-variate-lr 1e-3 --log-file logs/v-lbfgs-n-disc-1339_2.json --v-optimizer lbfgs --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1339 --control-variate-lr 1e-3 --log-file logs/v-adam-n-disc-1339_3.json --v-optimizer adam --batch-size 4000 &

