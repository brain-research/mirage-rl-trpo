python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1338 --control-variate-lr 1e-3 --log-file logs/v-lbfgs-y-disc-1338_0.json --v-optimizer lbfgs --use-disc-avg-v --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1338 --control-variate-lr 1e-3 --log-file logs/v-adam-y-disc-1338_1.json --v-optimizer adam --use-disc-avg-v --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1338 --control-variate-lr 1e-3 --log-file logs/v-lbfgs-n-disc-1338_2.json --v-optimizer lbfgs --batch-size 4000 &
python main.py --env-name "HalfCheetah-v1" --gamma 0.99 --tau 0.95 --baseline v --n-epochs 3000 --max-time 1000 --seed 1338 --control-variate-lr 1e-3 --log-file logs/v-adam-n-disc-1338_3.json --v-optimizer adam --batch-size 4000 &


