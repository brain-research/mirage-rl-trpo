
ENV="Walker2d-v2"
BASELINE1="vh0"
BASELINE2="vh1"

for SEED in `seq 1001 1005`
do
  python main.py --env-name $ENV --gamma 0.99 --tau 1.00 --baseline $BASELINE1 --batch-size 25000 --n-epochs 250 --max-time 1000 --seed $SEED --log-file logs/${ENV}/${BASELINE1}_${SEED}.txt &
  python main.py --env-name $ENV --gamma 0.99 --tau 1.00 --baseline $BASELINE2 --batch-size 25000 --n-epochs 250 --max-time 1000 --seed $SEED --log-file logs/${ENV}/${BASELINE2}_${SEED}.txt &
done


