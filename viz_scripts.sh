dvalues=(10 20 30 40 50 60 70 80 90 100 110)
train_sizes=(1000 2000 3000 4000 5000 10000 20000 30000 40000 50000)

for d in ${dvalues[@]}; do
  echo "$d, 20000"
  python main.py --mode train --model ae --r 10 --d $d --train_size 20000 --epochs 20 --seed 0
  python main.py --mode test --model ae --r 10 --d $d --dwn_mode cls --dwn_model svm --seed 0
  python main.py --mode test --model ae --r 10 --d $d --dwn_mode reg --dwn_model linear --seed 0
done

for train_size in ${train_sizes[@]}; do
  echo "40, $train_size"
  python main.py --mode train --model ae --r 10 --d 40 --train_size $train_size --epochs 20 --seed 0
  python main.py --mode test --model ae --r 10 --d 40 --dwn_mode cls --dwn_model svm --seed 0
  python main.py --mode test --model ae --r 10 --d 40 --dwn_mode reg --dwn_model linear --seed 0
done
