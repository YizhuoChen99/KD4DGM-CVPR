nohup python main.py --uid student \
--s-arch 32 \
--t-arch 128 \
--teacher-model ./CKPT/teacher-model.pth.tar \
> student.out 2>&1 &

