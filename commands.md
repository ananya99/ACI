eval:

python cambrian/main.py --eval example=detection env.renderer.render_modes='[human]' env.frame_skip=5

train: 

bash scripts/run.sh cambrian/main.py --train example=detection