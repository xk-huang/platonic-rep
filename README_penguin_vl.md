git clone https://github.com/minyoungg/platonic-rep.git
cd platonic-rep

python -m venv .venv
source .venv/bin/activate


pip install --upgrade pip
pip install uv 
uv pip install -r requirements transformers==4.51.3


pip install packaging
pip install psutil
pip install ninja
pip install whell

pip install flash-attn==2.8.3 --no-build-isolation


python export_hf_images.py --num-images 10000 --output-dir ../tmp/extracted_images

python compare_cknna_penguin_siglip2.py \
  --images ../tmp/extracted_images \
  --out results/penguin_vs_siglip2 \
  --max-images 10000 \
  --batch-size 32 \
  --topk 10 \
  --drop-embedding-layer