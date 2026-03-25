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