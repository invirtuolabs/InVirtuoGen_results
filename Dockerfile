FROM nvcr.io/nvidia/pytorch:24.04-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install the rest of the dependencies.
RUN pip install \
    datasets \
    transformers \
    dacite \
    pyyaml \
    numpy \
    packaging \
    safetensors \
    selfies \
    tqdm \
    dataclasses \
    rdkit \
    tokenizers \
    atomInSmiles \
    "jsonargparse[signatures]>=4.27.7" \
    lightning \
    torchdata \
    einops

# Copy project files into the container.
WORKDIR /app
COPY . /app

# Install the module as an editable package.
RUN pip install -e .
RUN pip install -U --no-dependencies pytdc==1.1.15 fuzzywuzzy huggingface-hub

