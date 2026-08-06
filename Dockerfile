# GAVE2 - LaTIM submission
#
# Inference image: Tasks 1 and 2 end-to-end from raw images, and Task 3 given
# precomputed optic-disc masks (shipped in the Release for the validation set).
#
# Optic-disc segmentation itself is NOT in this image. It needs TensorFlow 2.13,
# which does not coexist cleanly with PyTorch 2.2 (CUDA runtime conflict). The
# script is provided separately; see README.md.
#
# Build:
#   docker build -t latim/gave2:1.0.0 .
#
# Run (GPU):
#   docker run --gpus all -v $PWD/data:/data -v $PWD/weights:/weights \
#              -v $PWD/work:/work latim/gave2:1.0.0 \
#              bash run_pipeline.sh /data /weights /work
#
# Run (CPU only - works, but roughly 20x slower):
#   docker run -v $PWD/data:/data -v $PWD/weights:/weights \
#              -v $PWD/work:/work latim/gave2:1.0.0 \
#              bash run_pipeline.sh /data /weights /work

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="GAVE2 - LaTIM submission"
LABEL org.opencontainers.image.description="Retinal artery/vein segmentation and biomarker quantification"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# libgl1 / libglib2.0-0 are required by OpenCV even in its headless build.
# git is not needed at runtime and is deliberately omitted.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so the layer is cached across code edits.
# torch/torchvision are already present in the base image at the pinned
# versions; installing them again from the CUDA index would be a ~2.5 GB no-op,
# so requirements-docker.txt omits them.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY model.py preprocessing.py transformations.py postprocessing.py \
     biomarkers.py infer.py final.py replay_task3.py pipeline.yaml \
     run_pipeline.sh ./

# Fail the build rather than ship a broken image: import every module and
# instantiate both architectures.
RUN python -c "\
import model, preprocessing, transformations, postprocessing, biomarkers, infer, final, replay_task3; \
m1 = model.build_model('task1'); m2 = model.build_model('task2'); \
print('smoke test OK:', type(m1).__name__, type(m2).__name__)"

CMD ["python", "-c", "print('GAVE2 LaTIM image. See README.md; entry point: run_pipeline.sh')"]
