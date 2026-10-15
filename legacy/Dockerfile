# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
#
# It is the same image your development session runs, and it supports both
# gfx942 (MI300X, CDNA) and gfx1100 (Navi 31, RDNA3), so one submission runs on
# whichever GPU you are given. It ships torch and PIL ONLY; add whatever else
# you need to requirements.txt.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# The vision reader runs a real OCR binary as a child process, and the
# evaluation container has no network, so the engine must be *in* the image --
# otherwise every image in the corpus comes back VISION_UNAVAILABLE and the
# questions whose answer is only in a picture are unanswerable.
#
# apt rather than a Python package because ocr_engine.py resolves the engine
# through PATH and execs it. It looks for `tesseract` by default (config
# DEFAULT_OCR_ENGINE), so installing it here is the whole wiring: no
# MC3_TESSERACT override is needed inside the image, and the env var stays free
# for a development machine that has the engine unpacked somewhere unusual.
#
# Placed above the weights copy so that it caches. Editing a Python file reuses
# this layer, and it only invalidates when the engine itself changes.
#
# --no-install-recommends avoids the X11, extra image-format and additional
# language packs the default set pulls in; `-eng` is the only language the
# corpus needs (config OCR_LANGUAGE). The version check makes a broken install a
# build failure rather than a corpus-wide VISION_UNAVAILABLE discovered at
# scoring time. The lists are dropped in this same layer so they do not persist
# into the image and count against the size limit.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
 && tesseract --version \
 && rm -rf /var/lib/apt/lists/*

# ⚠️ SHIP YOUR WEIGHTS IN THE IMAGE. Your container has NO outbound network
# access during evaluation -- it runs on an internal Docker network with no
# route off the box. Code that downloads a model at evaluation time will hang
# and then time out, scoring zero, no matter how good the model is.
#
# Test that assumption the way the harness enforces it:
#
#     docker run --rm --network none your-image:tag python3 /app/app.py --index /app/corpus
#
# Weights count against the 60 GiB uncompressed image limit.
#
# Copied before the application code on purpose. Docker caches a layer until one
# of its inputs changes, so putting the weights first means editing a Python
# file reuses this 129 MB layer instead of re-copying it on every rebuild.
COPY models/ /models/

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/

# The harness invokes this script TWICE in different shapes. Once to index,
# before any graded clock:
#
#     python3 /app/app.py --index /app/corpus
#
# then once per question, ten times:
#
#     python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
#
# reading /app/output/query_01_output.json after each. The corpus is copied in
# for you at evaluation time; both directories are created if missing, but
# creating them here costs nothing and makes local testing behave identically.
RUN mkdir -p /app/corpus /app/output

# No service, no port, no entrypoint script. The container only has to stay up
# long enough to be exec'd into, which is what this does.
CMD ["sleep", "infinity"]
