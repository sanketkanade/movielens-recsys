# Reproduces the whole experiment: download -> split -> tune -> test -> results/.
#
#   docker build -t movielens-recsys .
#   docker run --rm -v "$PWD/results:/app/results" movielens-recsys
#
# Same Python minor version and pinned library versions as local development.
FROM python:3.12-slim

# Unbuffered output so training progress streams live; no pip cache in the image.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY download_data.py run.py ./
COPY recsys/ recsys/
COPY tests/ tests/

# The build fails if the metric/model unit tests fail.
RUN python -m pytest -q tests

# The dataset is downloaded at run time (MovieLens may not be redistributed,
# so it is never baked into the image or committed to git).
CMD ["python", "run.py"]
