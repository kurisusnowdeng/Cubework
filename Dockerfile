FROM cr-cn-beijing.volces.com/hpcaitech/zbian-dev:latest

WORKDIR /workspace

RUN cd apps/Cubework && git fetch --all && git reset --hard origin/main && pip install --upgrade --no-cache-dir -e .
