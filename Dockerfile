FROM cr-cn-beijing.volces.com/hpcaitech/hyadis-dev:latest

WORKDIR apps

RUN cd ./ColossalAI && git remote set-url origin https://github.com/kurisusnowdeng/ColossalAI.git && \
    git fetch --all && git checkout main && git reset --hard origin/main

RUN git clone https://github.com/kurisusnowdeng/Cubework.git && cd ./Cubework && \
    pip install --upgrade --verbose --no-cache-dir -e .

RUN git clone https://github.com/hpcaitech/HyaDIS.git && cd ./HyaDIS && git checkout dev &&\
    pip install --upgrade --verbose --no-cache-dir -e .
