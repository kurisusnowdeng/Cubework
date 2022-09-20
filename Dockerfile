FROM cr-cn-beijing.volces.com/hpcaitech/zbian-dev:latest

WORKDIR /workspace

RUN sed -i '$d' ~/.bashrc && \
    sed -i '$d' ~/.bashrc && \
    sed -i '$d' ~/.bashrc && \
    sed -i '$d' ~/.bashrc && \
    sed -i '$d' ~/.bashrc && \
    sed -i '$d' ~/.bashrc

RUN echo 'alias cdp="cd /nas/code/zbian"' >> ~/.bashrc && \
    echo 'alias cds="cd /nas/data"' >> ~/.bashrc && \
    echo 'alias proxy-start="export http_proxy=http://100.68.174.164:3128 https_proxy=http://100.68.174.164:3128"'  >> ~/.bashrc && \
    echo 'alias proxy-stop="unset http_proxy https_proxy"' >> ~/.bashrc && \
    echo 'export proxy_addr=http://100.68.174.164:3128'

RUN apt-get update && apt-get install -y htop tmux

RUN git config --global http.proxy http://100.68.174.164:3128 && git config --global https.proxy http://100.68.174.164:3128
