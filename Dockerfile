# FAMLI2
#
# VERSION               golob/famli2:2.0.0.pre


FROM --platform=amd64 debian:bookworm-slim

RUN export TZ=Etc/UTC
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get update && \
apt-get -y install tzdata && \
apt-get install -y \
    pigz \
    python3-pip \
&& apt-get clean \
&& apt-get purge \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD . /src/

RUN cd /src/ && \
pip3 install --break-system-packages . && \
cd /root/

WORKDIR /root/