# Build with cuda-devel image
FROM nvidia/cuda:9.0-devel-ubuntu16.04

RUN apt-get update
RUN apt-get -y --no-install-recommends install \
	build-essential
RUN apt-get -y --no-install-recommends install \
	libomp-dev

COPY . /work

RUN make -C /work CUDA=1

# Install build artifact to cuda-runtime image
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get -y --no-install-recommends install \
	libomp5

RUN mkdir /work

COPY --from=0 /work/nbody.sh /work/
COPY --from=0 /work/nbody /work/
