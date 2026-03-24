FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir '.[proxy]'

FROM python:3.12-slim

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/routellect /usr/local/bin/routellect
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

WORKDIR /app
COPY src/ src/

ENV ROUTELLECT_PROXY_HOST=0.0.0.0
ENV ROUTELLECT_PROXY_PORT=11411

EXPOSE 11411

VOLUME /root/.routellect

ENTRYPOINT ["routellect"]
