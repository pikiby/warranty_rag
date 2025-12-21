#!/bin/bash

# Минимальный скрипт запуска для минимального RAG-приложения.

set -e

mkdir -p data/chroma docs
docker-compose up --build