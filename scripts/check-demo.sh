#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
compose() { docker compose -f docker-compose.yml -f compose.demo.yml "$@"; }
if [ -z "${GIT_COMMIT:-}" ]; then
    GIT_COMMIT=$(git rev-parse HEAD)
    if [ -n "$(git status --porcelain)" ]; then
        GIT_COMMIT="${GIT_COMMIT}-dirty"
    fi
fi
export GIT_COMMIT
compose config --quiet
compose build mlflow train api test verify
compose run --rm --no-deps test
compose up -d --wait mlflow
compose run --rm train
compose up -d --wait api
compose run --rm verify
before=$(compose exec -T api python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health/ready'))['run_id'])")
compose up -d --force-recreate --wait mlflow api
compose run --rm verify
after=$(compose exec -T api python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health/ready'))['run_id'])")
test "$before" = "$after"
printf '%s\n' 'PASS: container integration and persistence (synthetic demo).'
