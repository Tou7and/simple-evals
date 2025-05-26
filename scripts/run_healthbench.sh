#!/bin/bash
set -e

# Read Azure Key in env file
if [ -f .env ]; then
  source .env
else
  echo ".env file not found!"
  exit 1
fi

# python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1-nano --example 10 
# python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4o --example 10 
python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1 --example 10 

