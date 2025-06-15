cd ./evaluation

MODEL=gpt-4o  # Use gpt-4o (gpt-4o-2024-08-06) by default
API_KEY=sk-xxx  # YOUR_OPENAI_API_KEY

python run_evaluation.py \
  --submission_file path/to/submission.json \
  --metadata_file path/to/metadata.json \
  --output_dir ./results \
  --model $MODEL \
  --api_key $API_KEY
