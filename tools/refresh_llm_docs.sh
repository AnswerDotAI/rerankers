#!/bin/bash

echo "Refreshing LLM documentation files..."

echo "Generating API list documentation..."
pysym2md rerankers --output_file apilist.txt

echo "Generating context files..."
llms_txt2ctx llms.txt > llms-ctx.txt
llms_txt2ctx llms.txt --optional True > llms-ctx-full.txt

echo "✅ Documentation refresh complete!"
