#!/bin/bash

source "$(dirname "$0")/.env"

query=$1
directory=$2
number_of_images=$3

mkdir -p $directory

response=$(curl -s "https://pixabay.com/api/?key=$PIXBAY_API_KEY&q=$query&image_type=photo&per_page=$number_of_images")
urls=$(echo $response | jq -r '.hits[].largeImageURL')

echo "Query: $query"
echo "Directory: $directory"
count=1
for url in $urls
do
    wget -q -O $directory/image_$count.jpg $url
    echo "  Downloaded image $count"
    count=$((count+1))
done

echo ""