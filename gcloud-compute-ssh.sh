#! /bin/sh
host="$1"
zone="europe-west4-a"
shift
exec gcloud compute ssh --zone="$zone" "$host" -- "$@"
